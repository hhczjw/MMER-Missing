"""GPU1 lock-controlled small-model verification, isolated from training RNG."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/r1_aligned_20261009'))
from plan import BASE, read, write
from numerical_guard import install


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Sequential(nn.Linear(4,4),nn.Dropout(.1))
        self.proj_audio = nn.Linear(4,4)
        self.head = nn.Linear(4,1)
    def forward(self,text,audio,video,**kw):
        return dict(logits=self.head(self.text_encoder(text)+self.proj_audio(audio)+video))


def main():
    output = Path(sys.argv[1])
    sys.path.insert(0,str(BASE/'source'))
    spec = importlib.util.spec_from_file_location('parity_native_trainer',BASE/'source/trainer.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from utils.seed import set_seed
    set_seed(2024)
    a = Model();b = deepcopy(a)
    args = SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
    logger = SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)
    native = module.Trainer(deepcopy(args),a,logger)
    batches = [dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),
                    labels={'M':torch.randn(4)}) for _ in range(8)]
    cpu_before = torch.get_rng_state().clone();cuda_before = torch.cuda.get_rng_state().clone()
    loss = [native.train_one_epoch(batches,e) for e in (1,2)]
    cpu_after = torch.get_rng_state().clone();cuda_after = torch.cuda.get_rng_state().clone()
    saved_optimizer = deepcopy(native.optimizer.state_dict())
    install(module,output,write)
    observed = module.Trainer(deepcopy(args),b,logger)
    torch.set_rng_state(cpu_before);torch.cuda.set_rng_state(cuda_before)
    guarded_loss = [observed.train_one_epoch(batches,e) for e in (1,2)]
    report = dict(loss_exact=loss==guarded_loss,
        parameters_exact=all(torch.equal(p,b.state_dict()[n]) for n,p in a.state_dict().items()),
        optimizer_exact=all(torch.equal(v,observed.optimizer.state_dict()['state'][k][key])
                            for k,s in saved_optimizer['state'].items() for key,v in s.items()),
        scheduler_exact=native.scheduler.state_dict()==observed.scheduler.state_dict(),
        cpu_rng_exact=torch.equal(cpu_after,torch.get_rng_state()),
        cuda_rng_exact=torch.equal(cuda_after,torch.cuda.get_rng_state()),
        updates_checked=observed.numerical_guard.updates_checked,
        physical_gpu=1,steps=16,scope='small model/dropout native Trainer parity; not full-model deterministic-replay proof')
    write(output/'cuda_parity.json',report)
    print(json.dumps(report),flush=True)
    if not all(v for k,v in report.items() if k.endswith('_exact')):
        raise RuntimeError('Finite-path GPU parity failed')


if __name__ == '__main__':
    main()
