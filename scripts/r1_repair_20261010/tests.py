"""CPU tests of the repair: native equality and failures before poisoned updates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys
import os

# The native Trainer selects CUDA whenever available. Hide devices before
# importing torch so these safety tests cannot claim CPU-only while using GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/r1_aligned_20261009'))
from plan import BASE, read, write
import numerical_guard as guard


def trainer_module():
    sys.path.insert(0, str(BASE/'source'))
    spec = importlib.util.spec_from_file_location('test_native_trainer', BASE/'source/trainer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Linear(4,4)
        self.proj_audio = nn.Linear(4,4)
        self.head = nn.Linear(4,1)
    def forward(self, text, audio, video, **kw):
        x = self.text_encoder(text) + self.proj_audio(audio) + video
        return {'logits':self.head(x)}


class Tests(unittest.TestCase):
    def test_finite_clip_bitwise_and_rng(self):
        original = torch.nn.utils.clip_grad_norm_
        a = [nn.Parameter(torch.randn(8,8)) for _ in range(4)]
        b = deepcopy(a)
        for x,y in zip(a,b):
            x.grad = torch.randn_like(x); y.grad = x.grad.clone()
        rng = torch.get_rng_state().clone()
        g = guard.NumericalGuard(None, '/tmp', lambda *a:None)
        na = original(a, .1)
        nb = g.checked_clip(original, b, .1)
        self.assertTrue(torch.equal(na,nb))
        self.assertTrue(all(torch.equal(x.grad,y.grad) for x,y in zip(a,b)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_nonfinite_rejected_before_gradient_modification(self):
        p = nn.Parameter(torch.ones(3)); p.grad = torch.tensor([1.,float('inf'),2.])
        g = guard.NumericalGuard(None, '/tmp', lambda *a:None)
        with patch.object(g,'abort',side_effect=RuntimeError('captured')) as abort:
            with self.assertRaisesRegex(RuntimeError,'captured'):
                g.checked_clip(torch.nn.utils.clip_grad_norm_,[p],.1)
            self.assertEqual(abort.call_args.args[0],'backward_or_norm')
        self.assertTrue(torch.equal(p.grad,torch.tensor([1.,float('inf'),2.])))

    def test_native_multi_batch_multi_epoch_equality(self):
        module = trainer_module(); original_clip = torch.nn.utils.clip_grad_norm_
        torch.manual_seed(123)
        a = Model(); b = deepcopy(a)
        args = SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
        logger = SimpleNamespace(info=lambda *a:None, warning=lambda *a:None)
        native = module.Trainer(deepcopy(args),a,logger)
        batches = [dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),
                        labels={'M':torch.randn(4)}) for _ in range(5)]
        rng = torch.get_rng_state().clone()
        loss = [native.train_one_epoch(batches,e) for e in (1,2)]
        state = deepcopy(native.optimizer.state_dict()); sched = deepcopy(native.scheduler.state_dict())
        after = torch.get_rng_state().clone()
        try:
            with tempfile.TemporaryDirectory() as temp:
                guard.install(module,temp,write)
                observed = module.Trainer(deepcopy(args),b,logger)
                torch.set_rng_state(rng)
                guarded_loss = [observed.train_one_epoch(batches,e) for e in (1,2)]
                self.assertEqual(loss,guarded_loss)
                self.assertTrue(torch.equal(after,torch.get_rng_state()))
                self.assertTrue(all(torch.equal(p,b.state_dict()[n]) for n,p in a.state_dict().items()))
                self.assertEqual(sched,observed.scheduler.state_dict())
                for k, s in state['state'].items():
                    for key,v in s.items():
                        self.assertTrue(torch.equal(v,observed.optimizer.state_dict()['state'][k][key]))
                self.assertEqual(observed.numerical_guard.updates_checked,10)
                self.assertEqual(read(Path(temp)/'numerical_progress.json')['epoch'],2)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip

    def test_backward_nan_captured_with_clean_parameter_and_no_update(self):
        module = trainer_module(); original_clip = torch.nn.utils.clip_grad_norm_
        try:
            with tempfile.TemporaryDirectory() as temp:
                guard.install(module,temp,write)
                model = Model(); init = deepcopy(model.state_dict())
                model.head.weight.register_hook(lambda grad: grad * float('nan'))
                args = SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
                t = module.Trainer(args,model,SimpleNamespace(info=lambda *a:None,warning=lambda *a:None))
                batch = dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),labels={'M':torch.randn(4)})
                with self.assertRaisesRegex(RuntimeError,'Numerical failure captured'):
                    t.train_one_epoch([batch],1)
                report = read(Path(temp)/'numerical_failure.json')
                self.assertEqual(report['stage'],'backward_or_norm')
                self.assertEqual(report['bad_gradients'][0]['name'],'head.weight')
                self.assertEqual(report['bad_parameters'],[])
                self.assertEqual(t.global_step,0)
                self.assertTrue(all(torch.equal(p,model.state_dict()[n]) for n,p in init.items()))
                self.assertTrue((Path(temp)/'failure_snapshot.pt').exists())
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU tests initialized CUDA')
    raise SystemExit(0 if result.wasSuccessful() else 1)
