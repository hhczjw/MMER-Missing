"""Isolated multi-step native/native/adapter parity, without altering frozen jobs."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/checkpoint_nvme_20261007'))
sys.path.insert(0, str(ROOT / 'scripts/research2_ablations'))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
from data_cache import install_cache
install_cache(ROOT / 'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
import numpy as np
import torch
from runtime import arguments, loaders, optimizer, interventions, UMRMv3, set_seed
from scripts.train import train_epoch, TaskLoss
from queue_guard import acquire_idle_gpus


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def snapshot(model, gradients=False):
    return {n: (p.grad if gradients else p).detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and (not gradients or p.grad is not None)}


def differences(a, b):
    if a.keys() != b.keys():
        raise AssertionError('Parameter names differ')
    rows = []
    for name in a:
        error = (a[name].float() - b[name].float()).abs()
        rows.append(dict(name=name, max_abs=float(error.max()),
                         changed=int(torch.count_nonzero(error)), numel=error.numel()))
    rows.sort(key=lambda r: r['max_abs'], reverse=True)
    return dict(max_abs=rows[0]['max_abs'], changed=sum(r['changed'] for r in rows),
                top=rows[:12])


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--wait-worker', type=int)
    cli = parser.parse_args()
    if cli.steps < 1:
        raise ValueError('steps must be positive')
    gpu_locks = acquire_idle_gpus()
    cli.output.mkdir(parents=True, exist_ok=False)
    write(cli.output / 'status.json', dict(phase='waiting_for_gpu', pid=os.getpid(),
                                          wait_worker=cli.wait_worker))
    if cli.wait_worker:
        until = time.monotonic() + 1800
        proc = Path('/proc') / str(cli.wait_worker) / 'stat'
        while proc.exists() and proc.read_text().split(') ', 1)[1][0] != 'Z':
            if time.monotonic() > until:
                raise TimeoutError('Existing worker did not finish within 30 minutes')
            time.sleep(3)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    write(cli.output / 'status.json', dict(phase='initializing', pid=os.getpid()))
    args = arguments('R2-T', cli.seed)
    set_seed(cli.seed)
    train, valid, _ = loaders(args)
    model = UMRMv3(args).to(device)
    initial = snapshot(model)
    # Capture initialization/data evidence before any training step or intervention.
    iterator = iter(train)
    batches = [next(iterator) for _ in range(cli.steps)]
    del iterator
    batch_evidence = [dict(ids=list(map(str, b['id'])),
                           hashes={k:digest(b[k]) for k in ['text','audio','vision','label']})
                      for b in batches]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    report = dict(seed=cli.seed, steps=cli.steps, args=vars(args).copy(),
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   device=torch.cuda.get_device_name(),
                                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                  initialization={n:digest(v) for n,v in initial.items()}, batches=batch_evidence,
                  cpu_rng=digest(cpu_rng), cuda_rng=digest(cuda_rng), runs={}, comparisons={})
    evidence = {}
    for name, adapted in [('native_a',False), ('native_b',False), ('adapter',True)]:
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in initial:
                    p.copy_(initial[n].to(device))
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        run_args = deepcopy(args)
        run_args.token_train_rate_cap = .3
        opt = optimizer(model, run_args)
        for group in opt.param_groups:
            group['lr'] /= 5
        restore, _ = interventions(model,{}) if adapted else (lambda:None, [])
        forwards = []
        def trace(module, inputs, output):
            forwards.append(dict(training=module.training,
                input_hashes=[digest(x) for x in inputs[:3]],
                logits_hash=digest(output['logits']),
                mask_hash=digest(output['token_mask']),
                modality_hash=digest(output['modality_mask']),
                components={k:float(output[k].detach()) for k in
                            ['contrastive_loss','text_recon_loss']
                            if isinstance(output.get(k),torch.Tensor)}))
        hook = model.register_forward_hook(trace)
        rows = []
        states = []
        original_step = opt.step
        def step(*aa, **kw):
            gradients = snapshot(model, gradients=True)
            result = original_step(*aa, **kw)
            params = snapshot(model)
            states.append(dict(gradients=gradients,parameters=params,forwards=deepcopy(forwards)))
            rows.append(dict(step=len(rows),forwards=deepcopy(forwards),
                             sampled_rate=float(model.token_missing_rate),
                             parameter_hashes={n:digest(v) for n,v in params.items()}))
            forwards.clear()
            print('STEP',name,len(rows)-1,flush=True)
            return result
        opt.step = step
        # One call keeps the native corruption generator continuous across batches.
        loss = train_epoch(model,batches,TaskLoss(run_args.task_type,run_args.num_classes),
                           opt,run_args,device,1,60)
        hook.remove();restore()
        report['runs'][name] = dict(mean_loss=loss,steps=rows,
                                    rate_stats=run_args.token_train_last_stats)
        if name != 'native_a':
            comparisons = []
            for index,(a,b) in enumerate(zip(evidence['native_a'],states)):
                comparisons.append(dict(step=index,gradients=differences(a['gradients'],b['gradients']),
                    parameters=differences(a['parameters'],b['parameters']),
                    forwards_exact=a['forwards']==b['forwards']))
            report['comparisons'][name] = comparisons
            report.setdefault('mean_loss_differences',{})[name] = loss-report['runs']['native_a']['mean_loss']
            del states
        else:
            evidence[name] = states
        del opt
        model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()
        write(cli.output/'report.json',report)
    write(cli.output/'status.json',dict(phase='complete',pid=os.getpid(),
          peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2))
    print('COMPLETE',cli.output,flush=True)


if __name__ == '__main__':
    main()
