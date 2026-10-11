"""Validation-only, preregistered Token optimization; never opens test loader."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research2_token_optimization_20261010'
ARCHIVE = Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints/research2_token_optimization_20261010')
CANDIDATES = [
    dict(id='control', lr=3e-4, cap_period=1, text_recon_weight=.5),
    dict(id='lower_lr', lr=1e-4, cap_period=1, text_recon_weight=.5),
    dict(id='slower_curriculum', lr=3e-4, cap_period=5, text_recon_weight=.5),
    dict(id='lower_text_recon', lr=3e-4, cap_period=1, text_recon_weight=.1),
]
SEEDS = [42, 45]


def write(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def cap(candidate, epoch):
    return min(.9, .3 + .2 * ((epoch - 1) // candidate['cap_period']))


def gpu_lock():
    handle = (ROOT / 'results/research2_gpu1.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    if uuid in apps:
        handle.close()
        raise RuntimeError('GPU1 occupied; will not interrupt other jobs')
    return handle


def run(smoke):
    lock = gpu_lock()
    os.environ.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(ROOT / 'scripts/checkpoint_nvme_20261007'))
    from data_cache import install_cache
    install_cache(ROOT / 'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
    import runtime as rt
    import torch
    import numpy as np
    frozen = rt.read(ROOT / 'results/research2_ablations_gpu1_20261006_v3/plan.json')
    rt.assert_frozen(frozen)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    OUT.mkdir(exist_ok=True)
    state_path = OUT / ('smoke.json' if smoke else 'status.json')
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(phase='smoke' if smoke else 'training', pid=os.getpid(), gpu=1, completed=[], test_access=False)
    write(state_path, state)
    try:
        if not smoke:
            receipt = rt.read(OUT / 'smoke.json')
            assert receipt['phase'] == 'complete' and receipt['runner_sha256'] == digest(Path(__file__))
        manifest = dict(candidates=CANDIDATES, seeds=SEEDS, epochs=60, early_stopping=False,
                        validation='Token10 mean Acc7, exact tie MAE; fixed masks 20261006',
                        selection='mean best validation Acc7 over both preregistered seeds; tie mean MAE',
                        no_test_access=True, purpose='diagnostic development; not final confirmatory ablation',
                        runner_sha256=digest(Path(__file__)), frozen_source=frozen['source_sha256'],
                        future_gate='Freeze recipe and repeat all affected ablations with matched budgets; do not mix old/new results')
        if not smoke:
            write(OUT / 'plan.json', manifest)
        summaries = []
        for seed in ([42] if smoke else SEEDS):
            for candidate in CANDIDATES:
                job = candidate['id'] + f'_seed{seed}'
                state.update(current=job)
                write(state_path, state)
                args = rt.arguments('R2-T', seed, {'text_recon_weight': candidate['text_recon_weight']})
                args.learning_rate = candidate['lr']
                rt.set_seed(seed)
                train, valid, _ = rt.loaders(args)
                model = rt.UMRMv3(args).to(device)
                opt = rt.optimizer(model, args)
                _, plate = rt.schedules(opt, args, 'R2-T')
                base_lrs = [g['lr'] for g in opt.param_groups]
                best, history = None, []
                folder = OUT / ('smoke_' + job if smoke else job)
                folder.mkdir(exist_ok=False)
                write(folder / 'config.json', dict(args=vars(args), candidate=candidate))
                epochs = 1 if smoke else 60
                for epoch in range(1, epochs + 1):
                    begin = time.time()
                    if epoch <= 5:
                        for group, lr in zip(opt.param_groups, base_lrs):
                            group['lr'] = lr * epoch / 5
                    args.token_train_rate_cap = cap(candidate, epoch)
                    model.token_missing_rate = args.token_train_rate_cap
                    batches = [next(iter(train))] if smoke else train
                    probe = model.proj_audio.weight.detach().clone() if smoke else None
                    lr_start = [g['lr'] for g in opt.param_groups]
                    loss = rt.train_epoch(model, batches, rt.TaskLoss(args.task_type, args.num_classes), opt, args, device, epoch, 60)
                    assert np.isfinite(loss)
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    if smoke:
                        assert not torch.equal(probe, model.proj_audio.weight)
                    val = rt.validation(model, [next(iter(valid))] if smoke else valid, args, 'R2-T', device)
                    value = val['avg_over_rates']
                    assert all(np.isfinite(v) for v in value.values())
                    if rt.improve('R2-T', value, best):
                        best, best_epoch = value.copy(), epoch
                        if not smoke:
                            rt.checkpoint(folder / 'best_Acc7.pth', model, args, epoch, val, candidate)
                        write(folder / 'selection.json', dict(epoch=epoch, validation=val))
                    if epoch >= 5:
                        plate.step(value['Acc7'])
                    row = dict(epoch=epoch, loss=loss, validation=value, lr_start=lr_start,
                               lr_next=[g['lr'] for g in opt.param_groups], rate_cap=args.token_train_rate_cap,
                               token_stats=args.token_train_last_stats, seconds=time.time()-begin)
                    history.append(row)
                    write(folder / 'history.json', history)
                    state.update(epoch=epoch, best_epoch=best_epoch, best_validation=best)
                    write(state_path, state)
                    print(job, epoch, loss, value['Acc7'], 'best', best_epoch, flush=True)
                summary = dict(job=job, candidate=candidate['id'], seed=seed, best_epoch=best_epoch, validation=best)
                if not smoke:
                    destination = ARCHIVE / job
                    destination.mkdir(parents=True, exist_ok=False)
                    cp = folder / 'best_Acc7.pth'
                    shutil.copy2(cp, destination / cp.name)
                    assert digest(cp) == digest(destination / cp.name)
                    summary['checkpoint'] = str(destination / cp.name)
                summaries.append(summary)
                write(OUT / ('smoke_summary.json' if smoke else 'validation_summary.json'), summaries)
                state['completed'].append(job)
                write(state_path, state)
                del model, opt, plate, train, valid, batches, probe
                gc.collect()
                torch.cuda.empty_cache()
        if not smoke:
            ranked = []
            for c in CANDIDATES:
                members = [s for s in summaries if s['candidate'] == c['id']]
                ranked.append(dict(candidate=c, mean_Acc7=float(np.mean([s['validation']['Acc7'] for s in members])),
                                   mean_MAE=float(np.mean([s['validation']['MAE'] for s in members]))))
            ranked.sort(key=lambda r: (-r['mean_Acc7'], r['mean_MAE']))
            write(OUT / 'validation_ranking.json', dict(ranked=ranked, final_test_performed=False,
                  next_step='Review paired stability, freeze recipe, then matched confirmatory ablations; no claim all modules beneficial'))
        state.update(phase='complete', runner_sha256=digest(Path(__file__)))
        write(state_path, state)
    except BaseException:
        state.update(phase='failed', error=traceback.format_exc())
        write(state_path, state)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'run', 'launch'], required=True)
    mode = parser.parse_args().mode
    if mode == 'launch':
        if (OUT / 'status.json').exists():
            raise FileExistsError('Existing run; refusing duplicate launch')
        with (OUT / 'console.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--mode', 'run'],
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print('Launched PID', process.pid)
    else:
        run(mode == 'smoke')
