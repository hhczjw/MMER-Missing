"""Re-evaluate archived p0B with the recovered native code, without training."""
import argparse
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/p0B_legacy_replay_20261008'
sys.path.insert(0, str(ROOT / 'scripts/reproduction_20261008'))
from queue_guard import acquire_idle_gpus, check_queues


def acquire_gpu0_validation():
    """Explicit GPU0-only check alongside the known GPU1 baseline controller.

    The baseline controller reserves both legacy locks although its commands
    target GPU1. Keep that reservation intact and use an exclusive diagnostic
    lease, checking the actual GPU and the controller's visible-device setting.
    """
    check_queues()
    handle = (ROOT / 'results/r1_gpu0_validation.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = ROOT / 'results/research2_legacy_replay_20261009/run_01/status.json'
        state = json.loads(state_path.read_text())
        if state.get('physical_gpu') != 1:
            raise RuntimeError('Research2 is not confined to physical GPU1')
        if state.get('phase') == 'running':
            pid = state.get('worker_pid')
            proc = Path('/proc') / str(pid)
            if not proc.exists():
                raise RuntimeError('Recheck research2 boundary before concurrent verification')
            values = (proc / 'environ').read_bytes().split(b'\0')
            if b'CUDA_VISIBLE_DEVICES=1' not in values:
                raise RuntimeError('Research2 worker GPU confinement is not verified')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                    '--format=csv,noheader'], text=True)
        gpu0 = next(line.split(',')[1].strip() for line in inventory.splitlines()
                    if line.split(',')[0].strip() == '0')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
               '--format=csv,noheader'], text=True)
        if any(gpu0 in line for line in apps.splitlines()):
            raise RuntimeError('Physical GPU0 already has a compute process')
        return [handle]
    except BaseException:
        handle.close()
        raise


class EvaluationDone(Exception):
    pass


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-idle', action='store_true')
    parser.add_argument('--physical-gpu', type=int, choices=[0, 1], default=1)
    parser.add_argument('--allow-concurrent-r2', action='store_true',
                        help='Explicit user-authorized GPU0 diagnostic alongside GPU1 R2')
    cli = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(cli.physical_gpu):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must match the physical GPU argument')
    if cli.physical_gpu == 0 and not cli.allow_concurrent_r2:
        raise RuntimeError('GPU0 check requires explicit concurrent diagnostic authorization')
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    while True:
        try:
            locks = acquire_gpu0_validation() if cli.physical_gpu == 0 else acquire_idle_gpus()
            break
        except (RuntimeError, BlockingIOError) as error:
            write(cli.output / 'status.json', dict(phase='waiting_for_gpu', pid=os.getpid(),
                  reason=str(error), cuda_started=False, updated_at=datetime.now().isoformat()))
            if not cli.wait_for_idle or time.monotonic() - started > 24 * 3600:
                raise
            time.sleep(15)
    try:
        from launch_p0b import verify_sources, verify_inputs
        manifest = json.loads((BASE / 'recovery_manifest.json').read_text())
        verify_sources(manifest)
        verify_inputs(manifest)
        write(cli.output / 'status.json', dict(phase='evaluating', pid=os.getpid(), physical_gpu=cli.physical_gpu))
        sys.path.insert(0, str(BASE))
        from data_cache import install_cache
        assert install_cache(BASE / 'input_cache_index.json')
        sys.path.insert(0, str(BASE / 'source'))
        os.chdir(BASE)
        import torch
        import train
        original_config = train.load_config
        original_loader = train.MMDataLoader
        holders = {}

        def config(dataset):
            args = original_config(dataset)
            for key in ['logs_dir', 'checkpoints_dir', 'results_dir']:
                setattr(args, key, str(cli.output / key))
            return args

        def loaders(*args, **kwargs):
            value = original_loader(*args, **kwargs)
            holders.update(value)
            return value

        class EvaluationTrainer(train.Trainer):
            def __init__(self, args, model, logger):
                super().__init__(args, model, logger)
                archived = Path(manifest['preserved_best']['checkpoint'])
                payload = torch.load(archived, map_location='cpu', mmap=True, weights_only=False)
                self.model.load_state_dict(payload['model'], strict=True)
                del payload
                with torch.no_grad():
                    metrics = self.evaluate(holders['test'], split='archived_best_Acc2')
                expected = json.loads((BASE / manifest['preserved_best']['result']).read_text())['secondary_Acc2']
                differences = {key: metrics[key] - expected[key] for key in
                               ['Acc2', 'Acc2_has0', 'F1', 'F1_has0', 'Acc7', 'MAE', 'Corr']}
                write(cli.output / 'report.json', dict(checkpoint=str(archived), strict_state_dict=True,
                    physical_gpu=cli.physical_gpu,
                    metrics=metrics, historical=expected, differences=differences,
                    tolerance=1e-5, passed=all(abs(v) < 1e-5 for v in differences.values()),
                    source='recovered native train.py/model/data/Trainer.evaluate', training_steps=0))
                raise EvaluationDone()

        train.load_config = config
        train.MMDataLoader = loaders
        train.Trainer = EvaluationTrainer
        sys.argv = [str(BASE / 'source/train.py'), '--dataset', 'MOSI', '--seed', '2024', '--exp_tag', 'p0B']
        try:
            train.main()
        except EvaluationDone:
            pass
        report = json.loads((cli.output / 'report.json').read_text())
        write(cli.output / 'status.json', dict(phase='complete', pid=os.getpid(),
              parity_passed=report['passed'], finished_at=datetime.now().isoformat()))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as error:
        write(cli.output / 'status.json', dict(phase='failed', error=repr(error), pid=os.getpid()))
        raise
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
