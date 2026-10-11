"""Authorized seed42 fresh restart using GPU1-only locking, unchanged native training."""
import fcntl
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_p0b_seed42 as native_launcher

OUTPUT = ROOT / 'results/p0B_legacy_replay_20261009_seed42/run_02'
BOOTSTRAP = Path(__file__).with_name('bootstrap_p0b_seed42_run02.py')
AUTHORITY = '2026-10-09 human: 可以先中断训练，释放锁，让研究二的消融实验占用GPU0，然后在GPU1中重新训练研究一'


def acquire_gpu1_only():
    lock = (ROOT / 'results/research2_gpu1.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
            '--format=csv,noheader'], text=True, timeout=10)
        gpu1 = next(line for line in inventory.splitlines() if line.split(',')[0].strip() == '1')
        expected_uuid = native_launcher.legacy.read(native_launcher.BASE / 'recovery_manifest.json')['current_gpu1_uuid']
        if expected_uuid not in gpu1:
            raise RuntimeError('GPU1 identity changed')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader'], text=True, timeout=10)
        if any(expected_uuid in line for line in processes.splitlines()):
            raise RuntimeError('Deferred: GPUs still have compute processes: physical GPU1 is busy')
        return [lock]
    except BaseException:
        lock.close()
        raise


def main():
    # Keep the original launcher/source unchanged, overriding only orchestration/output.
    native_launcher.OUTPUT = OUTPUT
    native_launcher.BOOTSTRAP = BOOTSTRAP
    native_launcher.AUTHORITY = AUTHORITY
    native_launcher.acquire_idle_gpus = acquire_gpu1_only
    helper_paths = [Path(__file__), BOOTSTRAP,
        Path(__file__).with_name('bootstrap_p0b_seed42.py')]
    helper_fingerprints = {str(path):native_launcher.legacy.sha(path) for path in helper_paths}
    original_verify = native_launcher.verify_helpers
    def verify_all(fingerprints):
        original_verify(fingerprints)
        original_verify(helper_fingerprints)
        (OUTPUT / 'single_gpu_restart.json').write_text(__import__('json').dumps(dict(
            authority=AUTHORITY, physical_gpu=1, gpu_lock=str(ROOT / 'results/research2_gpu1.lock'),
            prior_interrupted_run=str(OUTPUT.with_name('run_01')),
            native_source_unchanged=True, native_training_config_unchanged=True,
            from_scratch=True, old_checkpoint_resume=False,
            helper_sha256=helper_fingerprints), ensure_ascii=False, indent=2))
    native_launcher.verify_helpers = verify_all
    native_launcher.main()


if __name__ == '__main__':
    main()
