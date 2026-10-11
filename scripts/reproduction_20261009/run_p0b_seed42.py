"""CPU-only wait for R2 success, then one native seed42 run on physical GPU1."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reproduction_20261008'))
import launch_p0b as legacy
from queue_guard import acquire_idle_gpus, live

BASE = legacy.BASE
OUTPUT = ROOT / 'results/p0B_legacy_replay_20261009_seed42/run_01'
R2 = ROOT / 'results/research2_legacy_replay_20261009/run_01'
BOOTSTRAP = Path(__file__).with_name('bootstrap_p0b_seed42.py')
AUTHORITY = '2026-10-09 human: 切换seed 为42 再跑一次。'


def write(value):
    value['updated_at'] = datetime.now().astimezone().isoformat()
    temp = OUTPUT / 'status.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(OUTPUT / 'status.json')


def dependency_ready(state):
    if state['phase'] == 'failed':
        raise RuntimeError('Research2 failed; seed42 will not start automatically')
    if state['phase'] != 'complete':
        if not live(state.get('pid', -1)):
            raise RuntimeError('Research2 controller lost before successful completion')
        return False
    expected = {'Token_seed45', 'Modality_seed43', 'Modality_seed45', 'Modality_seed48',
                'Modality_seed42', 'Modality_seed44', 'Missing_ensemble', 'Full_ensemble'}
    if not expected.issubset(state.get('completed', [])):
        raise RuntimeError('Research2 complete marker lacks required jobs')
    if any(live(state.get(k, -1)) for k in ['pid', 'worker_pid']):
        return False
    for name in ['missing_ensemble.json', 'full_ensemble.json']:
        if not math.isfinite(float(legacy.read(R2 / name)['TAV']['MAE'])):
            raise RuntimeError('Invalid completed Research2 ensemble')
    return True


def verify_helpers(fingerprints):
    for name, expected in fingerprints.items():
        if legacy.sha(Path(name)) != expected:
            raise RuntimeError(f'Queued helper changed: {name}')


def environment(manifest):
    inventory = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True, timeout=10)
    gpu1 = next(line for line in inventory.splitlines() if line.split(',')[0].strip() == '1')
    expected = manifest['expected_current_environment']
    if manifest['current_gpu1_uuid'] not in gpu1 or expected['driver'] not in gpu1:
        raise RuntimeError('GPU1 identity or driver changed; review required')
    env = os.environ.copy()
    for key in manifest['expected_unset_environment']:
        if env.get(key) is not None:
            raise RuntimeError(f'Unreviewed inherited override: {key}')
    env.update(CUDA_VISIBLE_DEVICES='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=str(BASE / 'source/mamba') + ':' + str(BASE / 'source'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    probe = "import importlib.metadata as m,json,torch,sys;print(json.dumps(dict(python=sys.version.split()[0],packages={k:m.version(k) for k in ['torch','triton','transformers','numpy','tokenizers','causal-conv1d']},cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_initialized=torch.cuda.is_initialized())))"
    actual = json.loads(subprocess.check_output([legacy.PYTHON, '-c', probe],
        env=dict(env, CUDA_VISIBLE_DEVICES=''), text=True))
    if actual['cuda_initialized'] or any(actual[k] != expected[k] for k in ['python','packages','cuda','cudnn']):
        raise RuntimeError('Authorized candidate dependency fingerprint changed')
    (OUTPUT / 'environment_at_launch.json').write_text(json.dumps(dict(cpu_probe=actual,
        gpu_inventory=inventory, settings={k:env.get(k) for k in manifest['capture_environment_keys']},
        historical_alignment='authorized candidate; exact historical equality unknown'), indent=2))
    return env


def main():
    manifest = legacy.read(BASE / 'recovery_manifest.json')
    legacy.verify_sources(manifest)
    review = legacy.read(BASE / 'environment_review.json')
    if review['state'] != 'user_authorized_candidate' or not review.get('evidence_or_user_direction'):
        raise RuntimeError('Candidate environment authorization missing')
    helpers = {str(path):legacy.sha(path) for path in [Path(__file__), BOOTSTRAP,
        ROOT / 'scripts/reproduction_20261008/launch_p0b.py',
        ROOT / 'scripts/reproduction_20261008/queue_guard.py']}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    state = dict(phase='waiting_research2', pid=os.getpid(), seed=42, physical_gpu=1,
        authority=AUTHORITY, dependency=str(R2), source_base=str(BASE),
        comparison_seed2024='Only seed and independent output paths change; native code/config unchanged',
        historical_alignment='user-authorized candidate, not proven exact')
    write(state)
    (OUTPUT / 'manifest_at_queue.json').write_text(json.dumps(dict(recovery_manifest=manifest,
        helper_sha256=helpers, review=review, seed=42, authority=AUTHORITY), ensure_ascii=False, indent=2))
    locks = []
    try:
        while not dependency_ready(legacy.read(R2 / 'status.json')):
            time.sleep(15)
        state['phase'] = 'waiting_resources'
        write(state)
        while True:
            try:
                locks = acquire_idle_gpus()
                break
            except BlockingIOError:
                time.sleep(15)
            except RuntimeError as error:
                if not str(error).startswith('Deferred: GPUs still have compute processes:'):
                    raise
                time.sleep(15)
        verify_helpers(helpers)
        legacy.verify_sources(manifest)
        legacy.verify_inputs(manifest)
        env = environment(manifest)
        with (OUTPUT / 'console.log').open('x') as console:
            child = subprocess.Popen([legacy.PYTHON, '-u', str(BOOTSTRAP)], cwd=BASE,
                env=env, stdout=console, stderr=subprocess.STDOUT,
                pass_fds=tuple(handle.fileno() for handle in locks))
            state.update(phase='train', worker_pid=child.pid)
            write(state)
            code = child.wait()
        if code:
            raise RuntimeError(f'Native seed42 exited {code}; no automatic retries or protocol changes')
        report = legacy.read(OUTPUT / 'results/MOSI_p0B_seed42_test.json')
        for key, metric in [('primary_MAE','MAE'), ('secondary_Acc2','Acc2'),
                            ('tertiary_Acc7','Acc7'), ('quaternary_F1_has0','F1_has0')]:
            if not report[key] or not all(math.isfinite(float(v)) for v in report[key].values()):
                raise RuntimeError(f'Invalid test result {key}')
            checkpoint = OUTPUT / f'checkpoints/MOSI_p0B_seed42_best_{metric}.pt'
            if not checkpoint.is_file() or not checkpoint.stat().st_size:
                raise RuntimeError(f'Missing checkpoint {metric}')
        state.update(phase='complete', exit_code=0)
        write(state)
    except BaseException:
        state.update(phase='failed', traceback=traceback.format_exc())
        write(state)
        raise
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
