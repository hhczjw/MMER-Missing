"""Isolated p0B seed2024 replay. No CUDA until queues, review and locks pass."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from queue_guard import acquire_idle_gpus, check_queues

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/p0B_legacy_replay_20261008'
OUTPUT = ROOT / 'results/p0B_legacy_replay_20261008/run_01'
PYTHON = '/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources(manifest):
    for name, info in manifest['files'].items():
        if sha(BASE / name) != info['sha256']:
            raise RuntimeError(f'Frozen file changed: {name}')


def verify_inputs(manifest):
    mapping = read(BASE / 'input_cache_index.json')['mapping']
    for name, expected in manifest['data_sha256'].items():
        actual = Path(mapping[name]['local']) if name in mapping else Path(name)
        if sha(actual) != expected:
            raise RuntimeError(f'Input bytes changed: {name}')
    for name, info in manifest['pretrained_files'].items():
        if sha(Path(name)) != info['sha256']:
            raise RuntimeError(f'Pretrained cache changed: {name}')
    for model, revision in manifest['pretrained_revisions'].items():
        ref = Path('/home/zjw/.cache/huggingface/hub') / ('models--' + model) / 'refs/main'
        if ref.read_text().strip() != revision:
            raise RuntimeError(f'Cached pretrained revision changed: {model}')


def write_status(value):
    value['updated_at'] = datetime.now().isoformat()
    tmp = OUTPUT / 'status.tmp'
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(OUTPUT / 'status.json')


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--launch', action='store_true')
    cli = p.parse_args()
    manifest = read(BASE / 'recovery_manifest.json')
    verify_sources(manifest)
    review = read(BASE / 'environment_review.json')
    if cli.verify_only:
        try:
            queue = check_queues()
        except RuntimeError as e:
            queue = str(e)
        print(json.dumps(dict(source_fingerprints='pass', environment_review=review['state'],
                              queue=queue, output_exists=OUTPUT.exists(),
                              cuda_diagnostic_or_training_started=False), ensure_ascii=False, indent=2))
        return
    # A name match does not establish historical package/device parity.
    if review['state'] not in ('historical_alignment_verified', 'user_authorized_candidate'):
        raise RuntimeError('Pending historical environment/source review; no CUDA launch authorized')
    if not review.get('evidence_or_user_direction'):
        raise RuntimeError('Missing traceable review decision')
    locks = acquire_idle_gpus()
    created_output = False
    try:
        verify_inputs(manifest)
        inventory = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'
        ], text=True, timeout=10)
        gpu1 = next((line for line in inventory.splitlines() if line.split(',')[0].strip() == '1'), '')
        if manifest['current_gpu1_uuid'] not in gpu1:
            raise RuntimeError('Physical GPU1 has changed: re-review device identity')
        if manifest['expected_current_environment']['driver'] not in gpu1:
            raise RuntimeError('Driver has changed: re-review environment')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   PYTHONPATH=str(BASE / 'source/mamba') + ':' + str(BASE / 'source'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # Unknown inherited overrides must be reviewed, not silently injected.
        for key in manifest['expected_unset_environment']:
            if env.get(key) is not None:
                raise RuntimeError(f'Unexpected inherited setting {key}; review before launch')
        probe = "import importlib.metadata as m,json,torch,sys; print(json.dumps(dict(python=sys.version.split()[0], packages={k:m.version(k) for k in ['torch','triton','transformers','numpy','tokenizers','causal-conv1d']}, cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_initialized=torch.cuda.is_initialized())))"
        cpu_env = env.copy()
        cpu_env['CUDA_VISIBLE_DEVICES'] = ''
        actual = json.loads(subprocess.check_output([PYTHON, '-c', probe], env=cpu_env, text=True))
        expected = manifest['expected_current_environment']
        if actual['cuda_initialized'] or any(actual[k] != expected[k] for k in ['python', 'packages', 'cuda', 'cudnn']):
            raise RuntimeError('Current dependency fingerprint changed: re-review environment')
        # mkdir is exclusive. Never resume or overwrite a previous attempt.
        OUTPUT.mkdir(parents=True, exist_ok=False)
        created_output = True
        (OUTPUT / 'review_at_launch.json').write_text(json.dumps(review, indent=2, ensure_ascii=False))
        (OUTPUT / 'environment_at_launch.json').write_text(json.dumps(dict(cpu_probe=actual, gpu_inventory=inventory, settings={k:env.get(k) for k in manifest['capture_environment_keys']}), indent=2))
        write_status(dict(phase='starting', pid=os.getpid(), seed=2024, physical_gpu=1,
                          restoration_scope=manifest['restoration_scope']))
        with (OUTPUT / 'console.log').open('x') as console:
            child = subprocess.Popen([PYTHON, '-u', str(BASE / 'bootstrap.py')], cwd=BASE,
                                     env=env, stdout=console, stderr=subprocess.STDOUT,
                                     pass_fds=tuple(h.fileno() for h in locks))
            write_status(dict(phase='train', pid=os.getpid(), worker_pid=child.pid, seed=2024, physical_gpu=1))
            result = child.wait()
        if result:
            write_status(dict(phase='failed', exit_code=result, reason='Native training failed; no automatic protocol changes or retries'))
            raise RuntimeError(f'Native training exited {result}; inspect {OUTPUT}/console.log')
        report = read(OUTPUT / 'results/MOSI_p0B_seed2024_test.json')
        for key, metric in [('primary_MAE','MAE'), ('secondary_Acc2','Acc2'), ('tertiary_Acc7','Acc7'), ('quaternary_F1_has0','F1_has0')]:
            if not math.isfinite(float(report[key][metric])):
                raise RuntimeError(f'Invalid result {key}')
            if not (OUTPUT / f'checkpoints/MOSI_p0B_seed2024_best_{metric}.pt').is_file():
                raise RuntimeError(f'Missing checkpoint {metric}')
        write_status(dict(phase='complete', seed=2024, physical_gpu=1, exit_code=0))
        print(OUTPUT)
    except BaseException as e:
        if created_output:
            write_status(dict(phase='failed', reason=str(e)))
        raise
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
