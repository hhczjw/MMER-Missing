"""Serial native R2 replay after successful R1; retain GPU locks, fail closed."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/research2_legacy_replay_20261009'
OUTPUT = ROOT / 'results/research2_legacy_replay_20261009/run_01'
R1_OUTPUT = ROOT / 'results/p0B_legacy_replay_20261008/run_01'
PYTHON = '/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python'
sys.path.insert(0, str(ROOT / 'scripts/reproduction_20261008'))
from queue_guard import acquire_idle_gpus

TAG = 'umrmv3_kd0.3_recon_inj_rbl_rbp_lora_bi_uni0.1_g0.05_rcd0.1_cdkd_bal_ch_MOSI_seed{}'
MODEL_FLAGS = ['--bert-model', 'FacebookAI/roberta-large', '--bert-pretokenized',
    '--feature-path', 'CMU-MOSI/Processed/unaligned_50_roberta.pkl', '--lora-bert',
    '--uni', '--gap-weight', '0.05', '--rcd', '--cd-kd', '--balanced-missing',
    '--combo-heads', '--bi']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def alive(pid):
    stat = Path('/proc') / str(pid) / 'stat'
    return stat.exists() and stat.read_text().rsplit(') ', 1)[1][0] != 'Z'


def verify_sources(manifest):
    for name, info in manifest['files'].items():
        if sha(BASE / 'source' / name) != info['sha256']:
            raise RuntimeError(f'Frozen source changed: {name}')
    for name, expected in manifest['helper_sha256'].items():
        if sha(BASE / name) != expected:
            raise RuntimeError(f'Frozen helper changed: {name}')


def verify_inputs(manifest):
    if sha(Path(manifest['data_local_path'])) != next(iter(manifest['data_sha256'].values())):
        raise RuntimeError('Dataset bytes changed')
    for name, info in manifest['pretrained_files'].items():
        if sha(Path(name)) != info['sha256']:
            raise RuntimeError(f'Pretrained bytes changed: {name}')
    ref = Path('/home/zjw/.cache/huggingface/hub/models--FacebookAI--roberta-large/refs/main')
    if ref.read_text().strip() != manifest['pretrained_revision']:
        raise RuntimeError('Pretrained revision changed')


def environment(manifest):
    env = os.environ.copy()
    for key in manifest['inherited_overrides_must_be_unset']:
        if env.get(key) is not None:
            raise RuntimeError(f'Unreviewed inherited override: {key}')
    env.update(CUDA_VISIBLE_DEVICES='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=str(BASE / 'source/mamba') + ':' + str(BASE / 'source'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    inventory = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True, timeout=10)
    gpu1 = next(line for line in inventory.splitlines() if line.split(',')[0].strip() == '1')
    expected = manifest['expected_environment']
    if manifest['gpu1_uuid'] not in gpu1 or expected['driver'] not in gpu1:
        raise RuntimeError('GPU1 identity or driver changed')
    probe = "import importlib.metadata as m,json,torch,sys;print(json.dumps(dict(python=sys.version.split()[0],packages={k:m.version(k) for k in ['torch','triton','transformers','numpy','tokenizers','causal-conv1d']},cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_initialized=torch.cuda.is_initialized())))"
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
    actual = json.loads(subprocess.check_output([PYTHON, '-c', probe], env=cpu_env, text=True))
    if actual['cuda_initialized'] or any(actual[k] != expected[k] for k in ['python','packages','cuda','cudnn']):
        raise RuntimeError('Authorized candidate dependency fingerprint changed')
    write(OUTPUT / 'environment_at_launch.json', dict(cpu_probe=actual, gpu_inventory=inventory,
        runtime={k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','PYTHONPATH']},
        historical_equality='unknown; explicitly authorized candidate'))
    return env


def run_native(entry, flags, cwd, log, env, locks, state, job):
    verify_sources(read(BASE / 'manifest.json'))
    command = [PYTHON, '-u', str(BASE / 'bootstrap.py'), entry] + flags
    state.update(phase='running', current_job=job, command=command, current_output=str(cwd))
    write(OUTPUT / 'status.json', state)
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, pass_fds=tuple(h.fileno() for h in locks))
        state['worker_pid'] = child.pid
        write(OUTPUT / 'status.json', state)
        code = child.wait()
    state.pop('worker_pid', None)
    if code:
        raise RuntimeError(f'{job} exited {code}; no automatic retries or protocol fallback; see {log}')


def main():
    manifest = read(BASE / 'manifest.json')
    verify_sources(manifest)
    review = read(ROOT / 'experiments/p0B_legacy_replay_20261008/environment_review.json')
    if review['state'] != 'user_authorized_candidate' or not review.get('evidence_or_user_direction'):
        raise RuntimeError('Explicit candidate authorization missing')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    state = dict(phase='waiting_r1', pid=os.getpid(), completed=[], physical_gpu=1,
        started_at=datetime.now().astimezone().isoformat(),
        authority=manifest['authority'], historical_alignment='candidate, not proven exact',
        queued_jobs=['Token_seed45'] + [f'Modality_seed{s}' for s in manifest['modality']['seeds']] + ['Missing_ensemble','Full_ensemble'])
    write(OUTPUT / 'status.json', state)
    shutil.copy2(BASE / 'manifest.json', OUTPUT / 'frozen_manifest.json')
    shutil.copy2(ROOT / 'experiments/p0B_legacy_replay_20261008/environment_review.json', OUTPUT / 'review_at_queue.json')
    locks = []
    try:
        captured = read(R1_OUTPUT / 'status.json')
        r1_pids = [captured.get(k) for k in ['pid','worker_pid'] if captured.get(k)]
        while True:
            r1 = read(R1_OUTPUT / 'status.json')
            if r1['phase'] == 'failed':
                raise RuntimeError('R1 failed; R2 not started automatically. Inspect R1 and request direction')
            if r1['phase'] == 'complete' and not any(alive(pid) for pid in r1_pids):
                break
            if r1['phase'] != 'complete' and not any(alive(pid) for pid in r1_pids):
                raise RuntimeError('R1 launcher/worker exited without successful completion')
            time.sleep(10)
        locks = acquire_idle_gpus()
        verify_sources(manifest)
        verify_inputs(manifest)
        env = environment(manifest)
        # The original native entry performs its original seed44 smoke before seed45.
        token = OUTPUT / 'token'
        run_native('train_token_input.py', ['--output', str(token), '--device', 'cuda:0',
            '--seeds','45','--epochs','60','--min-epochs','20','--patience','15',
            '--batch-size','8','--eval-batch-size','16','--lr','0.0003'],
            OUTPUT, OUTPUT / 'token.log', env, locks, state, 'Token_seed45')
        if read(token / 'status.json')['phase'] != 'complete' or not read(token / 'seed45/test.json'):
            raise RuntimeError('Missing Token completion/test')
        state['completed'].append('Token_seed45')
        write(OUTPUT / 'status.json', state)
        config = str(BASE / 'source/configs/umrm_v3_config.json')
        checkpoints = {}
        for seed in manifest['modality']['seeds']:
            folder = OUTPUT / f'modality_seed{seed}'
            folder.mkdir(exist_ok=False)
            write(folder / 'status.json', dict(phase='starting', seed=seed))
            flags = ['--model','umrmv3','--config',config,'--dataset','MOSI',
                '--epochs','60','--batch-size','8','--seed',str(seed),'--device','cuda:0',
                '--kd','--kd-weight','0.3','--text-recon','--text-inject','--num-layers','2',
                '--uni-weight','0.1','--rcd-weight','0.1','--skip-extra-eval'] + MODEL_FLAGS
            run_native('train.py', flags, folder, folder / 'console.log', env, locks, state, f'Modality_seed{seed}')
            result = folder / 'results' / (TAG.format(seed) + '.json')
            report = read(result)
            if not math.isfinite(float(report['TAV']['MAE'])):
                raise RuntimeError('Invalid native test metric')
            cp = folder / 'checkpoints' / (TAG.format(seed) + '_best_MAE.pth')
            if not cp.is_file():
                raise RuntimeError('Missing native MAE checkpoint')
            checkpoints[seed] = cp
            shutil.copy2(result, folder / 'test.json')
            write(folder / 'status.json',dict(phase='complete', seed=seed, checkpoint=str(cp),
                test_sha256=sha(folder / 'test.json')))
            state['completed'].append(f'Modality_seed{seed}')
            write(OUTPUT / 'status.json',state)
        for name, members in [('Missing',manifest['ensembles']['missing']),('Full',manifest['ensembles']['full'])]:
            flags = ['--config',config,'--dataset','MOSI','--device','cuda:0',
                '--output',str(OUTPUT / f'{name.lower()}_ensemble.json'),
                '--checkpoints'] + [str(checkpoints[s]) for s in members] + MODEL_FLAGS
            run_native('reeval_ensemble.py', flags, OUTPUT, OUTPUT / f'{name.lower()}_ensemble.log',
                env, locks, state, name + '_ensemble')
            report = read(OUTPUT / f'{name.lower()}_ensemble.json')
            if not math.isfinite(float(report['TAV']['MAE'])):
                raise RuntimeError('Invalid ensemble result')
            state['completed'].append(name + '_ensemble')
            write(OUTPUT / 'status.json',state)
        state.update(phase='complete', finished_at=datetime.now().astimezone().isoformat())
        write(OUTPUT / 'status.json',state)
    except BaseException:
        state.update(phase='failed', traceback=traceback.format_exc())
        write(OUTPUT / 'status.json',state)
        raise
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
