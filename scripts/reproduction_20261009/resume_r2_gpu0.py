"""Resume only remaining paused R2 ablations on GPU0; preserve all completed jobs."""
from datetime import datetime
import fcntl
import hashlib
import importlib.util
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
OUTPUT = ROOT / 'results/research2_ablations_gpu1_20261006_v3'
RECEIPT = OUTPUT / 'gpu0_resume_20261009'
HERE = ROOT / 'scripts/checkpoint_nvme_20261007'
ADAPTER = Path(__file__).with_name('r2_gpu0_worker.py')
LOCAL = ROOT / 'results/checkpoint_nvme_staging_20261007' / OUTPUT.name
AUTHORITY = '2026-10-09 human: 继续进行研究二的消融实验，这次跑GPU0上。'
GPU0_UUID = 'GPU-07abbe75-7a55-4623-70aa-a9572c83957e'
sys.path.insert(0, str(HERE))
from staging import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def validate_queue(plan, queue):
    if queue['phase'] != 'paused':
        raise RuntimeError('Expected paused R2 queue; no duplicate controller allowed')
    if any(alive(queue.get(k, -1)) for k in ['pid', 'active_worker_pid']):
        raise RuntimeError('Previous R2 controller/worker still alive')
    expected = {job['job_id'] for job in plan['jobs']}
    completed = queue['completed']
    if len(completed) != len(set(completed)) or not set(completed).issubset(expected):
        raise RuntimeError('Invalid completed-job list')
    for name in completed:
        folder = OUTPUT / 'train' / name
        report = read(folder / 'test.json')
        if read(folder / 'status.json')['phase'] != 'complete' or not report:
            raise RuntimeError(f'Invalid completed job: {name}')
        values = report['missing_avg'] if name.startswith('R2-M') else report['avg_over_rates']
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise RuntimeError(f'Invalid completed metrics: {name}')
    remaining = [job for job in plan['jobs'] if job['job_id'] not in completed]
    for job in remaining:
        name = job['job_id']
        paths = [OUTPUT / 'train' / name, OUTPUT / 'logs' / f'{name}.log', LOCAL / name,
                 Path(plan['checkpoint_root']) / job['track'] / name,
                 OUTPUT / f'nvme_archive_{name}.json']
        if any(path.exists() for path in paths):
            raise RuntimeError(f'Uncompleted job already has artifacts: {name}; no overwrite/retry')
    return remaining


def gpu0_idle():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader'], text=True, timeout=10)
    gpu0 = next(line for line in inventory.splitlines() if line.split(',')[0].strip() == '0')
    if GPU0_UUID not in gpu0:
        raise RuntimeError('Physical GPU0 identity changed')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader'], text=True, timeout=10)
    return not any(GPU0_UUID in line for line in processes.splitlines())


def main():
    plan = read(OUTPUT / 'plan.json')
    queue = read(OUTPUT / 'status.json')
    remaining = validate_queue(plan, queue)
    if not remaining:
        raise RuntimeError('No remaining R2 ablations')
    # Exclusive receipt registration prevents duplicate resume controllers.
    RECEIPT.mkdir(exist_ok=False)
    for name in ['status.json', 'plan.json', 'pause_request.json', 'nvme_transition.json', 'active_worker.json']:
        shutil.copy2(OUTPUT / name, RECEIPT / ('prior_' + name))
    helpers = {str(p):sha(p) for p in [Path(__file__), ADAPTER, HERE / 'worker.py',
                                      HERE / 'data_cache.py', HERE / 'staging.py']}
    state = dict(phase='waiting_gpu0_lock', pid=os.getpid(), physical_gpu=0,
        authority=AUTHORITY, previously_completed=len(queue['completed']),
        remaining=[job['job_id'] for job in remaining], started_at=datetime.now().astimezone().isoformat(),
        source_sha256=plan['source_sha256'], helper_sha256=helpers, plan_sha256=sha(OUTPUT / 'plan.json'),
        protocol_changes='Physical GPU1 -> GPU0 only; device assertion/metadata adapted, training unchanged',
        completed_jobs_preserved=True, old_pause_request_preserved_as_history=True)
    atomic_json(RECEIPT / 'status.json', state)
    lock = Path('/tmp/mmer_ablation_physical_gpu0.lock').open('a')
    mutated_queue = False
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(15)
                continue
            if gpu0_idle():
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(15)
        queue = read(OUTPUT / 'status.json')
        if validate_queue(plan, queue) != remaining or sha(OUTPUT / 'plan.json') != state['plan_sha256']:
            raise RuntimeError('Queue/plan changed while waiting')
        for name, expected in helpers.items():
            if sha(name) != expected:
                raise RuntimeError(f'Resume helper changed: {name}')
        # Import original utilities CPU-only and install original verified input cache.
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
            os.environ.pop(key, None)
        from data_cache import install_cache
        if not install_cache(ROOT / 'results/checkpoint_nvme_staging_20261007/input_cache/index.json'):
            raise RuntimeError('Original input cache unavailable')
        queue_file = ROOT / 'scripts/research2_ablations/run_queue.py'
        sys.path.insert(0, str(queue_file.parent))
        spec = importlib.util.spec_from_file_location('r2_original_queue_gpu0', queue_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.assert_frozen(plan)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        queue.update(phase='train', pid=os.getpid(), physical_gpu=0, controller=str(Path(__file__)),
            resume_record=str(RECEIPT / 'status.json'), resume_authority=AUTHORITY)
        mutated_queue = True
        atomic_json(OUTPUT / 'status.json', queue)
        state['phase'] = 'train'
        atomic_json(RECEIPT / 'status.json', state)
        for job in remaining:
            name = job['job_id']
            module.assert_frozen(plan)
            while shutil.disk_usage(LOCAL.parent).free < 12 * 1024**3:
                state.update(phase='waiting_nvme_space', current_job=name)
                atomic_json(RECEIPT / 'status.json', state)
                time.sleep(15)
            destination = Path(plan['checkpoint_root']) / job['track'] / name
            receipt = OUTPUT / f'nvme_archive_{name}.json'
            log = OUTPUT / 'logs' / f'{name}.log'
            command = [sys.executable, str(HERE / 'worker.py'), '--destination', str(destination),
                '--local', str(LOCAL / name), '--receipt', str(receipt), str(ADAPTER),
                '--output', str(OUTPUT), '--job', name]
            queue.update(phase='train', current_job=name)
            state.update(phase='train', current_job=name)
            atomic_json(OUTPUT / 'status.json', queue)
            atomic_json(RECEIPT / 'status.json', state)
            print('START_GPU0', name, flush=True)
            with log.open('x') as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                queue['active_worker_pid'] = process.pid
                state['active_worker_pid'] = process.pid
                atomic_json(OUTPUT / 'status.json', queue)
                atomic_json(RECEIPT / 'status.json', state)
                atomic_json(OUTPUT / 'active_worker.json', dict(pid=process.pid, job=name,
                    physical_gpu=0, stage='train', log=str(log), command=command, checkpoint_storage='NVMe'))
                code = process.wait()
            if code or read(OUTPUT / 'train' / name / 'status.json')['phase'] != 'complete':
                raise RuntimeError(f'Worker failed {name}, exit={code}; no automatic retry')
            if read(receipt)['phase'] not in ('staged','archiving','archived','archive_failed'):
                raise RuntimeError('Missing completed NVMe staging receipt')
            if not read(OUTPUT / 'train' / name / 'test.json'):
                raise RuntimeError('Missing test result')
            queue['completed'].append(name)
            module.aggregate(OUTPUT, plan)
            atomic_json(OUTPUT / 'status.json', queue)
            state['completed_this_resume'] = [x for x in state['remaining'] if x in queue['completed']]
            atomic_json(RECEIPT / 'status.json', state)
            print('DONE_GPU0', name, flush=True)
        queue.update(phase='complete', finished_at=datetime.now().astimezone().isoformat())
        state.update(phase='complete', finished_at=queue['finished_at'])
        atomic_json(OUTPUT / 'status.json', queue)
        atomic_json(RECEIPT / 'status.json', state)
    except BaseException:
        state.update(phase='failed', traceback=traceback.format_exc())
        atomic_json(RECEIPT / 'status.json', state)
        if mutated_queue:
            queue.update(phase='failed', traceback=state['traceback'])
            atomic_json(OUTPUT / 'status.json', queue)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
