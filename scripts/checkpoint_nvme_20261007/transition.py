"""Adopt a live queue after its current worker, then stage subsequent jobs on NVMe."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from staging import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = stat[stat.rindex(')') + 2:].split()
        return fields[19], fields[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue-pid', type=int, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--batch-size', type=int, help='R1 batch size for every remaining job')
    cli = parser.parse_args()
    if cli.batch_size is not None and (cli.gpu != 0 or cli.batch_size < 1):
        raise ValueError('Positive batch override is restricted to GPU0 R1')
    output = cli.output.resolve()
    plan = read(output / 'plan.json')
    old_state = read(output / 'status.json')
    current = old_state['current_job']
    original_worker_identity = process_identity(cli.worker_pid)
    original_queue_identity = process_identity(cli.queue_pid)
    if original_worker_identity is None or original_queue_identity is None:
        raise RuntimeError('Original processes have changed; do not adopt stale queue')
    proc = Path('/proc') / str(cli.queue_pid)
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if not any(name in command for name in ['run_queue.py','checkpoint_nvme_20261007/transition.py','checkpoint_nvme_20261007/resume_gpu0.py']) or str(output) not in command:
        raise RuntimeError('Original queue command does not match')
    worker_command = (Path('/proc') / str(cli.worker_pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if current.split('_seed')[0] not in worker_command or str(output) not in worker_command:
        raise RuntimeError('Original worker command does not match')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cli.gpu), PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    allowed = {'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM',
               'R2_SNAPSHOT_ROOT', 'PYTHONPATH', 'CUDA_DEVICE_ORDER', 'PYTORCH_ALLOC_CONF'}
    for entry in (proc / 'environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            key, value = entry.decode().split('=', 1)
            if key in allowed:
                env[key] = value
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(key,None)
    lock_path = Path('/tmp/mmer_ablation_physical_gpu0.lock') if cli.gpu == 0 else ROOT / 'results/research2_gpu1.lock'
    transition = output / 'nvme_transition.json'
    local_root = ROOT / 'results/checkpoint_nvme_staging_20261007' / output.name
    local_root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(local_root.parent).free < 12 * 1024**3:
        raise RuntimeError('At least 12 GiB NVMe free space required')
    state = dict(phase='waiting_current_experiment', pid=os.getpid(), gpu=cli.gpu,
                 old_queue_pid=cli.queue_pid, current_worker_pid=cli.worker_pid,
                 current_job=current, staging_root=str(local_root),
                 destination_root=plan['checkpoint_root'],
                 saving='NVMe during training; independent asynchronous HDD archive after each experiment',
                 original_worker_untouched=True, started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    if cli.batch_size is not None:state['batch_size']=cli.batch_size
    atomic_json(transition, state)
    # Stop only the queue controller. Its worker has a separate PID and continues.
    if process_identity(cli.queue_pid) != original_queue_identity:
        raise RuntimeError('Queue identity changed')
    os.kill(cli.queue_pid, signal.SIGSTOP)
    try:
        if read(output / 'status.json')['current_job'] != current:
            raise RuntimeError('Queue advanced during handover')
        os.kill(cli.queue_pid, signal.SIGTERM)
    finally:
        # Deliver pending SIGTERM to the stopped controller, never its process group.
        try:
            os.kill(cli.queue_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    lock = lock_path.open('a')
    deadline = time.monotonic() + 20
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                raise RuntimeError('Original queue did not release its lock')
            time.sleep(.5)
    state['queue_lock_acquired'] = True
    atomic_json(transition, state)
    queue_state = read(output / 'status.json')
    queue_state.update(pid=os.getpid(), active_worker_pid=cli.worker_pid,
                       checkpoint_storage='NVMe switch after current experiment',
                       controller=str(Path(__file__).resolve()), staging_root=str(local_root))
    if cli.batch_size is not None:queue_state['batch_size']=cli.batch_size
    atomic_json(output / 'status.json', queue_state)
    while True:
        identity = process_identity(cli.worker_pid)
        if identity is None or identity[0] != original_worker_identity[0] or identity[1] == 'Z':
            break
        time.sleep(10)
    # A completed marker is required; never restart an unsuccessful current job.
    marker = output / 'train' / current / 'test.json'
    worker_state = read(output / 'train' / current / 'status.json')
    if not marker.exists() or worker_state['phase'] != 'complete':
        raise RuntimeError('Current worker did not finish successfully; retained all outputs')
    if cli.gpu == 0 and not read(marker).get('audited'):
        raise RuntimeError('Current R1 worker result is not audited')
    if cli.gpu == 0:
        queue_file = ROOT / 'experiments/ablation_p0b_0919_gradient_repair_20261007/runner/run_queue.py'
        completed_key = 'completed_train'
    else:
        queue_file = ROOT / 'scripts/research2_ablations/run_queue.py'
        completed_key = 'completed'
    sys.path.insert(0, str(queue_file.parent))
    spec = importlib.util.spec_from_file_location('original_queue', queue_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from data_cache import install_cache
    install_cache(ROOT/'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
    module.assert_frozen(plan)
    queue = read(output / 'status.json')
    queue.update(pid=os.getpid(), checkpoint_storage='NVMe per-experiment staging',
                 staging_root=str(local_root), controller=str(Path(__file__).resolve()))
    if current not in queue[completed_key]:
        queue[completed_key].append(current)
    atomic_json(output / 'status.json', queue)
    module.aggregate(output, plan)
    state.update(phase='active', current_experiment_completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(transition, state)
    for job in plan['jobs']:
        name = job['job_id']
        if name in queue[completed_key]:
            continue
        module.assert_frozen(plan)
        while shutil.disk_usage(local_root.parent).free < 12 * 1024**3:
            state.update(phase='waiting_nvme_space', current_job=name)
            atomic_json(transition, state)
            time.sleep(15)
        if cli.gpu == 0:
            destination = Path(plan['checkpoint_root']) / 'train' / name
            arguments = ['--job', job['id'], '--seed', str(job['seed']), '--stage', 'train',
                         '--output', str(output / 'train' / name), '--checkpoints', str(destination),
                         '--manifest', str(output / 'plan.json')]
            log = output / 'logs' / f'train_{name}.log'
        else:
            destination = Path(plan['checkpoint_root']) / job['track'] / name
            arguments = ['--output', str(output), '--job', name]
            log = output / 'logs' / f'{name}.log'
        receipt = output / f'nvme_archive_{name}.json'
        local = local_root / name
        command = [sys.executable, str(HERE / 'worker.py'), '--destination', str(destination),
                   '--local', str(local), '--receipt', str(receipt), str(queue_file.with_name('worker.py')), *arguments]
        if cli.gpu==0 and cli.batch_size is not None:
            worker_index=command.index(str(queue_file.with_name('worker.py')))
            command[worker_index:worker_index+1]=['--batch-size',str(cli.batch_size),str(HERE/'r1_worker_recovery.py')]
            queue['batch_size']=cli.batch_size
        queue.update(phase='train', current_job=name)
        atomic_json(output / 'status.json', queue)
        state.update(phase='active', current_job=name)
        atomic_json(transition, state)
        print('START_NVME', name, flush=True)
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(output / 'active_worker.json', dict(pid=process.pid, stage='train', job=name,
                        log=str(log), command=command, checkpoint_storage='NVMe',
                        batch_size=cli.batch_size if cli.gpu==0 else None))
            code = process.wait()
        if code or read(receipt)['phase'] not in ('staged', 'archiving', 'archived', 'archive_failed'):
            raise RuntimeError(f'Worker/archive failed: {name} exit={code}; NVMe files retained')
        if read(output / 'train' / name / 'status.json')['phase'] != 'complete':
            raise RuntimeError('Missing completed worker state')
        if cli.gpu == 0 and not read(output / 'train' / name / 'test.json').get('audited'):
            raise RuntimeError('Missing audited R1 result')
        queue[completed_key].append(name)
        module.aggregate(output, plan)
        atomic_json(output / 'status.json', queue)
        print('DONE_NVME_ARCHIVE_QUEUED', name, flush=True)
    queue.update(phase='complete', finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(output / 'status.json', queue)
    state.update(phase='complete', finished_at=queue['finished_at'])
    atomic_json(transition, state)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        # Preserve the original plan; only add transition diagnostics.
        try:
            index = sys.argv.index('--output')
            output = Path(sys.argv[index + 1])
            path = output / 'nvme_transition.json'
            state = read(path) if path.exists() else {}
            state.update(phase='failed', traceback=error)
            atomic_json(path, state)
        finally:
            raise
