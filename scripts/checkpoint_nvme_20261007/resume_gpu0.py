"""Resume remaining R1 experiments with user-authorized smaller batch and NVMe staging."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from staging import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--r1-05-batch-size', type=int, help='Explicit batch override only for R1-05 across seeds')
    cli = parser.parse_args()
    if cli.batch_size < 1 or (cli.r1_05_batch_size is not None and cli.r1_05_batch_size < 1):
        raise ValueError('batch-size must be positive')
    output = cli.output.resolve()
    lock = open('/tmp/mmer_ablation_physical_gpu0.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = read(output / 'active_worker.json')
    if (Path('/proc') / str(active['pid'])).exists():
        raise RuntimeError('Prior GPU0 worker is still running')
    plan = read(output / 'plan.json')
    queue = read(output / 'status.json')
    if queue['current_job'] in queue['completed_train']:
        raise RuntimeError('Expected an incomplete failed current job')
    old_transition = read(output / 'nvme_transition.json')
    if old_transition['phase'] != 'failed':
        raise RuntimeError('Expected failed prior controller')
    queue_file = ROOT / 'experiments/ablation_p0b_0919_gradient_repair_20261007/runner/run_queue.py'
    sys.path.insert(0, str(queue_file.parent))
    spec = importlib.util.spec_from_file_location('original_r1_queue', queue_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from data_cache import install_cache
    install_cache(ROOT/'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
    module.assert_frozen(plan)
    local_root = ROOT / 'results/checkpoint_nvme_staging_20261007' / output.name
    if shutil.disk_usage(local_root.parent).free < 20 * 1024**3:
        raise RuntimeError('NVMe reserve below 12 GiB')
    failed_job = queue['current_job']
    archive = output / 'failed_attempts' / (time.strftime('%Y%m%d_%H%M%S') + '_' + failed_job + '_batch' + str(queue.get('batch_size',4)))
    archive.mkdir(parents=True, exist_ok=False)
    original_paths = {
        'train': output / 'train' / failed_job,
        'checkpoint_nvme': local_root / failed_job,
        'train.log': output / 'logs' / f'train_{failed_job}.log',
        'archive_receipt.json': output / f'nvme_archive_{failed_job}.json',
    }
    for label, source in original_paths.items():
        if source.exists():
            source.rename(archive / label)
    atomic_json(archive / 'controller.json', old_transition)
    atomic_json(archive / 'active_worker.json', active)
    if (output/'batch_recovery.json').exists():shutil.copy2(output/'batch_recovery.json',archive/'prior_batch_recovery.json')
    atomic_json(output / 'batch_recovery.json', dict(
        original_batch_size=plan['reference_args']['R1']['batch_size'], new_batch_size=cli.batch_size,
        preserved_failed_attempt=str(archive), restart_job=failed_job,
        restart_from='fresh original seed initialization to keep a single batch size throughout this experiment',
        per_variant_batch_overrides={'R1-05':cli.r1_05_batch_size} if cli.r1_05_batch_size else {},
        completed_jobs_retained=queue['completed_train'], applied_to='remaining GPU0 jobs only',
        training_protocol_change=True, no_gradient_accumulation=True,
        reason='explicit user instruction following CUDA OOM during emergency backward',
        started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    state = dict(phase='active', pid=os.getpid(), gpu=0, current_job=failed_job,
                 staging_root=str(local_root), destination_root=plan['checkpoint_root'],
                 batch_size=cli.batch_size, original_batch_size=4,
                 controller='resume_gpu0', queue_lock_acquired=True,
                 saving='NVMe with independent asynchronous HDD archive',
                 started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(output / 'nvme_transition.json', state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', PYTORCH_ALLOC_CONF='expandable_segments:True')
    queue.update(pid=os.getpid(), phase='train', batch_size=cli.batch_size,
                 controller=str(Path(__file__).resolve()),
                 checkpoint_storage='NVMe per-experiment staging', protocol_change_record=str(output / 'batch_recovery.json'))
    atomic_json(output / 'status.json', queue)
    for job in plan['jobs']:
        name = job['job_id']
        if name in queue['completed_train']:
            continue
        module.assert_frozen(plan)
        while shutil.disk_usage(local_root.parent).free < 20 * 1024**3:
            state.update(phase='waiting_nvme_space', current_job=name)
            atomic_json(output / 'nvme_transition.json', state)
            time.sleep(15)
        job_batch = cli.r1_05_batch_size if job['id']=='R1-05' and cli.r1_05_batch_size is not None else cli.batch_size
        destination = Path(plan['checkpoint_root']) / 'train' / name
        local = local_root / name
        receipt = output / f'nvme_archive_{name}.json'
        log = output / 'logs' / f'train_{name}.log'
        command = [sys.executable, str(HERE / 'worker.py'), '--destination', str(destination),
                   '--local', str(local), '--receipt', str(receipt), '--batch-size', str(job_batch),
                   str(HERE/'r1_worker_recovery.py'), '--job', job['id'], '--seed', str(job['seed']),
                   '--stage', 'train', '--output', str(output / 'train' / name),
                   '--checkpoints', str(destination), '--manifest', str(output / 'plan.json')]
        queue.update(phase='train', current_job=name, batch_size=job_batch)
        state.update(phase='active', current_job=name, batch_size=job_batch)
        atomic_json(output / 'status.json', queue)
        atomic_json(output / 'nvme_transition.json', state)
        print('START_NVME', name, 'batch', job_batch, flush=True)
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            queue.update(active_worker_pid=process.pid)
            atomic_json(output / 'status.json', queue)
            atomic_json(output / 'active_worker.json', dict(pid=process.pid, job=name, stage='train',
                        log=str(log), command=command, checkpoint_storage='NVMe', batch_size=job_batch))
            code = process.wait()
        if code or read(receipt)['phase'] not in ('staged', 'archiving', 'archived', 'archive_failed'):
            raise RuntimeError(f'Worker failed: {name} exit={code}; NVMe files retained')
        result = read(output / 'train' / name / 'test.json')
        if not result.get('audited') or read(output / 'train' / name / 'status.json')['phase'] != 'complete':
            raise RuntimeError('Expected audited completed experiment')
        queue['completed_train'].append(name)
        module.aggregate(output, plan)
        summary_path = output / 'summary.json'
        summary = read(summary_path)
        summary['training_batch_protocol'] = dict(original_batch_size=4, recovery_batch_size=cli.batch_size,
              change_record=str(output / 'batch_recovery.json'), mixed_batch_sizes=True)
        atomic_json(summary_path, summary)
        atomic_json(output / 'status.json', queue)
        print('DONE_ASYNC_ARCHIVE_QUEUED', name, flush=True)
    queue.update(phase='complete', finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    state.update(phase='complete', finished_at=queue['finished_at'])
    atomic_json(output / 'status.json', queue)
    atomic_json(output / 'nvme_transition.json', state)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            for name in ['nvme_transition.json', 'status.json']:
                path = output / name
                value = read(path) if path.exists() else {}
                value.update(phase='failed', traceback=error)
                atomic_json(path, value)
        raise
