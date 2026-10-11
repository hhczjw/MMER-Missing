"""Fail closed: full completion, or explicit user-authorized successful boundary pause."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORKS = [
    ('mosi_ablations_p0b_0919_gradient_repair_20261007', 'completed_train'),
    ('research2_ablations_gpu1_20261006_v3', 'completed'),
]


def read(path):
    return json.loads(path.read_text())


def live(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def check_queues(root=ROOT):
    evidence = []
    policy_path = root / 'scripts/reproduction_20261008/scheduling_policy.json'
    policy = read(policy_path) if policy_path.exists() else {}
    for name, key in WORKS:
        folder = root / 'results' / name
        status = read(folder / 'status.json')
        plan = read(folder / 'plan.json')
        expected = {job['job_id'] for job in plan['jobs']}
        completed = set(status.get(key, []))
        paused = status.get('phase') == 'paused'
        if paused:
            if policy.get('execution_gate') != 'complete_or_explicitly_paused_after_current_success':
                raise RuntimeError(f'Deferred: {name} paused without human scheduling authorization')
            request = read(folder / 'pause_request.json')
            job = request.get('current_job')
            if (request.get('phase') != 'paused' or request.get('authority') != policy.get('authority')
                    or job != status.get('current_job') or job not in completed
                    or not request.get('controller_exited') or not request.get('worker_exited')):
                raise RuntimeError(f'Deferred: {name} pause is not a verified successful job boundary')
            if request.get('plan_sha256') != hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest():
                raise RuntimeError(f'Deferred: {name} plan changed after pause')
            result_path = folder / 'train' / job / 'test.json'
            if request.get('current_worker_result_sha256') != hashlib.sha256(result_path.read_bytes()).hexdigest():
                raise RuntimeError(f'Deferred: {name} paused worker result changed')
        elif status.get('phase') != 'complete' or not expected.issubset(completed):
            raise RuntimeError(f'Deferred: full work {name} is not complete '
                               f'({len(expected & completed)}/{len(expected)})')
        if not completed.issubset(expected):
            raise RuntimeError(f'Unknown completed jobs: {name}')
        if any(live(status.get(k, -1)) for k in ['pid', 'active_worker_pid']):
            raise RuntimeError(f'Deferred: {name} controller/worker is still alive')
        for job in completed if paused else expected:
            marker = read(folder / 'train' / job / 'status.json')
            result = read(folder / 'train' / job / 'test.json')
            if marker.get('phase') != 'complete' or not result:
                raise RuntimeError(f'Missing successful completion: {job}')
        evidence.append(dict(work=name, jobs=len(expected), completed=len(completed),
                             phase=status['phase']))
    return evidence


def acquire_idle_gpus():
    """Caller must retain returned handles for the entire diagnostic process."""
    check_queues()
    handles = []
    try:
        for path in [Path('/tmp/mmer_ablation_physical_gpu0.lock'),
                     ROOT / 'results/research2_gpu1.lock']:
            handle = path.open('a')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_queues()
        output = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'
        ], text=True, timeout=10).strip()
        if output:
            raise RuntimeError(f'Deferred: GPUs still have compute processes: {output}')
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


if __name__ == '__main__':
    print(json.dumps(check_queues(), indent=2))
