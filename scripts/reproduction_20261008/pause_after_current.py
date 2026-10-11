"""Stop queue scheduling only; let captured worker finish, then release controller locks."""
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import platform
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
WORKS = [('mosi_ablations_p0b_0919_gradient_repair_20261007', 'completed_train'),
         ('research2_ablations_gpu1_20261006_v3', 'completed')]
AUTHORITY = 'Human request: 等GPU1和GPU0的当前任务结束后，暂停work'


def open_pid_handle(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # The installed Python lacks os.pidfd_open and glibc2.31 lacks its wrapper.
    # Linux x86_64 syscall434 returns a PID-reuse-safe handle; fail on other ABIs.
    if platform.machine() != 'x86_64':
        raise RuntimeError('pidfd_open unavailable on this architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(start=fields[19], state=fields[0], parent=int(fields[1]),
                    command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def same_alive(pid, captured):
    value = identity(pid)
    return value is not None and value['start'] == captured['start'] and value['state'] != 'Z'


def capture(folder, completed_key):
    status = read(folder / 'status.json')
    active = read(folder / 'active_worker.json')
    controller_pid, worker_pid = status['pid'], active['pid']
    controller, worker = identity(controller_pid), identity(worker_pid)
    if status['phase'] != 'train' or not controller or not worker:
        raise RuntimeError(f'Expected live train queue: {folder.name}')
    if worker['parent'] != controller_pid or controller_pid == worker_pid:
        raise RuntimeError('Controller/worker parentage mismatch; no signals sent')
    if str(folder) not in controller['command'] or not any(x in controller['command'] for x in
            ['checkpoint_nvme_20261007/transition.py', 'checkpoint_nvme_20261007/resume_gpu1.py', 'checkpoint_nvme_20261007/resume_gpu0.py']):
        raise RuntimeError('Unexpected controller identity')
    if str(folder) not in worker['command'] or active['job'] not in worker['command']:
        raise RuntimeError('Unexpected active worker identity')
    if active['job'] != status['current_job'] or active['job'] in status[completed_key]:
        raise RuntimeError('Active job changed or is already counted')
    return dict(work=folder.name, completed_key=completed_key, current_job=active['job'],
                controller_pid=controller_pid, controller_identity=controller,
                worker_pid=worker_pid, worker_identity=worker,
                plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),
                phase='waiting_current_worker', authority=AUTHORITY)


def validated_result(folder, job):
    marker = read(folder / 'train' / job / 'status.json')
    result_path = folder / 'train' / job / 'test.json'
    result = read(result_path)
    if marker.get('phase') != 'complete' or not result:
        raise RuntimeError(f'Current worker did not complete successfully: {job}')
    if folder.name.startswith('mosi_') and not result.get('audited'):
        raise RuntimeError(f'Current R1 result is not audited: {job}')
    return hashlib.sha256(result_path.read_bytes()).hexdigest()


def finish_one(folder, row, controller_fd):
    result_hash = validated_result(folder, row['current_job'])
    if same_alive(row['controller_pid'], row['controller_identity']):
        if identity(row['controller_pid'])['state'] not in ('T', 't'):
            raise RuntimeError('Controller unexpectedly resumed; refuse to signal it')
        # No group signals. Worker already exited; TERM remains pending until CONT.
        signal.pidfd_send_signal(controller_fd, signal.SIGTERM)
        signal.pidfd_send_signal(controller_fd, signal.SIGCONT)
        until = time.monotonic() + 10
        while same_alive(row['controller_pid'], row['controller_identity']):
            if time.monotonic() > until:
                raise RuntimeError('Controller did not release after successful worker')
            time.sleep(.1)
    queue = read(folder / 'status.json')
    if queue['current_job'] != row['current_job']:
        raise RuntimeError('Unexpected job advance during pause')
    if row['current_job'] not in queue[row['completed_key']]:
        queue[row['completed_key']].append(row['current_job'])
    row.update(phase='paused', current_worker_result_sha256=result_hash,
               paused_at=datetime.now().astimezone().isoformat(), controller_exited=True,
               worker_exited=True, remaining_jobs_retained=True)
    queue.update(phase='paused', pause_request=str(folder/'pause_request.json'),
                 pause_reason=AUTHORITY, paused_at=row['paused_at'])
    write(folder / 'status.json', queue)
    transition = read(folder / 'nvme_transition.json')
    transition.update(phase='paused', pause_request=str(folder/'pause_request.json'),
                      paused_at=row['paused_at'], remaining_jobs_retained=True)
    write(folder / 'nvme_transition.json', transition)
    write(folder / 'pause_request.json', row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--receipt', type=Path)
    cli = parser.parse_args()
    rows = [capture(ROOT/'results'/name, key) for name,key in WORKS]
    if not cli.arm:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if cli.receipt is None or cli.receipt.exists():
        raise RuntimeError('Arm requires a fresh explicit receipt path')
    cli.receipt.parent.mkdir(parents=True, exist_ok=True)
    fds = [open_pid_handle(row['controller_pid']) for row in rows]
    stopped = []
    armed = False
    try:
        for row,fd in zip(rows,fds):
            if not same_alive(row['controller_pid'],row['controller_identity']):
                raise RuntimeError('Controller identity changed before stop')
            signal.pidfd_send_signal(fd, signal.SIGSTOP)
            stopped.append(fd)
        for row in rows:
            folder=ROOT/'results'/row['work']
            if read(folder/'active_worker.json')['pid'] != row['worker_pid'] or read(folder/'status.json')['current_job'] != row['current_job']:
                raise RuntimeError('Worker advanced while arming; roll back controller stop')
            if identity(row['worker_pid'])['state'] in ('T','t'):
                raise RuntimeError('Worker must not be stopped')
            row.update(monitor_pid=os.getpid(),requested_at=datetime.now().astimezone().isoformat())
            write(folder/'pause_request.json',row)
        armed=True
        receipt=dict(phase='waiting_current_workers', pid=os.getpid(), authority=AUTHORITY,
                     controllers_only_stopped=True, workers_untouched=True, works=rows)
        write(cli.receipt,receipt)
        print('ARMED',json.dumps(rows,ensure_ascii=False),flush=True)
        while any(row['phase']=='waiting_current_worker' for row in rows):
            for row,fd in zip(rows,fds):
                if row['phase']!='waiting_current_worker':continue
                folder=ROOT/'results'/row['work']
                if not same_alive(row['worker_pid'],row['worker_identity']):
                    try:
                        finish_one(folder,row,fd)
                    except BaseException:
                        row.update(phase='pause_failed',traceback=traceback.format_exc(),
                                   controller_left_stopped=True)
                        write(folder/'pause_request.json',row)
                    write(cli.receipt,receipt)
            if any(row['phase']=='waiting_current_worker' for row in rows):time.sleep(5)
        receipt['phase']='paused' if all(row['phase']=='paused' for row in rows) else 'failed'
        write(cli.receipt,receipt)
    except BaseException:
        if not armed:
            for fd in stopped:
                try:signal.pidfd_send_signal(fd,signal.SIGCONT)
                except ProcessLookupError:pass
        raise
    finally:
        for fd in fds:os.close(fd)


if __name__ == '__main__':
    main()
