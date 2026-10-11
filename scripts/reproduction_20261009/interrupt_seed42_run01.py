"""Explicitly authorized interruption of this one captured seed42 run; preserve files."""
import argparse
import ctypes
from datetime import datetime
import os
from pathlib import Path
import platform
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reproduction_20261008'))
from pause_after_current import identity, same_alive, open_pid_handle, read, write

OUTPUT = ROOT / 'results/p0B_legacy_replay_20261009_seed42/run_01'
PARENT = 558094
WORKER = 563353
AUTHORITY = '2026-10-09 human: 可以先中断训练，释放锁，让研究二的消融实验占用GPU0，然后在GPU1中重新训练研究一'


def send(fd, signum):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, signum)
        return
    if platform.machine() != 'x86_64':
        raise RuntimeError('Unsupported ABI; refuse unsafe PID-only signal')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(signum),
                          ctypes.c_void_p(0), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interrupt', action='store_true')
    cli = parser.parse_args()
    state = read(OUTPUT / 'status.json')
    parent, worker = identity(PARENT), identity(WORKER)
    if (state['phase'] != 'train' or state['pid'] != PARENT or state['worker_pid'] != WORKER
            or not parent or not worker or worker['parent'] != PARENT
            or not parent['command'].endswith('scripts/reproduction_20261009/run_p0b_seed42.py ')
            or not worker['command'].endswith('scripts/reproduction_20261009/bootstrap_p0b_seed42.py ')):
        raise RuntimeError('Captured run identity mismatch; no signals sent')
    fd = open_pid_handle(WORKER)
    try:
        if not same_alive(WORKER, worker):
            raise RuntimeError('Worker changed before signal')
        send(fd, 0)
        print('Captured worker/parent verified; PID-reuse-safe signal supported', flush=True)
        if not cli.interrupt:
            return
        receipt = OUTPUT / 'user_interruption.json'
        if receipt.exists():
            raise RuntimeError('Interruption already recorded; no duplicate action')
        record = dict(phase='interrupt_requested', authority=AUTHORITY, previous_status=state,
            controller_identity=parent, worker_identity=worker, requested_at=datetime.now().astimezone().isoformat(),
            logs_and_checkpoints_preserved=True, restart_from_scratch_in='run_02')
        write(receipt, record)
        send(fd, signal.SIGINT)
        deadline = time.monotonic() + 30
        while same_alive(WORKER, worker) or same_alive(PARENT, parent):
            if time.monotonic() > deadline:
                record.update(phase='attention_required', reason='Target did not exit after SIGINT; no forced kill')
                write(receipt, record)
                raise RuntimeError('Target still alive; investigate rather than release/bypass locks')
            time.sleep(.25)
        final = read(OUTPUT / 'status.json')
        record.update(phase='interrupted', controller_exited=True, worker_exited=True,
            launcher_exit_record=final, interrupted_at=datetime.now().astimezone().isoformat())
        write(receipt, record)
        final.update(phase='interrupted_by_user', interruption_record=str(receipt),
            authority=AUTHORITY, reason='User requested fresh restart for independent per-GPU locks',
            updated_at=record['interrupted_at'], restart_output=str(OUTPUT.with_name('run_02')))
        write(OUTPUT / 'status.json', final)
        print('Captured run interrupted; old logs/checkpoints preserved; controller and worker exited', flush=True)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
