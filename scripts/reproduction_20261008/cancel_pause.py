"""Cancel the authorized boundary-pause monitor; resume only captured schedulers."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import time

from pause_after_current import identity, open_pid_handle, read, write, same_alive, ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path,required=True)
    cli=parser.parse_args()
    receipt=read(cli.receipt)
    if receipt['phase']!='waiting_current_workers':
        raise RuntimeError('Pause already finalized; inspect before resuming')
    monitor=identity(receipt['pid'])
    if not monitor or 'reproduction_20261008/pause_after_current.py' not in monitor['command'] or str(cli.receipt.resolve()) not in monitor['command']:
        raise RuntimeError('Unexpected pause monitor identity')
    fds=[]
    try:
        for row in receipt['works']:
            if row['phase']!='waiting_current_worker' or not same_alive(row['controller_pid'],row['controller_identity']):
                raise RuntimeError('Controller already exited or identity changed')
            if identity(row['controller_pid'])['state'] not in ('T','t'):
                raise RuntimeError('Expected stopped scheduler')
            fds.append(open_pid_handle(row['controller_pid']))
        monitor_fd=open_pid_handle(receipt['pid'])
        fds.append(monitor_fd)
        if not same_alive(receipt['pid'],monitor):
            raise RuntimeError('Monitor identity changed')
        signal.pidfd_send_signal(monitor_fd,signal.SIGSTOP)
        # Kill only the cancellation monitor; never its process group or workers.
        signal.pidfd_send_signal(monitor_fd,signal.SIGTERM)
        signal.pidfd_send_signal(monitor_fd,signal.SIGCONT)
        until=time.monotonic()+10
        while same_alive(receipt['pid'],monitor):
            if time.monotonic()>until:raise RuntimeError('Pause monitor did not exit')
            time.sleep(.1)
        for row,fd in zip(receipt['works'],fds):
            signal.pidfd_send_signal(fd,signal.SIGCONT)
            row.update(phase='cancelled',cancelled_at=datetime.now().astimezone().isoformat(),
                       cancellation_authority='Human request: 算了，先等跑完所有work，再进行基线重跑吧。',
                       controller_resumed=True,workers_untouched=True)
            write(ROOT/'results'/row['work']/'pause_request.json',row)
        receipt.update(phase='cancelled',cancelled_at=datetime.now().astimezone().isoformat(),
                       cancellation_authority='Human request: 算了，先等跑完所有work，再进行基线重跑吧。',
                       monitor_terminated=True,controllers_resumed=True,workers_untouched=True)
        write(cli.receipt,receipt)
        print('Pause cancelled. Captured controllers resumed; workers untouched.')
    finally:
        for fd in fds:os.close(fd)


if __name__=='__main__':
    main()
