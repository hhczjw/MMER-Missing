"""One independent archiver for both GPUs; local files survive failures and restarts."""
import argparse
import fcntl
import json
from pathlib import Path
import os
import time
import traceback
from staging import CheckpointStaging, atomic_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, action='append', required=True)
    parser.add_argument('--state-directory', type=Path, default=ROOT / 'results/checkpoint_nvme_staging_20261007')
    cli = parser.parse_args()
    metadata = cli.state_directory
    metadata.mkdir(parents=True, exist_ok=True)
    lock = (metadata / 'archiver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = metadata / 'archiver_status.json'
    status = dict(phase='idle', pid=os.getpid(), outputs=[str(p) for p in cli.output],
                  started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(status_path, status)
    while True:
        candidates = []
        for output in cli.output:
            for receipt in output.glob('nvme_archive_*.json'):
                state = json.loads(receipt.read_text())
                if state['phase'] in ('staged', 'archiving', 'archive_failed'):
                    if state.get('retry_after', 0) <= time.time():
                        candidates.append((state.get('ready_at', ''), receipt, state))
        if not candidates:
            status.update(phase='idle', updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            atomic_json(status_path, status)
            # Exit only after both queues and all successful-job archives finish.
            transitions = [p / 'nvme_transition.json' for p in cli.output]
            if all(p.exists() and json.loads(p.read_text()).get('phase') == 'complete' for p in transitions):
                pending = []
                for output in cli.output:
                    pending += [p for p in output.glob('nvme_archive_*.json')
                                if json.loads(p.read_text())['phase'] != 'archived']
                if not pending:
                    status.update(phase='complete')
                    atomic_json(status_path, status)
                    return
            time.sleep(5)
            continue
        _, receipt, state = sorted(candidates, key=lambda row: (row[0], str(row[1])))[0]
        status.update(phase='archiving', receipt=str(receipt), staging=state['staging'],
                      destination=state['destination'], updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        atomic_json(status_path, status)
        try:
            helper = CheckpointStaging(state['destination'], state['staging'])
            helper.archive(receipt)
            print('ARCHIVED', receipt.name, flush=True)
        except Exception:
            state.update(phase='archive_failed', traceback=traceback.format_exc(),
                         retry_after=time.time() + 60, local_checkpoint_retained=True)
            atomic_json(receipt, state)
            print('ARCHIVE_FAILED_LOCAL_RETAINED', receipt.name, flush=True)


if __name__ == '__main__':
    main()
