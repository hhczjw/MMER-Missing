"""Serialize archived-checkpoint evaluation and two-step diagnostic after GPU idle."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/reproduction_20261009/r1_verification_01'


def write(value):
    value['updated_at'] = datetime.now().isoformat()
    temporary = OUTPUT / 'status.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(OUTPUT / 'status.json')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    jobs = [
        ('archived_evaluation', ROOT / 'scripts/reproduction_20261009/verify_r1_checkpoint.py',
         ['--output', str(OUTPUT / 'archived_evaluation'), '--wait-for-idle']),
        ('two_step_parity', ROOT / 'scripts/reproduction_20261008/diagnose_r1.py',
         ['--output', str(OUTPUT / 'two_step_parity'), '--steps', '2', '--batch-size', '4']),
    ]
    state = dict(phase='waiting_or_verifying', pid=os.getpid(), completed=[],
                 physical_gpu=1, training_scope='two diagnostic optimizer steps only')
    try:
        for name, script, flags in jobs:
            state.update(current_job=name)
            write(state)
            with (OUTPUT / (name + '.log')).open('x') as stream:
                process = subprocess.Popen([sys.executable, '-u', str(script), *flags],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                state['worker_pid'] = process.pid
                write(state)
                code = process.wait()
            if code:
                raise RuntimeError(f'{name} failed with exit {code}; see {name}.log')
            state['completed'].append(name)
        state.update(phase='complete')
        state.pop('worker_pid', None)
        write(state)
    except BaseException as error:
        state.update(phase='failed', error=repr(error))
        write(state)
        raise


if __name__ == '__main__':
    main()
