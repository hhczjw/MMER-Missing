"""Detach one GPU0 queue with a persistent log; no shell interpolation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from plan import ROOT,atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'results/mosi_ablations_gpu0_20261006')
    p.add_argument('--checkpoints',type=Path,default=Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints/mosi_ablations_gpu0_20261006'))
    p.add_argument('--log',type=Path,default=ROOT/'logs/mosi_ablations_gpu0_20261006.log')
    cli=p.parse_args()
    for path in [cli.output,cli.checkpoints,cli.log]:
        if path.exists():raise FileExistsError(f'Exclusive launch target exists: {path}')
    cli.log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/'scripts/ablations/run_queue.py'),'--output',str(cli.output),'--checkpoints',str(cli.checkpoints)]
    with cli.log.open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,command=command,log=str(cli.log),output=str(cli.output),checkpoints=str(cli.checkpoints))
    atomic_json(cli.log.with_suffix('.launch.json'),record)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
