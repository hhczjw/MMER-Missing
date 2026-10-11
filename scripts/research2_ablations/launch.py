"""Detach the user-authorized GPU1 ablation queue with a persistent log."""
import os
import argparse
from pathlib import Path
import subprocess
import sys
from plan import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='research2_ablations_gpu1_20261006')
    name=ap.parse_args().name
    if '/' in name or name.startswith('.'):raise ValueError(name)
    out=ROOT/'results'/name
    ckpt=Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints')/name
    log=ROOT/'logs'/(name+'.log')
    for p in [out,ckpt,log]:
        if p.exists():raise FileExistsError(p)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               R2_SNAPSHOT_ROOT=str(ROOT/'experiments/research2_ablations_gpu1_20261006/snapshot'))
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_queue.py')),
               '--output',str(out),'--checkpoints',str(ckpt)],cwd=ROOT,env=env,
               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print('LAUNCHED',child.pid,'LOG',log,'OUTPUT',out,'CHECKPOINTS',ckpt,flush=True)


if __name__=='__main__':main()
