"""Explicit GPU0 restart of the original validation-only Token optimization.

The original runner is not edited. Its AST changes only the lock/device/index
labels; all training, seeds, candidates, losses and selection remain identical.
Fresh smoke and all eight jobs, no old checkpoint/optimizer/result reuse.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'scripts/research2_ablations/optimize_token_20261010.py'
OUTPUT = ROOT/'results/research2_token_optimization_20261010'
RECEIPT = ROOT/'results/reproduction_20261010/r2_gpu0_restart.json'
PYTHON = '/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def transformed(source):
    tree = ast.parse(source,filename=str(SOURCE))
    changes = []
    lock = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu_lock')
    for node in ast.walk(lock):
        if isinstance(node,ast.Constant):
            if node.value=='results/research2_gpu1.lock':
                changes.append((node.lineno,node.value,'results/research2_gpu0.lock'))
                node.value='results/research2_gpu0.lock'
            elif node.value=='1':
                changes.append((node.lineno,node.value,'0'));node.value='0'
            elif node.value=='GPU1 occupied; will not interrupt other jobs':
                changes.append((node.lineno,node.value,'GPU0 occupied; will not interrupt other jobs'))
                node.value='GPU0 occupied; will not interrupt other jobs'
    run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    for node in ast.walk(run):
        if isinstance(node,ast.keyword) and node.arg=='CUDA_VISIBLE_DEVICES':
            assert isinstance(node.value,ast.Constant) and node.value.value=='1'
            changes.append((node.lineno,'CUDA_VISIBLE_DEVICES=1','CUDA_VISIBLE_DEVICES=0'))
            node.value.value='0'
        if isinstance(node,ast.keyword) and node.arg=='gpu':
            assert isinstance(node.value,ast.Constant) and node.value.value==1
            changes.append((node.lineno,'state.gpu=1','state.gpu=0'))
            node.value.value=0
    assert len(changes)==5, changes
    ast.fix_missing_locations(tree)
    return tree,changes


def load_native(receipt):
    if sha(SOURCE)!=receipt['source_sha256'] or sha(__file__)!=receipt['adapter_sha256']:
        raise RuntimeError('Restart source/adapter changed; no training')
    tree,changes=transformed(SOURCE.read_text())
    if [list(c) for c in changes]!=receipt['only_changes']:
        raise RuntimeError('Unexpected GPU adapter changes')
    scope={'__file__':str(SOURCE),'__name__':'gpu0_native_token_optimization'}
    exec(compile(tree,str(SOURCE),'exec'),scope)
    return scope


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['verify','smoke','run','queue'],required=True)
    cli=parser.parse_args()
    receipt=json.loads(RECEIPT.read_text())
    native=load_native(receipt)
    if cli.mode=='verify':
        assert native['CANDIDATES']==receipt['candidates'] and native['SEEDS']==[42,45]
        print('CPU AST verification: only five GPU lock/device/label changes; original protocol intact')
        return
    if cli.mode in ('smoke','run'):
        native['run'](cli.mode=='smoke')
        return
    status_path=OUTPUT/'gpu0_restart_status.json'
    if status_path.exists() or (OUTPUT/'status.json').exists() or (OUTPUT/'smoke.json').exists():
        raise RuntimeError('Existing restart detected; refusing duplicate launch')
    state=dict(phase='smoke',pid=os.getpid(),gpu=0,from_scratch=True,
               original_runner_sha256=receipt['source_sha256'],adapter_sha256=receipt['adapter_sha256'],
               authority=receipt['authority'],started_at=datetime.now().astimezone().isoformat())
    write(status_path,state)
    try:
        for mode in ('smoke','run'):
            state.update(phase='smoke' if mode=='smoke' else 'train')
            write(status_path,state)
            log=OUTPUT/('gpu0_'+mode+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'--mode',mode],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
                    stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state['worker_pid']=child.pid;write(status_path,state)
                code=child.wait()
            state.pop('worker_pid',None)
            if code:
                raise RuntimeError(f'GPU0 {mode} exited{code}; no retry or changed protocol')
        state.update(phase='complete',finished_at=datetime.now().astimezone().isoformat())
        write(status_path,state)
    except BaseException:
        state.update(phase='failed',error=traceback.format_exc())
        write(status_path,state)
        raise


if __name__=='__main__':
    main()
