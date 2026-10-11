"""Native Token42–47 extension: reuse completed45; wait R1, own GPU1 only."""
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import run_research2 as legacy
BASE = legacy.BASE
OUTPUT = ROOT / 'results/token_multiseed_42_47_20261009/run_01'
R1 = ROOT / 'results/p0B_legacy_replay_20261009_seed42/run_02'
OLD_TOKEN = ROOT / 'results/research2_legacy_replay_20261009/run_01/token'
SEEDS = [42,43,44,45,46,47]
NEW_SEEDS = [42,43,44,46,47]
AUTHORITY = '2026-10-09 user: Token添加训练，seed 42,43,44,45,46,47'
ENSEMBLE = Path(__file__).with_name('token_multiseed_ensemble.py')
CONFIG_CHECK = Path(__file__).with_name('token_multiseed_config_check.py')

def stamp():
    return datetime.now().astimezone().isoformat()

def read(p):
    return legacy.read(p)

def write(p,x):
    legacy.write(p,x)

def native_flags(seed,folder):
    return ['--output',str(folder),'--device','cuda:0','--seeds',str(seed),
        '--epochs','60','--min-epochs','20','--patience','15',
        '--batch-size','8','--eval-batch-size','16','--lr','0.0003']

def dependency_ready(state):
    if state.get('phase') in ('failed','interrupted_by_user'):
        raise RuntimeError('R1 did not complete successfully; Token queue will not start')
    pids=[state[k] for k in ('pid','worker_pid') if state.get(k)]
    if state.get('phase')!='complete':
        if not pids or not any(legacy.alive(pid) for pid in pids):
            raise RuntimeError('R1 lost before successful completion')
        return False
    if any(legacy.alive(pid) for pid in pids):
        return False
    result=read(R1/'results/MOSI_p0B_seed42_test.json')
    for key in ('primary_MAE','secondary_Acc2','tertiary_Acc7','quaternary_F1_has0'):
        if not math.isfinite(float(result[key]['MAE'])):
            raise RuntimeError('Invalid R1 completion/test')
    return True

def acquire_gpu1(manifest):
    lock=(ROOT/'results/research2_gpu1.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,driver_version','--format=csv,noheader'],text=True,timeout=10)
        row=next(s for s in inventory.splitlines() if s.split(',')[0].strip()=='1')
        if manifest['gpu1_uuid'] not in row or manifest['expected_environment']['driver'] not in row:
            raise RuntimeError('GPU1 identity/driver changed')
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
        if any(manifest['gpu1_uuid'] in s for s in processes.splitlines()):
            raise BlockingIOError('Physical GPU1 busy')
        return lock
    except BaseException:
        lock.close()
        raise

def verify_pin(plan):
    legacy.verify_sources(plan['frozen_manifest'])
    for name,digest in plan['helpers'].items():
        if legacy.sha(Path(name))!=digest:
            raise RuntimeError('Queued helper changed: '+name)
    for name,digest in plan['reused45_sha256'].items():
        if legacy.sha(Path(name))!=digest:
            raise RuntimeError('Completed seed45 changed: '+name)

def validate_finished(folder,seed):
    state=read(folder/'status.json')
    if state.get('phase')!='complete' or seed not in state.get('completed_seeds',[]):
        raise RuntimeError(f'Token{seed} missing completion marker')
    result=read(folder/f'seed{seed}/test.json')
    if not math.isfinite(float(result['avg_over_rates']['MAE'])) or set(result['per_rate'])!={f'{i/10:.1f}' for i in range(10)}:
        raise RuntimeError('Invalid Token test curve')
    for name in ('best_Acc7.pth','test.predictions.npz','selection.json','config.json'):
        if not (folder/f'seed{seed}'/name).is_file():
            raise RuntimeError('Missing completed Token file: '+name)
    return result

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    state=dict(phase='preflight_cpu',pid=os.getpid(),physical_gpu=1,
        authority=AUTHORITY,seeds=SEEDS,new_seeds=NEW_SEEDS,reused_seeds=[45],
        completed_seeds=[45],dependency=str(R1),created_at=stamp(),
        historical_alignment='authorized candidate; not proven historical equality',
        no_current_worker_interruption=True)
    write(OUTPUT/'status.json',state)
    lock=None
    try:
        manifest=read(BASE/'manifest.json')
        review=read(ROOT/'experiments/p0B_legacy_replay_20261008/environment_review.json')
        if review.get('state')!='user_authorized_candidate':
            raise RuntimeError('Candidate review authorization missing')
        validate_finished(OLD_TOKEN,45)
        paths=[OLD_TOKEN/'run.json',OLD_TOKEN/'status.json']+[OLD_TOKEN/'seed45'/n for n in ('config.json','selection.json','test.json','test.predictions.npz','best_Acc7.pth')]
        plan=dict(authority=AUTHORITY,created_at=stamp(),seeds=SEEDS,new_seeds=NEW_SEEDS,
            reuse45=str(OLD_TOKEN/'seed45'),frozen_manifest=manifest,
            helpers={str(p):legacy.sha(p) for p in (Path(__file__),ENSEMBLE,CONFIG_CHECK,Path(legacy.__file__))},
            reused45_sha256={str(p):legacy.sha(p) for p in paths},
            commands={str(s):[legacy.PYTHON,'-u',str(BASE/'bootstrap.py'),'train_token_input.py']+native_flags(s,OUTPUT/f'native_seed{s}') for s in NEW_SEEDS},
            ensemble=dict(seeds=SEEDS,weights=[1/6]*6,method='fixed equal prediction average after validation selection; never test-selected subset'),
            dependency_pid_snapshot={k:read(R1/'status.json').get(k) for k in ('pid','worker_pid')},
            gpu_lock=str(ROOT/'results/research2_gpu1.lock'))
        write(OUTPUT/'plan.json',plan)
        verify_pin(plan)
        legacy.OUTPUT=OUTPUT
        env=legacy.environment(manifest)  # CPU-only dependency probe, CUDA uninitialized.
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
        subprocess.run([legacy.PYTHON,str(CONFIG_CHECK),'--output',str(OUTPUT/'configuration_check.json')],env=cpu_env,cwd=ROOT,check=True)
        state['phase']='waiting_research1'
        write(OUTPUT/'status.json',state)
        print('QUEUED: GPU1 after R1; new42/43/44/46/47, completed45 reused',flush=True)
        while not dependency_ready(read(R1/'status.json')):
            time.sleep(15)
        state['phase']='waiting_gpu1_lock'
        write(OUTPUT/'status.json',state)
        while lock is None:
            try:lock=acquire_gpu1(manifest)
            except BlockingIOError:time.sleep(15)
        verify_pin(plan)
        legacy.verify_inputs(manifest)
        env=legacy.environment(manifest)
        (OUTPUT/'seed45').symlink_to(OLD_TOKEN/'seed45',target_is_directory=True)
        for seed in NEW_SEEDS:
            verify_pin(plan)
            folder=OUTPUT/f'native_seed{seed}'
            command=plan['commands'][str(seed)]
            state.update(phase='training',current_seed=seed,command=command,started_training_at=stamp())
            write(OUTPUT/'status.json',state)
            print('START Token',seed,stamp(),flush=True)
            with (OUTPUT/f'seed{seed}.console.log').open('x') as stream:
                child=subprocess.Popen(command,cwd=OUTPUT,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                state['worker_pid']=child.pid
                write(OUTPUT/'status.json',state)
                code=child.wait()
            state.pop('worker_pid',None)
            if code:raise RuntimeError(f'Token{seed} exited {code}; no auto retry or parameter fallback')
            validate_finished(folder,seed)
            (OUTPUT/f'seed{seed}').symlink_to(folder/f'seed{seed}',target_is_directory=True)
            state['completed_seeds'].append(seed)
            write(OUTPUT/'status.json',state)
        verify_pin(plan)
        state.update(phase='ensemble_cpu',current_seed=None)
        write(OUTPUT/'status.json',state)
        subprocess.run([legacy.PYTHON,str(ENSEMBLE),'--output',str(OUTPUT)],env=dict(env,CUDA_VISIBLE_DEVICES=''),cwd=OUTPUT,check=True)
        report=read(OUTPUT/'ensemble.json')
        if report['seeds']!=SEEDS or not math.isfinite(report['avg_over_rates']['MAE']):
            raise RuntimeError('Invalid fixed six-seed ensemble')
        state.update(phase='complete',finished_at=stamp())
        write(OUTPUT/'status.json',state)
        print('COMPLETE',json.dumps(report['avg_over_rates']),flush=True)
    except BaseException:
        state.update(phase='failed',traceback=traceback.format_exc(),failed_at=stamp())
        write(OUTPUT/'status.json',state)
        raise
    finally:
        if lock is not None:lock.close()

if __name__=='__main__':main()
