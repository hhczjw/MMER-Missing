"""Move the remaining fixed queue at a completed-job boundary; no epoch resume."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research2_token_optimization_20261010'
ORIGINAL = Path(__file__).with_name('optimize_token_20261010.py')
MIGRATION = OUT / 'migration_gpu0.json'


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def process_identity(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p/'stat').read_text().rsplit(') ',1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(start=stat[19], command=(p/'cmdline').read_bytes().decode().split('\0'))
    except FileNotFoundError:
        return None


def gpu0_lock():
    handle = Path('/tmp/mmer_ablation_physical_gpu0.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    if uuid in apps:
        handle.close()
        raise RuntimeError('GPU0 busy; original queue not interrupted')
    return handle


def adapted_source(source):
    replacements = {
        "CUDA_VISIBLE_DEVICES='1'": "CUDA_VISIBLE_DEVICES='0'",
        "    if state_path.exists():\n        raise FileExistsError(state_path)":
            "    assert not smoke\n    previous = rt.read(state_path)\n    assert previous['phase'] == 'migration_ready_gpu0'",
        "gpu=1, completed=[]": "gpu=0, completed=previous['completed'].copy()",
        "        summaries = []": "        summaries = rt.read(OUT / 'validation_summary.json')",
        "                state.update(current=job)":
            "                if job in state['completed']:\n                    continue\n                state.update(current=job, epoch=0, best_epoch=None, best_validation=None)",
    }
    for old,new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Unexpected frozen runner structure: '+old)
        source = source.replace(old,new)
    compile(source,str(ORIGINAL),'exec')
    return source


def run():
    request = read(MIGRATION)
    old_pid = request['old_pid']
    expected = request['old_identity']
    target = request['boundary_job']
    source = ORIGINAL.read_text()
    assert sha(ORIGINAL) == request['source_sha256']
    adapted = adapted_source(source)
    lock = None
    try:
        # Reserve the idle destination while waiting for the existing job to finish.
        lock = gpu0_lock()
        request.update(controller_pid=os.getpid(), phase='waiting_job_boundary', gpu0_reserved=True)
        write(MIGRATION,request)
        while True:
            state = read(OUT/'status.json')
            if state['phase'] == 'failed':
                raise RuntimeError('Original queue failed; no automatic restart')
            completed = state.get('completed',[])
            summaries = read(OUT/'validation_summary.json')
            if target in completed and target in [s['job'] for s in summaries]:
                break
            if process_identity(old_pid) != expected:
                raise RuntimeError('Original worker changed/exited before successful boundary')
            time.sleep(.25)
        # Identity-bound SIGTERM only after the requested job and archive succeeded.
        if process_identity(old_pid) == expected:
            assert str(ORIGINAL) in expected['command']
            os.kill(old_pid,signal.SIGTERM)
            deadline = time.monotonic()+30
            while process_identity(old_pid) == expected and time.monotonic()<deadline:
                time.sleep(.2)
            if process_identity(old_pid) == expected:
                raise RuntimeError('Worker did not exit; no force kill or duplicate launch')
        elif process_identity(old_pid) is not None:
            raise RuntimeError('PID identity changed; refusing to signal')
        state = read(OUT/'status.json')
        summaries = read(OUT/'validation_summary.json')
        completed = [s['job'] for s in summaries]
        archive = OUT/'migration_gpu0_records'
        archive.mkdir(exist_ok=False)
        write(archive/'old_status.json',state)
        write(archive/'old_plan.json',read(OUT/'plan.json'))
        namespace = dict(__name__='migration_adapter',__file__=str(ORIGINAL))
        exec(compile(adapted,str(ORIGINAL),'exec'),namespace)
        expected_jobs = [c['id']+f'_seed{s}' for s in namespace['SEEDS'] for c in namespace['CANDIDATES']]
        assert len(set(completed))==len(completed) and set(completed)<=set(expected_jobs)
        for entry in summaries:
            job = entry['job']
            h = read(OUT/job/'history.json')
            assert len(h)==60 and h[-1]['epoch']==60
            assert sha(OUT/job/'best_Acc7.pth') == sha(Path(entry['checkpoint']))
        retained = []
        for job in expected_jobs:
            folder = OUT/job
            if job not in completed and folder.exists():
                target_folder = archive / ('interrupted_initialization_'+job)
                folder.rename(target_folder)
                retained.append(str(target_folder))
        write(OUT/'status.json',dict(phase='migration_ready_gpu0',completed=completed,old_pid=old_pid,gpu=0))
        request.update(phase='running_on_gpu0', completed_preserved=completed,
                       remaining=[j for j in expected_jobs if j not in completed], retained_partial=retained,
                       effective_runner_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                       original_stopped=True, destination_pid=os.getpid())
        write(MIGRATION,request)
        namespace['gpu_lock'] = lambda: lock
        namespace['run'](False)
        request.update(phase='queue_complete')
        write(MIGRATION,request)
    except BaseException:
        request.update(phase='failed',error=traceback.format_exc())
        write(MIGRATION,request)
        raise
    finally:
        if lock is not None:
            lock.close()


def launch():
    if MIGRATION.exists():
        raise FileExistsError(MIGRATION)
    state = read(OUT/'status.json')
    assert state['phase']=='training' and state['gpu']==1
    ident = process_identity(state['pid'])
    assert ident and str(ORIGINAL) in ident['command']
    adapted_source(ORIGINAL.read_text())
    write(MIGRATION,dict(phase='requested',authority='User requested remaining optimization queue move to GPU0',
                        old_pid=state['pid'],old_identity=ident,boundary_job=state['current'],
                        source_sha256=sha(ORIGINAL),original_completed=state['completed'],
                        policy='finish current job; preserve completed; remaining jobs start fresh on GPU0'))
    with (OUT/'gpu0_migration.console.log').open('x') as log:
        p = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run'],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Migration controller PID',p.pid)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',action='store_true')
    args = ap.parse_args()
    run() if args.run else launch()
