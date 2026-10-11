"""Explicitly authorized recovery of stopped GPU1 native R1 ablations.

Preserve old completed results and failed attempt, verify immutable pins, use
the same native bootstrap with a fail-fast safety wrapper. No silent fallback.
"""
import os
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/r1_aligned_20261009'))
import run_queue as q
from plan import BASE, OUTPUT, PYTHON, REPORT_KEY, read, write, sha

REPAIR = None
HELPERS = Path(__file__).parent
AUTHORITY = '2026-10-10 human: 进行定位修复并且恢复GPU1消融队列。'


def validate_completed(folder,job):
    state = read(folder/'status.json')
    if state.get('phase') != 'complete':
        raise RuntimeError('Completed job marker missing: '+job['job_id'])
    test = read(folder/'test.json')
    native = read(folder/f'results/MOSI_p0B_seed{job["seed"]}_test.json')
    if test['job'] != job or test['summary'] != native[REPORT_KEY] or not q.finite_metrics(test['summary']):
        raise RuntimeError('Completed result/config mismatch')
    if 'non-finite loss' in (folder/'console.log').read_text():
        raise RuntimeError('Nonfinite batch in completed job')
    for metric in ('MAE','Acc2','Acc7','F1_has0'):
        if not (folder/f'checkpoints/MOSI_p0B_seed{job["seed"]}_best_{metric}.pt').is_file():
            raise RuntimeError('Missing completed checkpoint')
    return read(folder/'intervention.json')['initial_state_sha256']


def main():
    global REPAIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovery-name', default='repair_20261010_02')
    cli = parser.parse_args()
    if not cli.recovery_name.startswith('repair_20261010') or Path(cli.recovery_name).name != cli.recovery_name:
        raise RuntimeError('Invalid scoped recovery directory')
    REPAIR = OUTPUT/cli.recovery_name
    old = read(OUTPUT/'status.json')
    plan = read(OUTPUT/'plan.json')
    if old['phase'] != 'failed' or q.alive(old.get('pid')) or q.alive(old.get('active_worker_pid')):
        raise RuntimeError('Not an exited failed queue; refuse duplicate start')
    if old['current_job'] != 'R1-03_seed2024' or old['completed'] != [j['job_id'] for j in plan['jobs'][:3]]:
        raise RuntimeError('Unexpected failure boundary; review required')
    REPAIR.mkdir(exist_ok=False)
    write(REPAIR/'previous_queue_status.json',old)
    state = dict(old,phase='repair_preflight_cpu',pid=os.getpid(),authority=AUTHORITY,
                 recovery_directory=str(REPAIR),recovery_started_at=q.stamp())
    state.pop('traceback',None); state.pop('failed_at',None)
    write(REPAIR/'status.json',state)
    lock = None
    try:
        q.verify_pin(plan)
        manifest = plan['frozen_native_manifest']
        if read(BASE/'environment_review.json')['state'] != 'user_authorized_candidate':
            raise RuntimeError('Candidate authorization missing')
        env = q.environment(manifest)
        subprocess.run([PYTHON,str(HELPERS/'tests.py')],cwd=ROOT,
                       env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
        subprocess.run([PYTHON,str(ROOT/'scripts/r1_aligned_20261009/tests.py')],cwd=ROOT,
                       env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
        initial_by_seed = {}
        preserved = []
        for job in plan['jobs'][:3]:
            folder = OUTPUT/'train'/job['job_id']
            initial = validate_completed(folder,job)
            if initial_by_seed.setdefault(job['seed'],initial) != initial:
                raise RuntimeError('Completed initialization mismatch')
            preserved.append(dict(job=job['job_id'],test_sha256=sha(folder/'test.json'),
                                  native_test_sha256=sha(folder/f'results/MOSI_p0B_seed{job["seed"]}_test.json')))
        if not q.dependency_ready(read(q.TOKEN/'status.json')):
            raise RuntimeError('Token prerequisite not complete/exited')
        q.legacy.verify_inputs(manifest)
        # Existing queue still owes30 checkpoint sets; reserve ~25GiB extra
        # for failure capture and temporary checkpoint files, never delete.
        if shutil.disk_usage(OUTPUT).free < (30*5.4+25)*1024**3:
            raise RuntimeError('Insufficient space for preserved results and remaining native checkpoints')
        # CPU-only waiter; never kill a lock owner, delete a lock, or allocate
        # CUDA while a concurrent user task occupies GPU1. Freeze our helpers
        # before waiting and reverify every byte after acquiring the device.
        helper_pins = {str(p):sha(p) for p in sorted(HELPERS.glob('*.py'))}
        state.update(phase='waiting_gpu1_lock',cpu_tests='17 passed',
                     waiting_since=q.stamp(),policy='Do not interrupt concurrent Token optimization')
        write(REPAIR/'status.json',state)
        print('WAIT_GPU1_LOCK',q.stamp(),flush=True)
        while lock is None:
            try:
                lock = q.acquire_gpu1(manifest)
            except BlockingIOError:
                time.sleep(15)
        q.verify_pin(plan)
        for name,digest in helper_pins.items():
            if sha(name) != digest:
                raise RuntimeError('Repair helper changed while waiting')
        q.legacy.verify_inputs(manifest)
        env = q.environment(manifest)
        state.update(phase='cuda_finite_path_verification')
        write(REPAIR/'status.json',state)
        subprocess.run([PYTHON,str(HELPERS/'cuda_parity.py'),str(REPAIR)],
                       cwd=BASE,env=env,check=True,pass_fds=(lock.fileno(),))
        repair_plan = dict(authority=AUTHORITY,created_at=q.stamp(),source_plan_sha256=sha(OUTPUT/'plan.json'),
                           immutable_helpers=plan['helpers'],preserved=preserved,
                           repair_helpers=helper_pins,
                           safety_change='Fail-before-nonfinite update and capture; finite native path unchanged',
                           originating_fault='unresolved until failure captured/replayed',
                           from_scratch=True,old_checkpoint_resume=False,physical_gpu=1,
                           historical_equality='user-authorized candidate, NOT proved exact dated environment',
                           remaining_jobs=plan['jobs'][3:],automatic_retry=False,
                           comparison_boundary='First3 original native; later native with extra fail-fast checks. No numeric fallback allowed; CPU bitwise equality verified. CUDA kernel nondeterminism not excluded.')
        write(REPAIR/'plan.json',repair_plan)
        failed = OUTPUT/'train'/old['current_job']
        (REPAIR/'failed_attempts').mkdir()
        before = {name:sha(failed/name) for name in ('job.json','console.log','status.json','intervention.json','effective_config.json')}
        archived = REPAIR/'failed_attempts'/old['current_job']
        failed.rename(archived)
        write(REPAIR/'preservation.json',dict(source=str(failed),archived=str(archived),sha256=before,
                                            recoverable=True,original_completed_results_unchanged=True))
        write(OUTPUT/'status.json',state)
        write(REPAIR/'status.json',state)
        for job in plan['jobs'][3:]:
            q.verify_pin(plan)
            for name,value in repair_plan['repair_helpers'].items():
                if sha(name) != value:
                    raise RuntimeError('Repair helper changed during queue')
            if shutil.disk_usage(OUTPUT).free < 18*1024**3:
                raise RuntimeError('Disk reserve breached')
            folder = OUTPUT/'train'/job['job_id']; folder.mkdir(exist_ok=False)
            write(folder/'job.json',job)
            command = [PYTHON,'-u',str(HELPERS/'bootstrap_guarded.py'),
                       '--job-file',str(folder/'job.json'),'--output',str(folder)]
            state.update(phase='train',current_job=job['job_id'],current_output=str(folder),command=command)
            write(OUTPUT/'status.json',state)
            print('START_GUARDED',job['job_id'],q.stamp(),flush=True)
            with (folder/'console.log').open('x') as log:
                child = subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                state['active_worker_pid'] = child.pid; write(OUTPUT/'status.json',state)
                write(REPAIR/'status.json',state)
                code = child.wait()
            state.pop('active_worker_pid',None)
            if code:
                raise RuntimeError(f'{job["job_id"]} exited{code}; failure capture if present must be analyzed; no automatic retry')
            report = q.validate_job(folder,job)
            if (folder/'numerical_failure.json').exists():
                raise RuntimeError('Captured numerical failure cannot be certified complete')
            progress = read(folder/'numerical_progress.json')
            if progress['updates_checked'] <= 0:
                raise RuntimeError('Safety observer did not check updates')
            initial = read(folder/'intervention.json')['initial_state_sha256']
            if initial_by_seed.setdefault(job['seed'],initial) != initial:
                raise RuntimeError('Common initialization mismatch')
            write(folder/'test.json',dict(summary=report[REPORT_KEY],native_results=report,
                 report_key=REPORT_KEY,job=job,native_result_path=str(folder/f'results/MOSI_p0B_seed{job["seed"]}_test.json'),
                 initialization_matched_same_seed_full=True,safety_guard=True,repair_plan=str(REPAIR/'plan.json')))
            write(folder/'status.json',dict(phase='complete',job=job['job_id'],finished_at=q.stamp(),exit_code=0))
            state['completed'].append(job['job_id']);write(OUTPUT/'status.json',state)
            write(REPAIR/'status.json',state)
            q.aggregate(plan)
            print('COMPLETE',job['job_id'],report[REPORT_KEY],flush=True)
        state.update(phase='complete',current_job=None,finished_at=q.stamp())
        write(OUTPUT/'status.json',state)
        write(REPAIR/'status.json',state)
    except BaseException:
        state.update(phase='failed',traceback=traceback.format_exc(),failed_at=q.stamp())
        write(REPAIR/'recovery_failure.json',state)
        write(REPAIR/'status.json',state)
        # Preflight failure must not impersonate a recovered/started queue.
        if (REPAIR/'preservation.json').exists():
            write(OUTPUT/'status.json',state)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
