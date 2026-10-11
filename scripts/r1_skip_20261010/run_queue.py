"""Explicit user skip policy; resume unstarted jobs, preserve every old attempt."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
# Avoid colliding with this file's run_queue name when importing pinned helpers.
import importlib.util
sys.path.insert(0,str(ROOT/'scripts/r1_aligned_20261009'))
from plan import BASE,OUTPUT,PYTHON,REPORT_KEY,read,write,sha
spec=importlib.util.spec_from_file_location('original_queue',ROOT/'scripts/r1_aligned_20261009/run_queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
HELPERS=Path(__file__).parent
META=OUTPUT/'skip_20261010_01'
AUTHORITY='2026-10-10 human: 遇到NAN后直接跳过。直接跑seed2024就行，我们就用seed2024进行对比。'


def validate(folder,job):
    report=read(folder/f'results/MOSI_p0B_seed{job["seed"]}_test.json')
    for key,metric in [('primary_MAE','MAE'),('secondary_Acc2','Acc2'),('tertiary_Acc7','Acc7'),('quaternary_F1_has0','F1_has0')]:
        if not q.finite_metrics(report[key]) or not (folder/f'checkpoints/MOSI_p0B_seed{job["seed"]}_best_{metric}.pt').exists():
            raise RuntimeError('Invalid native test/checkpoint')
    if read(folder/'status.json')['phase']!='native_finished':raise RuntimeError('Native training incomplete')
    if read(folder/'effective_config.json')['job']!=job or not read(folder/'first_step_gradients.json'):
        raise RuntimeError('Config/intervention evidence missing')
    progress=read(folder/'numerical_progress.json')
    if progress['updates_checked']<=0:raise RuntimeError('No finite optimizer update')
    skips=read(folder/'skipped_batches.json') if (folder/'skipped_batches.json').exists() else {'count':0,'events':[]}
    if skips['count']!=progress['skipped_batches']:raise RuntimeError('Skip count mismatch')
    if (folder/'numerical_failure.json').exists():raise RuntimeError('Post-update corruption cannot be certified')
    return report,skips


def main():
    old=read(OUTPUT/'status.json');plan=read(OUTPUT/'plan.json')
    selected_jobs=[j for j in plan['jobs'] if j['seed']==2024]
    if len(selected_jobs)!=11:raise RuntimeError('Expected eleven seed2024 settings')
    expected=[j['job_id'] for j in plan['jobs'][:5]]
    if old['phase']!='failed' or old['completed']!=expected or old['current_job']!='R1-05_seed2024':
        raise RuntimeError('Unexpected resume boundary')
    if q.alive(old.get('pid')) or q.alive(old.get('active_worker_pid')):
        raise RuntimeError('Existing controller/worker alive')
    META.mkdir(exist_ok=False);write(META/'previous_status.json',old)
    state=dict(old,phase='skip_preflight',pid=os.getpid(),authority=AUTHORITY,
               recovery_directory=str(META),skip_policy=True,protocol_changed=True,
               total_jobs=11,selected_seeds=[2024],original_plan_jobs=33)
    state.pop('traceback',None);state.pop('failed_at',None)
    write(META/'status.json',state);lock=None
    try:
        q.verify_pin(plan)
        manifest=plan['frozen_native_manifest']
        if read(BASE/'environment_review.json')['state']!='user_authorized_candidate':
            raise RuntimeError('Candidate authorization missing')
        env=q.environment(manifest)
        subprocess.run([PYTHON,str(HELPERS/'tests.py')],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
        initial_by_seed={}
        preserved=[]
        for job in plan['jobs'][:5]:
            folder=OUTPUT/'train'/job['job_id'];test=read(folder/'test.json')
            if read(folder/'status.json')['phase']!='complete' or test['job']!=job or not q.finite_metrics(test['summary']):
                raise RuntimeError('Previously completed job invalid')
            initial=read(folder/'intervention.json')['initial_state_sha256']
            if initial_by_seed.setdefault(job['seed'],initial)!=initial:raise RuntimeError('Initialization mismatch')
            preserved.append(dict(job=job['job_id'],test_sha256=sha(folder/'test.json')))
        q.legacy.verify_inputs(manifest)
        if shutil.disk_usage(OUTPUT).free<(len(selected_jobs[5:])*5.4+18)*1024**3:
            raise RuntimeError('Insufficient space to preserve old snapshots and all remaining checkpoints')
        pins={str(p):sha(p) for p in sorted(HELPERS.glob('*.py'))}
        pins.update({str(p):sha(p) for p in [ROOT/'scripts/r1_repair_20261010/numerical_guard.py',ROOT/'scripts/r1_aligned_20261009/bootstrap.py']})
        state['phase']='waiting_gpu1_lock';write(META/'status.json',state)
        while lock is None:
            try:lock=q.acquire_gpu1(manifest)
            except BlockingIOError:time.sleep(15)
        q.verify_pin(plan)
        for name,digest in pins.items():
            if sha(name)!=digest:raise RuntimeError('Skip helper changed')
        write(META/'plan.json',dict(authority=AUTHORITY,helpers=pins,preserved=preserved,
            original_plan_sha256=sha(OUTPUT/'plan.json'),remaining_jobs=selected_jobs[5:],
            seeds=[2024],total_jobs=11,unscheduled_seeds=[42,1111],
            from_scratch=True,checkpoint_resume=False,
            changed='nonfinite batch skip before update; clear grads, do not update LR/global_step',
            boundary='First5 completed without skipped batches; later jobs authorized skip policy. Nonzero skips alter optimization trajectory; NOT a strictly unchanged original training protocol',
            reporting='Prespecified validation Acc2 checkpoint entire tuple; record every skip count, do not test-select'))
        archive=META/'failed_attempts';archive.mkdir()
        failed=OUTPUT/'train'/old['current_job']
        evidence={n:sha(failed/n) for n in ('console.log','job.json','numerical_failure.json')}
        failed.rename(archive/old['current_job'])
        write(META/'preservation.json',dict(archived=str(archive/old['current_job']),sha256=evidence,
            snapshot_preserved=True,completed5_unchanged=True))
        for job in selected_jobs[5:]:
            q.verify_pin(plan)
            for name,digest in pins.items():
                if sha(name)!=digest:raise RuntimeError('Skip helper changed while running')
            if shutil.disk_usage(OUTPUT).free<18*1024**3:raise RuntimeError('Disk reserve breached')
            folder=OUTPUT/'train'/job['job_id'];folder.mkdir(exist_ok=False);write(folder/'job.json',job)
            command=[PYTHON,'-u',str(HELPERS/'bootstrap.py'),'--job-file',str(folder/'job.json'),'--output',str(folder)]
            state.update(phase='train',current_job=job['job_id'],current_output=str(folder),command=command)
            write(OUTPUT/'status.json',state);write(META/'status.json',state)
            with (folder/'console.log').open('x') as log:
                child=subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                state['active_worker_pid']=child.pid;write(OUTPUT/'status.json',state);write(META/'status.json',state)
                print('START_SKIP_POLICY',job['job_id'],child.pid,q.stamp(),flush=True)
                code=child.wait()
            state.pop('active_worker_pid',None)
            if code:raise RuntimeError(f'{job["job_id"]} failed{code}; no automatic job retry')
            report,skips=validate(folder,job)
            initial=read(folder/'intervention.json')['initial_state_sha256']
            if initial_by_seed.setdefault(job['seed'],initial)!=initial:raise RuntimeError('Initialization mismatch')
            write(folder/'test.json',dict(summary=report[REPORT_KEY],native_results=report,report_key=REPORT_KEY,
                job=job,skip_policy=True,skipped_batches=skips['count'],protocol_changed=skips['count']>0,
                initialization_matched_same_seed_full=True,skip_plan=str(META/'plan.json')))
            write(folder/'status.json',dict(phase='complete',job=job['job_id'],finished_at=q.stamp(),skipped_batches=skips['count']))
            state['completed'].append(job['job_id']);write(OUTPUT/'status.json',state);write(META/'status.json',state)
            q.aggregate(plan)
            aggregate=read(OUTPUT/'aggregate.json')
            aggregate['skip_protocol_boundary']=dict(authority=AUTHORITY,changed_jobs_start='R1-05_seed2024',
                note='Metric means are descriptive; skips alter successful-update counts. Do not hide protocol difference or call all effects strictly matched original training.',
                completed_skip_counts={j:read(OUTPUT/'train'/j/'test.json').get('skipped_batches',0) for j in state['completed']})
            write(OUTPUT/'aggregate.json',aggregate)
            print('COMPLETE',job['job_id'],'skips',skips['count'],report[REPORT_KEY],flush=True)
        state.update(phase='complete',finished_at=q.stamp(),current_job=None)
        write(OUTPUT/'status.json',state);write(META/'status.json',state)
    except BaseException:
        state.update(phase='failed',traceback=traceback.format_exc(),failed_at=q.stamp())
        write(META/'status.json',state)
        if (META/'preservation.json').exists():write(OUTPUT/'status.json',state)
        raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
