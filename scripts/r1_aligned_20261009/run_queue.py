"""Authorized fresh R1 ablations on physical GPU1 after the current Token queue.

This launcher never resumes old ablations/checkpoints or interrupts GPU workers.
Every job runs the original native training/test entry, with model-only changes.
"""
from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).parent))
from plan import ROOT, BASE, OUTPUT, TOKEN, PYTHON, REPORT_KEY, AUTHORITY, jobs, read, write, sha
sys.path.insert(0,str(ROOT/'scripts/reproduction_20261008'))
import launch_p0b as legacy


def stamp():
    return datetime.now().astimezone().isoformat()


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ',1)[1][0]!='Z'
    except (OSError,ValueError,TypeError):
        return False


def finite_metrics(value):
    return bool(value) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value.values())


def dependency_ready(state):
    if state.get('phase')=='failed':
        raise RuntimeError('Token queue failed; no automatic R1 start or retry')
    pids=[state.get(k) for k in ('pid','worker_pid') if state.get(k)]
    if state.get('phase')!='complete':
        if not pids or not any(alive(p) for p in pids):
            raise RuntimeError('Token queue lost without completion')
        return False
    if any(alive(p) for p in pids):
        return False
    if set(state.get('completed_seeds',[]))!={42,43,44,45,46,47}:
        raise RuntimeError('Incomplete Token seeds at completion')
    for seed in (42,43,44,45,46,47):
        if not finite_metrics(read(TOKEN/f'seed{seed}/test.json')['avg_over_rates']):
            raise RuntimeError('Invalid completed Token test')
    report=read(TOKEN/'ensemble.json')
    if report['seeds']!=[42,43,44,45,46,47] or not finite_metrics(report['avg_over_rates']):
        raise RuntimeError('Invalid fixed Token ensemble')
    return True


def acquire_gpu1(manifest):
    lock=(ROOT/'results/research2_gpu1.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True,timeout=10)
        gpu=next(s for s in rows.splitlines() if s.split(',')[0].strip()=='1')
        if manifest['current_gpu1_uuid'] not in gpu or manifest['expected_current_environment']['driver'] not in gpu:
            raise RuntimeError('GPU1 identity/driver changed')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
        if any(manifest['current_gpu1_uuid'] in row for row in apps.splitlines()):
            raise BlockingIOError('GPU1 has compute processes')
        write(OUTPUT/'gpu_at_acquire.json',dict(inventory=rows,compute_processes=apps,time=stamp()))
        return lock
    except BaseException:
        lock.close()
        raise


def environment(manifest):
    import json
    env=os.environ.copy()
    for key in manifest['expected_unset_environment']:
        if env.get(key) is not None:
            raise RuntimeError('Unreviewed inherited override: '+key)
    env.update(CUDA_VISIBLE_DEVICES='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=str(BASE/'source/mamba')+':'+str(BASE/'source'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    probe="import importlib.metadata as m,json,torch,sys;print(json.dumps(dict(python=sys.version.split()[0],packages={k:m.version(k) for k in ['torch','triton','transformers','numpy','tokenizers','causal-conv1d']},cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_initialized=torch.cuda.is_initialized())))"
    actual=json.loads(subprocess.check_output([PYTHON,'-c',probe],env=dict(env,CUDA_VISIBLE_DEVICES=''),text=True))
    expected=manifest['expected_current_environment']
    if actual['cuda_initialized'] or any(actual[k]!=expected[k] for k in ('python','packages','cuda','cudnn')):
        raise RuntimeError('Authorized candidate environment changed')
    write(OUTPUT/'environment_at_queue.json',dict(cpu_probe=actual,
        settings={k:env.get(k) for k in manifest['capture_environment_keys']},
        historical_equality='not proven; user-authorized candidate'))
    return env


def verify_pin(plan):
    legacy.verify_sources(plan['frozen_native_manifest'])
    for name,digest in plan['helpers'].items():
        if sha(name)!=digest:
            raise RuntimeError('Queued helper changed: '+name)
    if sha(TOKEN/'plan.json')!=plan['token_dependency_plan_sha256']:
        raise RuntimeError('Token dependency plan changed')


def validate_job(folder,job):
    report=read(folder/f'results/MOSI_p0B_seed{job["seed"]}_test.json')
    for key,metric in [('primary_MAE','MAE'),('secondary_Acc2','Acc2'),('tertiary_Acc7','Acc7'),('quaternary_F1_has0','F1_has0')]:
        if not finite_metrics(report.get(key,{})):
            raise RuntimeError('Invalid native test: '+key)
        if not (folder/f'checkpoints/MOSI_p0B_seed{job["seed"]}_best_{metric}.pt').is_file():
            raise RuntimeError('Missing native checkpoint '+metric)
    if read(folder/'status.json').get('phase')!='native_finished':
        raise RuntimeError('Native entry has not finished')
    if not read(folder/'first_step_gradients.json'):
        raise RuntimeError('Gradient intervention evidence missing')
    config=read(folder/'effective_config.json')
    if config['job']!=job:
        raise RuntimeError('Actual job/config mismatch')
    console=(folder/'console.log').read_text()
    if 'non-finite loss' in console or 'nan_skipped=' in console:
        raise RuntimeError('Native training skipped nonfinite batches; report and stop without protocol fallback')
    return report


def aggregate(plan):
    rows=[]
    for index,title,changes in __import__('plan').SETTINGS:
        members=[]
        for job in plan['jobs']:
            if job['id']=='R1-'+index and job['job_id'] in read(OUTPUT/'status.json')['completed']:
                members.append(dict(seed=job['seed'],metrics=read(OUTPUT/'train'/job['job_id']/'test.json')['summary']))
        if not members:
            continue
        keys=members[0]['metrics']
        mean={k:statistics.mean(m['metrics'][k] for m in members) for k in keys}
        sd={k:statistics.stdev(m['metrics'][k] for m in members) if len(members)>1 else None for k in keys}
        paired=[]
        for m in members:
            base=OUTPUT/'train'/f'R1-00_seed{m["seed"]}'/'test.json'
            if base.exists():
                ref=read(base)['summary']
                paired.append(dict(seed=m['seed'],difference={k:m['metrics'][k]-ref[k] for k in keys}))
        rows.append(dict(id='R1-'+index,title=title,changes=changes,n=len(members),members=members,mean=mean,
            sample_std=sd,paired_vs_same_seed_full=paired,
            paired_mean={k:statistics.mean(p['difference'][k] for p in paired) for k in keys} if paired else None))
    write(OUTPUT/'aggregate.json',dict(report_key=REPORT_KEY,
        meaning='metric means, NOT a prediction ensemble; each member selected by validation Acc2 Non0; MAE controls early stop',
        historical_scores_not_used_as_paired_baseline=True,rows=rows))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    state=dict(phase='preflight_cpu',pid=os.getpid(),physical_gpu=1,authority=AUTHORITY,
        completed=[],total_jobs=len(jobs()),dependency=str(TOKEN),created_at=stamp(),
        historical_alignment='authorized candidate; exact dated historical source/environment remains unknown')
    write(OUTPUT/'status.json',state);lock=None
    try:
        manifest=read(BASE/'recovery_manifest.json')
        review=read(BASE/'environment_review.json')
        if review['state']!='user_authorized_candidate' or not review.get('evidence_or_user_direction'):
            raise RuntimeError('Candidate review authorization missing')
        paths=sorted(Path(__file__).parent.glob('*.py'))+[Path(legacy.__file__)]
        plan=dict(authority=AUTHORITY,jobs=jobs(),report_key=REPORT_KEY,
            seeds=[2024,42,1111],native_config=read(BASE/'source/configs/config.json'),
            frozen_native_manifest=manifest,helpers={str(p):sha(p) for p in paths},
            token_dependency_plan_sha256=sha(TOKEN/'plan.json'),
            checkpoint_selection='All four native validation checkpoints retained; prespecified secondary_Acc2 whole result used for every ablation; MAE early stop',
            from_scratch=True,only_original_pretrained_text=True,old_mm_checkpoint_resume=False,
            source_strategy='Immutable native snapshot reused by SHA256; model-only post-initialization interventions; no custom training loop/text pretokenization/checkpointing/gradient repair',
            storage='Preserve all four native checkpoints; reserve space, no deletion/automatic fallback',
            gpu_lock=str(ROOT/'results/research2_gpu1.lock'))
        write(OUTPUT/'plan.json',plan)
        write(OUTPUT/'review_at_queue.json',review)
        write(OUTPUT/'alignment.json',dict(
            target='Historical p0B effective configuration and approved native replay candidate, not claimed exact dated source/environment',
            original_candidate_source_sha256={k:v['sha256'] for k,v in manifest['files'].items()},
            old_queue='results/mosi_ablations_p0b_0919_gradient_repair_20261007 (preserved paused)',
            aligned=dict(train_entry='unchanged frozen native train.py',trainer='unchanged native Trainer methods plus passive receipts',
                data='native MMDataLoader and decode/re-encode each forward; same verified input cache',
                context='native np.roll context, except prespecified context-removal job',
                adamw='two parameter groups; lr1e-5/4e-4, weight_decay1e-4 including biases/norms; native foreach=None',
                scheduler='native warmup5778/total38520,120epochs,ratio0.15',
                loss='native MultiTaskLoss task_weights M:1, SmoothL1 beta0.5; sub_loss_lambda0.3 exists in config but no T/A/V supervision',
                precision='native FP32; no AMP, EMA, text activation checkpointing or custom gradient repair/retry',
                selection='four native validation checkpoints, MAE earlystop30; secondary_Acc2 whole test tuple for ablation reporting',
                gpu='physicalGPU1, authorized coupled_mamba_3 environment, expandable_segments:True'),
            old_differences=dict(training='custom worker loop, direct SmoothL1, foreach=False and torch.set_num_threads4',
                activation_recomputation='enabled on old RoBERTa; not native candidate',
                gradient_recovery='old safe_clip and reference_backward retries; not native candidate',
                pretokenization='old snapshot supports pretokenization but current old configure sets pretokenize_text=False; NOT an enabled difference',
                model_constructor='old backend manually constructed a subset of arguments; new wrapper delegates full unchanged native constructor',
                evaluation='old custom metrics/one selected checkpoint vs new four native reports'),
            interventions='only requested model branch/KV changes after full common initialization; context off changes native config; no new random parameter initialization',
            limitations='Historical source date, dependency/driver/GPU UUID and persistent_workers evidence remain incomplete; native candidate authorization already granted. Future component effects refer to new same-pipeline controls, not historical score gaps.'))
        verify_pin(plan)
        # 33*~5.4GiB checkpoint sets plus conservative reserve for other queues/temp.
        if shutil.disk_usage(OUTPUT).free < (185+25)*1024**3:
            raise RuntimeError('Insufficient disk for all four native checkpoints across33 jobs plus reserve')
        env=environment(manifest)
        subprocess.run([PYTHON,str(Path(__file__).with_name('tests.py'))],env=dict(env,CUDA_VISIBLE_DEVICES=''),cwd=ROOT,check=True)
        state.update(phase='waiting_token',cpu_tests='passed',source_fingerprints='passed')
        write(OUTPUT/'status.json',state)
        print('QUEUED33 native R1 ablations; GPU1 after Token42–47 complete and lock released',flush=True)
        while not dependency_ready(read(TOKEN/'status.json')):
            time.sleep(15)
        state['phase']='waiting_gpu1_lock';write(OUTPUT/'status.json',state)
        while lock is None:
            try:lock=acquire_gpu1(manifest)
            except BlockingIOError:time.sleep(15)
        verify_pin(plan);legacy.verify_inputs(manifest);env=environment(manifest)
        (OUTPUT/'train').mkdir(exist_ok=False)
        initial_by_seed={}
        for job in plan['jobs']:
            verify_pin(plan)
            if shutil.disk_usage(OUTPUT).free<18*1024**3:
                raise RuntimeError('Disk reserve breached; no checkpoint deletion/fallback')
            folder=OUTPUT/'train'/job['job_id'];folder.mkdir(exist_ok=False)
            write(folder/'job.json',job)
            command=[PYTHON,'-u',str(Path(__file__).with_name('bootstrap.py')),
                '--job-file',str(folder/'job.json'),'--output',str(folder)]
            state.update(phase='train',current_job=job['job_id'],current_output=str(folder),command=command)
            write(OUTPUT/'status.json',state)
            print('START',job['job_id'],stamp(),flush=True)
            with (folder/'console.log').open('x') as log:
                child=subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                state['active_worker_pid']=child.pid;write(OUTPUT/'status.json',state)
                code=child.wait()
            state.pop('active_worker_pid',None)
            if code:
                raise RuntimeError(f'{job["job_id"]} exited{code}; inspect console.log; no retry or changed protocol')
            report=validate_job(folder,job)
            initial=read(folder/'intervention.json')['initial_state_sha256']
            previous=initial_by_seed.setdefault(job['seed'],initial)
            if previous!=initial:
                raise RuntimeError('Same-seed pre-intervention initialization mismatch')
            write(folder/'test.json',dict(summary=report[REPORT_KEY],native_results=report,
                report_key=REPORT_KEY,job=job,native_result_path=str(folder/f'results/MOSI_p0B_seed{job["seed"]}_test.json'),
                initialization_matched_same_seed_full=True))
            write(folder/'status.json',dict(phase='complete',job=job['job_id'],finished_at=stamp(),exit_code=0))
            state['completed'].append(job['job_id']);write(OUTPUT/'status.json',state)
            aggregate(plan)
            print('COMPLETE',job['job_id'],report[REPORT_KEY],flush=True)
        state.update(phase='complete',current_job=None,finished_at=stamp())
        write(OUTPUT/'status.json',state)
    except BaseException:
        state.update(phase='failed',traceback=traceback.format_exc(),failed_at=stamp())
        write(OUTPUT/'status.json',state)
        raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
