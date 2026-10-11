"""Matched Token slow-curriculum ablations, fixed seeds42/45, physical GPU0."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research2_token_slow_ablations_20261010'
PYTHON='/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python'
GPU_UUID='GPU-07abbe75-7a55-4623-70aa-a9572c83957e'
LEGACY=ROOT/'results/research2_ablations_gpu1_20261006_v3/plan.json'
sys.path.insert(0,str(ROOT/'scripts/research2_ablations'))


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def stamp():return datetime.now().astimezone().isoformat()
def cap(epoch,changes):return changes.get('fixed_token_rate_cap',min(.9,.3+.2*((epoch-1)//5)))


def verify(plan):
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Changed pinned source: '+name)
    for name,digest in plan['data_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Changed data: '+name)


def prepare():
    from plan import variants,reference_contract
    old=read(LEGACY)
    jobs=[dict(v,seed=s,job_id=v['id']+f'_seed{s}') for s in [42,45] for v in variants('R2-T')]
    assert len(jobs)==30
    contract=reference_contract('R2-T')
    contract.update(seed_order=[42,45],epochs=60,early_stopping=False,patience=None,
        curriculum='cap=min(.9,.3+.2*((epoch-1)//5)); policy14 fixed cap=.9',
        token_rate_cap_ranges=[[1,5,.3],[6,10,.5],[11,15,.7],[16,60,.9]])
    sources=dict(old['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    opt=ROOT/'scripts/research2_ablations/optimize_token_20261010.py'
    sources[str(opt.relative_to(ROOT))]=sha(opt)
    plan=dict(authority='Human confirmed slow curriculum, Token ablations seeds42/45 on GPU0; GPU1 R1 untouched',
        jobs=jobs,total_jobs=30,seeds=[42,45],physical_gpu=0,gpu_uuid=GPU_UUID,
        contracts={'R2-T':contract},source_sha256=sources,data_sha256=old['data_sha256'],
        from_scratch=True,resume_checkpoint=False,pretrained_text_only=True,
        selection='Validation Token10 mean Acc7; exact tie MAE; same whole checkpoint for all test metrics',
        optimizer='Same as slower_curriculum optimization: native two-group AdamW, lr3e-4, wd1e-4, clip.1; LoRA8/16; batch8/eval16 FP32',
        schedule='Manual 5epoch warmup, max-mode ReduceLROnPlateau patience5 min_lr1e-6; no early stopping',
        validation_mask_seed=20261006,test_mask_seed=20261005,
        comparison='Fresh matched full control00 for each seed. No old fast-course mixing, no test-based selection or seed weighting',
        historical_environment_boundary='Same approved current coupled_mamba_3 candidate; not proven exactly identical historical environment',
        failure_policy='Native train_epoch behavior unchanged; fail queue on nonfinite epoch/gradients or worker error, no silent retry/precision/batch changes')
    verify(plan)
    OUT.mkdir(exist_ok=False);(OUT/'train').mkdir();(OUT/'smoke').mkdir()
    write(OUT/'plan.json',plan)
    return plan


def worker(job_id,smoke):
    plan=read(OUT/'plan.json');verify(plan)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    sys.path.insert(0,str(ROOT/'scripts/checkpoint_nvme_20261007'))
    from data_cache import install_cache
    install_cache(ROOT/'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
    import runtime as rt
    import torch
    import numpy as np
    rt.assert_frozen(plan)
    job=next(j for j in plan['jobs'] if j['job_id']==job_id)
    folder=OUT/('smoke' if smoke else 'train')/job_id;folder.mkdir(exist_ok=False)
    state=dict(phase='initializing',pid=os.getpid(),job=job_id,physical_gpu=0,started_at=stamp())
    write(folder/'status.json',state)
    try:
        device=torch.device('cuda:0');torch.cuda.set_device(0)
        args=rt.arguments('R2-T',job['seed'],job['changes']);args.learning_rate=3e-4
        rt.set_seed(job['seed']);train,valid,eval_args=rt.loaders(args)
        model=rt.UMRMv3(args).to(device)
        initial=hashlib.sha256()
        for name,t in model.state_dict().items():
            initial.update(name.encode());initial.update(t.detach().cpu().contiguous().numpy().tobytes())
        write(folder/'initialization.json',dict(sha256=initial.hexdigest(),before_intervention=True,seed=job['seed']))
        restore,rec_calls=rt.interventions(model,job['changes'])
        opt=rt.optimizer(model,args);_,plate=rt.schedules(opt,args,'R2-T')
        base_lrs=[g['lr'] for g in opt.param_groups]
        write(folder/'config.json',dict(args=vars(args),job=job,contract=plan['contracts']['R2-T']))
        best=None;best_epoch=0;history=[];cp=folder/'best_Acc7.pth'
        for epoch in range(1,(1 if smoke else 60)+1):
            if epoch<=5:
                for group,lr in zip(opt.param_groups,base_lrs):group['lr']=lr*epoch/5
            args.token_train_rate_cap=job['changes'].get('fixed_token_rate_cap',.7) if smoke else cap(epoch,job['changes'])
            model.token_missing_rate=args.token_train_rate_cap
            batches=[next(iter(train))] if smoke else train
            loss=rt.train_epoch(model,batches,rt.TaskLoss(args.task_type,args.num_classes),opt,args,device,epoch,60)
            assert np.isfinite(loss),'Nonfinite train loss'
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradient'
            rec_calls.clear()
            val=rt.validation(model,[next(iter(valid))] if smoke else valid,args,'R2-T',device)
            value=val['avg_over_rates'];assert all(np.isfinite(v) for v in value.values())
            if rt.improve('R2-T',value,best):
                best=value.copy();best_epoch=epoch
                if not smoke:rt.checkpoint(cp,model,args,epoch,val,job)
                write(folder/'selection.json',dict(epoch=epoch,validation=val))
            if epoch>=5:plate.step(value['Acc7'])
            history.append(dict(epoch=epoch,loss=loss,rate_cap=args.token_train_rate_cap,
                validation=value,lr_next=[g['lr'] for g in opt.param_groups],token_stats=args.token_train_last_stats))
            write(folder/'history.json',history)
            state.update(phase='training',epoch=epoch,best_epoch=best_epoch,best_validation=best)
            write(folder/'status.json',state)
            print(job_id,epoch,'cap',args.token_train_rate_cap,'loss',loss,'validAcc7',value['Acc7'],'best',best_epoch,flush=True)
        if not smoke:
            state['phase']='testing';write(folder/'status.json',state)
            payload=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
            assert payload['epoch']==best_epoch
            model.load_state_dict(payload['model'],strict=True);del payload
            test=rt.MMDataLoader(eval_args,'test');assert len(test.dataset)==686
            args.token_mask_seed=20261005
            result=rt.evaluate_curve(model,test,args,device,rt.RATES,prediction_archive=folder/'test.predictions.npz')
            assert all(np.isfinite(v) for v in result['avg_over_rates'].values())
            result.update(job=job,checkpoint=str(cp),selected_epoch=best_epoch,epochs_run=60,
                selection=plan['selection'],sample_count=686,protocol='slow_curriculum_5epochs')
            write(folder/'test.json',result)
        restore();state.update(phase='complete',finished_at=stamp());write(folder/'status.json',state)
    except BaseException:
        state.update(phase='failed',traceback=traceback.format_exc());write(folder/'status.json',state);raise


def aggregate(plan):
    import numpy as np
    rows=[]
    for variant in sorted({j['id'] for j in plan['jobs']}):
        members=[]
        for seed in [42,45]:
            p=OUT/'train'/f'{variant}_seed{seed}'/'test.json'
            if p.exists():
                value=read(p)['avg_over_rates'];members.append(dict(seed=seed,metrics=value))
        if not members:continue
        mean={k:float(np.mean([m['metrics'][k] for m in members])) for k in members[0]['metrics']}
        sd={k:float(np.std([m['metrics'][k] for m in members],ddof=1)) for k in mean} if len(members)==2 else None
        pairs=[]
        for m in members:
            p=OUT/'train'/f'R2-T-00_seed{m["seed"]}'/'test.json'
            if p.exists():pairs.append(dict(seed=m['seed'],delta={k:m['metrics'][k]-read(p)['avg_over_rates'][k] for k in mean}))
        rows.append(dict(id=variant,n=len(members),members=members,mean=mean,sample_std=sd,paired_differences=pairs))
    write(OUT/'aggregate.json',dict(rows=rows,metric_means_not_prediction_ensemble=True,
        boundary='Two preregistered seeds, matched60epoch cohort. Test scores descriptive; no recipe/seed selection or significance claim'))


def queue():
    plan=prepare();state=dict(phase='preflight',pid=os.getpid(),physical_gpu=0,completed=[],total_jobs=30,started_at=stamp())
    write(OUT/'status.json',state);lock=None
    try:
        if shutil.disk_usage(OUT).free<75*1024**3:raise RuntimeError('Need75GiB free including R1 reserve')
        lock=(ROOT/'results/research2_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actual=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()
        assert actual.split(',')[0].strip()==GPU_UUID
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if GPU_UUID in apps:raise RuntimeError('GPU0 occupied; do not interrupt')
        write(OUT/'environment.json',dict(python=sys.executable,python_version=sys.version,gpu=actual,compute_processes_before=apps))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
        def launch(job,smoke=False):
            verify(plan)
            state.update(phase='smoke' if smoke else 'training',current_job=job['job_id'])
            write(OUT/'status.json',state)
            log=OUT/(('smoke_' if smoke else '')+job['job_id']+'.log')
            command=[PYTHON,'-u',str(Path(__file__).resolve()),'--mode','worker','--job',job['job_id']]
            if smoke:command.append('--smoke')
            with log.open('x') as f:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                state['active_worker_pid']=child.pid;write(OUT/'status.json',state)
                print('START',state['phase'],job['job_id'],child.pid,stamp(),flush=True)
                code=child.wait()
            state.pop('active_worker_pid',None)
            if code:raise RuntimeError(f'{job["job_id"]} exited{code}; no auto retry')
            folder=OUT/('smoke' if smoke else 'train')/job['job_id']
            assert read(folder/'status.json')['phase']=='complete'
            if not smoke:
                assert read(folder/'test.json')['epochs_run']==60
                control=OUT/'train'/f'R2-T-00_seed{job["seed"]}'/'initialization.json'
                assert read(folder/'initialization.json')['sha256']==read(control)['sha256'],'Paired initialization mismatch'
        # Every existing mechanism was previously GPU-smoked; exercise all15 under the new pinned runner too.
        for job in plan['jobs'][:15]:
            launch(job,True)
            assert read(OUT/'smoke'/job['job_id']/'initialization.json')['sha256']==read(OUT/'smoke'/'R2-T-00_seed42'/'initialization.json')['sha256']
        write(OUT/'smoke_summary.json',dict(phase='complete',jobs=[j['job_id'] for j in plan['jobs'][:15]],test_access=False))
        for job in plan['jobs']:
            if shutil.disk_usage(OUT).free<40*1024**3:raise RuntimeError('Disk reserve breached; preserve results')
            launch(job);state['completed'].append(job['job_id']);write(OUT/'status.json',state);aggregate(plan)
        state.update(phase='complete',current_job=None,finished_at=stamp());write(OUT/'status.json',state)
    except BaseException:
        state.update(phase='failed',traceback=traceback.format_exc(),failed_at=stamp());write(OUT/'status.json',state);raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['queue','worker','verify'],required=True)
    ap.add_argument('--job');ap.add_argument('--smoke',action='store_true');cli=ap.parse_args()
    if cli.mode=='worker':worker(cli.job,cli.smoke)
    elif cli.mode=='queue':queue()
    else:
        assert [cap(i,{}) for i in [1,5,6,10,11,15,16,60]]==[.3,.3,.5,.5,.7,.7,.9,.9]
        assert all(cap(i,{'fixed_token_rate_cap':.9})==.9 for i in range(1,61))
        print('CPU schedule checks passed;30jobs=15settings*seeds42/45; no CUDA initialized')
