"""GPU1 seed42 Token HPO: 10x20-epoch screening, 2x60-epoch confirmation.

Shared training/model sources remain untouched. Test data is opened only after
both stages, when validation has selected the final checkpoints. No test-driven
ensemble, calibration or follow-up search is performed by this pipeline.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import _path_setup
from train_token_input import arguments,make_optimizer,write_json,RATES
from token_hpo_plan import candidates,joint,order,champions,finalists
from dataset.data_loader import MMDataLoader
from models import UMRMv3
from scripts.train import train_epoch,TaskLoss
from utils.seed import set_seed
from utils.metrics import eval_regression
from utils.token_evaluation import evaluate_curve


def make_args(cli,candidate,epochs):
    proxy=SimpleNamespace(epochs=epochs,batch_size=8,lr=3e-4)
    args=arguments(proxy,42)
    for key,value in candidate['params'].items():setattr(args,key,value)
    args.token_mask_seed=20261006
    return args


def smoke(cli,device):
    stress=candidates()[-1]
    stress['params'].update(batch_size=16,use_aux_cls=True)
    args=make_args(cli,stress,20)
    args.token_train_fixed_rate=.9
    set_seed(42)
    model=UMRMv3(args).to(device)
    batch=next(iter(MMDataLoader(args,'train')))
    before=model.proj_audio.weight.detach().clone()
    start=time.time()
    loss=train_epoch(model,[batch],TaskLoss(args.task_type,args.num_classes),make_optimizer(model,args),args,device,1,20)
    assert np.isfinite(loss) and not torch.equal(before,model.proj_audio.weight.detach())
    curve=evaluate_curve(model,[batch],args,device,[0.,.5,.9])
    assert all(np.isfinite(x) for x in curve['avg_over_rates'].values())
    result=dict(status='passed',smoke_only=True,seed=42,batch_size=16,lora_r=16,
                auxiliary_ce7=True,missing_rate=.9,loss=loss,parameter_updated=True,
                device=str(device),peak_gpu_mb=torch.cuda.max_memory_allocated(device)/1024**2,
                seconds=time.time()-start)
    write_json(cli.output/'smoke.json',result)
    print('SMOKE_PASSED',json.dumps(result),flush=True)
    del model,batch
    gc.collect();torch.cuda.empty_cache()


def train_trial(cli,candidate,stage,epochs,device,status):
    args=make_args(cli,candidate,epochs)
    set_seed(42)
    folder=cli.output/stage/candidate['id'];folder.mkdir(parents=True)
    checkpoint_dir=cli.checkpoint_root/stage/candidate['id'];checkpoint_dir.mkdir(parents=True)
    write_json(folder/'config.json',vars(args))
    status.update(phase='training',stage=stage,trial=candidate['id'],epoch=0,max_epochs=epochs,
                  active_params=candidate['params'])
    write_json(cli.output/'status.json',status)
    train_loader=MMDataLoader(args,'train')
    batch_size=args.batch_size;args.batch_size=16
    valid_loader=MMDataLoader(args,'valid');args.batch_size=batch_size
    model=UMRMv3(args).to(device)
    optimizer=make_optimizer(model,args)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5,min_lr=1e-6)
    base_lrs=[g['lr'] for g in optimizer.param_groups]
    best,history,stale={},[],0
    for epoch in range(1,epochs+1):
        start=time.time()
        if epoch<=5:
            for g,lr in zip(optimizer.param_groups,base_lrs):g['lr']=lr*epoch/5
        args.token_train_rate_cap=min(.9,.3+.2*(epoch-1))
        model.token_missing_rate=args.token_train_rate_cap
        loss=train_epoch(model,train_loader,TaskLoss(args.task_type,args.num_classes),optimizer,args,device,epoch,epochs)
        validation=evaluate_curve(model,valid_loader,args,device,RATES)
        avg=validation['avg_over_rates']
        assert all(np.isfinite(v) for v in avg.values()),'Nonfinite validation metric'
        joint_improved=False
        for objective in ['joint','Acc7']:
            improved=objective not in best or order(avg,objective)>order(best[objective]['avg'],objective)
            if improved:
                checkpoint=checkpoint_dir/f'best_{objective}.pth'
                point=dict(epoch=epoch,avg=avg,joint=joint(avg),checkpoint=str(checkpoint))
                tmp=checkpoint.with_suffix('.pth.tmp')
                torch.save(dict(model=model.state_dict(),args=vars(args).copy(),seed=42,
                                stage=stage,candidate=candidate,selection=point,validation=validation),tmp)
                tmp.replace(checkpoint)
                best[objective]=point
                if objective=='joint':joint_improved=True
        stale=0 if joint_improved else stale+1
        record=dict(epoch=epoch,loss=loss,validation=validation,joint=joint(avg),
                    train=args.token_train_last_stats,learning_rates=[g['lr'] for g in optimizer.param_groups],
                    seconds=time.time()-start)
        history.append(record)
        write_json(folder/'history.json',history)
        write_json(folder/'selection.json',best)
        status.update(phase='training',stage=stage,trial=candidate['id'],epoch=epoch,max_epochs=epochs,
                      best_validation_joint=best['joint']['joint'],best_validation_Acc7=best['Acc7']['avg']['Acc7'])
        write_json(cli.output/'status.json',status)
        print(f'{stage} {candidate["id"]} E{epoch:02d} loss={loss:.4f} valA7={avg["Acc7"]:.5f} valA2n={avg["Acc2_nonzero"]:.5f} J={joint(avg):.5f} bestJ={best["joint"]["joint"]:.5f} sec={record["seconds"]:.1f}',flush=True)
        if epoch>=5:scheduler.step(joint(avg))
        # Equal screening budget; only long confirmation uses early stopping.
        if stage=='confirm' and epoch>=20 and stale>=15:break
    row=dict(id=candidate['id'],params=candidate['params'],stage=stage,epochs_run=len(history),best=best,
             config=str(folder/'config.json'),history=str(folder/'history.json'))
    write_json(folder/'result.json',row)
    del model,optimizer,scheduler,train_loader,valid_loader
    gc.collect();torch.cuda.empty_cache()
    return row


def final_selection(screen,confirmed):
    # The screen reference supplies a fixed Non0 safeguard, even if it is not a finalist.
    reference=next(r for r in screen if r['id']=='00_reference')
    floor=reference['best']['joint']['avg']['Acc2_nonzero']-.01
    points=[dict(trial=row['id'],selector=key,selection=point)
            for row in confirmed for key,point in row['best'].items()]
    j=max(points,key=lambda p:order(p['selection']['avg'],'joint'))
    eligible=[p for p in points if p['selection']['avg']['Acc2_nonzero']>=floor]
    chosen=dict(joint=j,acc2_nonzero_floor=floor)
    if eligible:chosen['Acc7_guarded']=max(eligible,key=lambda p:order(p['selection']['avg'],'Acc7'))
    else:chosen['guard_not_met']=True
    return chosen


def evaluate_selected(cli,chosen,confirmed,device,status):
    # No test loader exists before this point.
    result={}
    unique={}
    for objective,point in chosen.items():
        if not isinstance(point,dict):continue
        unique.setdefault(point['selection']['checkpoint'],dict(point=point,objectives=[]))['objectives'].append(objective)
    for index,(checkpoint,entry) in enumerate(unique.items()):
        point=entry['point'];row=next(r for r in confirmed if r['id']==point['trial'])
        args=make_args(cli,row,cli.confirm_epochs);args.batch_size=16;args.token_mask_seed=20261005
        set_seed(42)
        model=UMRMv3(args).to(device)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['selection']==point['selection']
        model.load_state_dict(state['model'],strict=True);del state
        loader=MMDataLoader(args,'test')
        folder=cli.output/'final_test'/f'{index:02}_{point["trial"]}_{point["selector"]}';folder.mkdir(parents=True)
        archive=folder/'predictions.npz'
        test=evaluate_curve(model,loader,args,device,RATES,prediction_archive=archive)
        with np.load(archive) as saved:
            assert len(saved['labels'])==686
            for rate,metrics in test['per_rate'].items():
                recomputed=eval_regression(saved[f'r{rate}_preds'],saved['labels'])
                assert all(abs(metrics[k]-float(v))<1e-10 for k,v in recomputed.items())
        test.update(checkpoint=checkpoint,trial=point['trial'],selection=point['selection'],
                    objectives=entry['objectives'],seed=42,prediction_archive=str(archive),n_samples=686)
        write_json(folder/'test.json',test)
        for objective in entry['objectives']:result[objective]=test
        print('FINAL_TEST',point['trial'],entry['objectives'],json.dumps(test['avg_over_rates']),flush=True)
        del model,loader;gc.collect();torch.cuda.empty_cache()
    paper_rows=json.loads((cli.output/'plan.json').read_text())['paper_reference_rows']
    comparison={}
    for objective,test in result.items():
        m=test['avg_over_rates']
        comparison[objective]=[dict(method=r['method'],reported_Acc7=r['Acc7'],reported_Acc2_nonzero=r['Acc2_nonzero'],
            delta_Acc7_pp=100*m['Acc7']-r['Acc7'],delta_Acc2_nonzero_pp=100*m['Acc2_nonzero']-r['Acc2_nonzero'],
            both_reported_numbers_exceeded=100*m['Acc7']>r['Acc7'] and 100*m['Acc2_nonzero']>r['Acc2_nonzero']) for r in paper_rows]
    summary=dict(seed=42,protocol='input_valid_aware',test_mask_seed=20261005,
                 validation_selection=chosen,tests=result,paper_reference=comparison,
                 interpretation='Single-seed preliminary HPO. Paper features, masks and seed averaging differ; numeric exceedance is not matched-protocol SOTA evidence.',
                 follow_up='Freeze selected parameters, confirm at least three seeds, retain individual metrics and mean/std separately from ensembles.')
    write_json(cli.output/'summary.json',summary)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoint-root',type=Path,required=True)
    ap.add_argument('--device',default='cuda:1')
    ap.add_argument('--screen-epochs',type=int,default=20)
    ap.add_argument('--confirm-epochs',type=int,default=60)
    ap.add_argument('--smoke-only',action='store_true')
    cli=ap.parse_args()
    if cli.screen_epochs<1 or cli.confirm_epochs<cli.screen_epochs:ap.error('Invalid epoch budgets')
    cli.output=cli.output.resolve();cli.checkpoint_root=cli.checkpoint_root.resolve()
    if cli.output.exists() or cli.checkpoint_root.exists():raise FileExistsError('Use exclusive new HPO directories')
    if shutil.disk_usage(cli.checkpoint_root.parent).free<50*1024**3:raise RuntimeError('Need >=50 GiB free for full-weight candidate checkpoints')
    cli.output.mkdir(parents=True);cli.checkpoint_root.mkdir(parents=True)
    torch.set_num_threads(4)
    device=torch.device(cli.device);torch.cuda.set_device(device)
    status=dict(pid=os.getpid(),phase='smoke',seed=42,device=str(device),started_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed=[])
    write_json(cli.output/'status.json',status)
    plan=dict(seed=42,device=str(device),candidates=candidates(),screen_epochs=cli.screen_epochs,
              confirm_epochs=cli.confirm_epochs,finalists=2,
              objective='J = 0.5*(validation Token10 mean Acc7 + validation Token10 mean Acc2 Non0); ties Acc7, Acc2 Non0, lower MAE',
              guard='Acc7 champion must retain validation Non0 within 1 percentage point of reference joint checkpoint',
              validation_mask_seed=20261006,test_mask_seed=20261005,eval_batch_size=16,
              protocol='input_valid_aware; full three modalities; same corruption helpers as GPU0',
              checkpoint_root=str(cli.checkpoint_root),initialization='fresh per candidate, pretrained RoBERTa only',
              schedule='warmup 5, plateau on J; screening fixed budget; confirmation min20/patience15',
              paper_reference_rows=json.loads((ROOT/'docs/bjtu_missing_comparisons.json').read_text())['token'],
              no_test_during_search=True,dropout_scope='prediction / auxiliary heads; ISM dropout argument is unused in its current implementation',
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),ROOT/'scripts/token_hpo_plan.py',ROOT/'scripts/train_token_input.py',
                                       ROOT/'scripts/train.py',ROOT/'models/umrm_v3.py',ROOT/'models/umrm_v2.py',
                                       ROOT/'utils/token_protocol.py',ROOT/'utils/token_evaluation.py',ROOT/'utils/metrics.py']})
    write_json(cli.output/'plan.json',plan)
    try:
        smoke(cli,device)
        if not cli.smoke_only:
            screen=[]
            for candidate in candidates():
                screen.append(train_trial(cli,candidate,'screen',cli.screen_epochs,device,status))
                status['completed'].append('screen/'+candidate['id'])
                write_json(cli.output/'screen_leaderboard.json',screen)
            selected=finalists(screen)
            write_json(cli.output/'shortlist.json',dict(ids=selected,screen_champions=champions(screen)))
            confirmed=[]
            for name in selected:
                candidate=next(c for c in candidates() if c['id']==name)
                confirmed.append(train_trial(cli,candidate,'confirm',cli.confirm_epochs,device,status))
                status['completed'].append('confirm/'+name)
                write_json(cli.output/'confirm_leaderboard.json',confirmed)
            chosen=final_selection(screen,confirmed)
            write_json(cli.output/'frozen_test_selection.json',chosen)
            status.update(phase='testing');write_json(cli.output/'status.json',status)
            evaluate_selected(cli,chosen,confirmed,device,status)
        status.update(phase='smoke_passed' if cli.smoke_only else 'complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(cli.output/'status.json',status)
    except BaseException as error:
        status.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(cli.output/'status.json',status)
        raise


if __name__=='__main__':main()
