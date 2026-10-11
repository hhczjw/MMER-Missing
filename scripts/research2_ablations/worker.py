"""One new R2 model: train -> validation selection -> final test."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import time
import traceback
import torch
import numpy as np
from runtime import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--job',required=True);cli=ap.parse_args()
    manifest=read(cli.output/'plan.json');assert_frozen(manifest)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    device=torch.device('cuda:0');torch.cuda.set_device(0)
    job=next(j for j in manifest['jobs'] if j['job_id']==cli.job)
    track=job['track'];changes=job['changes']
    folder=cli.output/'train'/cli.job;folder.mkdir(parents=True,exist_ok=False)
    weights=Path(manifest['checkpoint_root'])/track/cli.job;weights.mkdir(parents=True,exist_ok=False)
    state=dict(phase='initializing',job=cli.job,pid=os.getpid(),physical_gpu=1,
               started_at=datetime.now().isoformat(timespec='seconds'))
    write(folder/'status.json',state)
    try:
        args=arguments(track,job['seed'],changes);set_seed(args.seed)
        # Native M constructs its model before loading data; native T loads first.
        if track=='R2-M':
            model=UMRMv3(args).to(device);train,valid,eval_args=loaders(args)
        else:
            train,valid,eval_args=loaders(args);model=UMRMv3(args).to(device)
        restore,rec_calls=interventions(model,changes)
        opt=optimizer(model,args);warm,plate=schedules(opt,args,track)
        base_lrs=[g['lr'] for g in opt.param_groups]
        best=None;best_epoch=0;stale=0;history=[]
        primary=weights/('best_MAE.pth' if track=='R2-M' else 'best_Acc7.pth')
        write(folder/'config.json',dict(args=vars(args),changes=changes,
              protocol=manifest['contracts'][track],checkpoint=str(primary)))
        for epoch in range(1,args.epochs+1):
            begin=time.time();torch.cuda.reset_peak_memory_stats()
            if track=='R2-T' and epoch<=5:
                for group,lr in zip(opt.param_groups,base_lrs):group['lr']=lr*epoch/5
            curriculum(model,args,track,epoch,changes)
            start_lrs=[g['lr'] for g in opt.param_groups]
            loss=train_epoch(model,train,TaskLoss(args.task_type,args.num_classes),opt,args,device,epoch,args.epochs)
            assert np.isfinite(loss),'Nonfinite train loss'
            if track=='R2-M' and args.use_curriculum:model.curriculum_scheduler.step()
            rec_calls.clear()
            result=validation(model,valid,args,track,device);value=summary(result,track)
            assert all(np.isfinite(v) for v in value.values())
            if improve(track,value,best):
                best=value.copy();best_epoch=epoch;stale=0
                checkpoint(primary,model,args,epoch,result,job)
                write(folder/'selection.json',dict(epoch=epoch,validation=result,checkpoint=str(primary)))
            else:stale+=1
            if track=='R2-M':
                if epoch<=5:warm.step()
                else:plate.step(value['MAE'])
            elif epoch>=5:plate.step(value['Acc7'])
            row=dict(epoch=epoch,loss=loss,validation=value,lr_start=start_lrs,
                     lr_next=[g['lr'] for g in opt.param_groups],seconds=time.time()-begin,
                     peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                     stale_epochs=stale,batches=160,token_stats=getattr(args,'token_train_last_stats',None))
            history.append(row);write(folder/'history.json',history)
            state.update(phase='training',epoch=epoch,max_epochs=args.epochs,best_epoch=best_epoch,
                         validation=value,epoch_seconds=row['seconds']);write(folder/'status.json',state)
            print(f'{cli.job} E{epoch:02d} loss={loss:.5f} validMAE={value["MAE"]:.5f} validAcc7={value["Acc7"]:.5f} validAcc2Non0={value["Acc2_nonzero"]:.5f} bestE={best_epoch} seconds={row["seconds"]:.1f}',flush=True)
            if stale>=15 and (track=='R2-M' or epoch>=20):break
        # Checkpoint selection is frozen before creating the final test loader.
        state.update(phase='testing');write(folder/'status.json',state)
        payload=torch.load(primary,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(payload['model'],strict=True);del payload
        test=MMDataLoader(eval_args,'test');assert len(test.dataset)==686
        prediction=folder/'test.predictions.npz'
        if track=='R2-M':result=evaluate_m(model,test,device,all_combos=True,archive=prediction)
        else:
            args.token_mask_seed=20261005
            result=evaluate_curve(model,test,args,device,RATES,prediction_archive=prediction)
        result.update(job=job,checkpoint=str(primary),selected_epoch=best_epoch,
                      epochs_run=epoch,selection='validation TAV MAE' if track=='R2-M' else 'validation Token10 Acc7, exact-tie MAE',
                      sample_count=686,prediction_archive=str(prediction))
        write(folder/'test.json',result)
        state.update(phase='complete',finished_at=datetime.now().isoformat(timespec='seconds'),
                     epochs_run=epoch,selected_epoch=best_epoch);write(folder/'status.json',state)
        print('TEST_COMPLETE',cli.job,flush=True);restore()
    except Exception:
        state.update(phase='failed',traceback=traceback.format_exc());write(folder/'status.json',state);raise


if __name__=='__main__':main()
