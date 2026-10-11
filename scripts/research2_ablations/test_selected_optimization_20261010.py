"""Final test of validation-selected recipe and control, never test selection."""
from copy import deepcopy
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/'results/research2_token_optimization_20261010'
OUT=TRAIN/'selected_test_20261010'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)


def read(path):return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=False)
    state=dict(phase='preflight',pid=os.getpid(),gpu=0,completed=[])
    write(OUT/'status.json',state)
    lock=None
    try:
        ranking=read(TRAIN/'validation_ranking.json')
        if read(TRAIN/'status.json')['phase']!='complete' or read(TRAIN/'gpu0_restart_status.json')['phase']!='complete':
            raise RuntimeError('Optimization incomplete')
        winner=ranking['ranked'][0]['candidate']['id']
        selected=[row for row in read(TRAIN/'validation_summary.json') if row['candidate'] in (winner,'control')]
        if {(r['candidate'],r['seed']) for r in selected}!={(c,s) for c in (winner,'control') for s in (42,45)}:
            raise RuntimeError('Fixed selected/control seeds missing')
        write(OUT/'plan.json',dict(winner=winner,selection='pre-existing two-seed validation meanAcc7, tieMAE; frozen before test',
            selected=selected,validation_ranking=ranking,rates=[i/10 for i in range(10)],eval_batch=16,
            test_mask_seed=20261005,physical_gpu=0,no_test_selection=True,
            ensemble='Prespecified equal-weight42/45 predictions for winner and control; NOT metric average',
            no_training=True))
        lock=(ROOT/'results/research2_gpu0.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected='GPU-07abbe75-7a55-4623-70aa-a9572c83957e'
        gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        if expected not in gpu or expected in apps:raise RuntimeError('GPU0 identity/idle check failed')
        os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        sys.path.insert(0,str(ROOT/'scripts/checkpoint_nvme_20261007'))
        from data_cache import install_cache
        if not install_cache(ROOT/'results/checkpoint_nvme_staging_20261007/input_cache/index.json'):
            raise RuntimeError('Input cache unavailable')
        import runtime as rt
        import numpy as np
        import torch
        rt.assert_frozen(rt.read(ROOT/'results/research2_ablations_gpu1_20261006_v3/plan.json'))
        device=torch.device('cuda:0')
        rows=[]
        for row in selected:
            job=row['job'];folder=OUT/job;folder.mkdir()
            state.update(phase='test',current_job=job);write(OUT/'status.json',state)
            cp=TRAIN/job/'best_Acc7.pth'
            cp_sha=rt.sha(cp)
            payload=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
            if payload['epoch']!=row['best_epoch'] or payload['validation']['avg_over_rates']!=row['validation']:
                raise RuntimeError('Checkpoint/validation selection mismatch')
            args=SimpleNamespace(**payload['args']);args.batch_size=16;args.num_workers=0;args.token_mask_seed=20261005
            rt.set_seed(args.seed)
            model=rt.UMRMv3(args).to(device)
            model.load_state_dict(payload['model'],strict=True)
            del payload
            test=rt.MMDataLoader(args,'test')
            if len(test.dataset)!=686:raise RuntimeError('Test sample count changed')
            result=rt.evaluate_curve(model,test,args,device,rt.RATES,prediction_archive=folder/'predictions.npz')
            result.update(job=job,candidate=row['candidate'],seed=row['seed'],selected_epoch=row['best_epoch'],
                          checkpoint=str(cp),checkpoint_sha256=cp_sha,selection='validation-only',sample_count=686)
            write(folder/'test.json',result)
            arrays=np.load(folder/'predictions.npz')
            for rate in rt.RATES:
                measured=rt.eval_regression(arrays[f'r{rate:.1f}_preds'],arrays['labels'])
                key=f'r{rate:.1f}'
                # Native rate-key spelling is inspected and matched below.
                expected_metrics=result['per_rate'][f'{rate:.1f}']
                if any(abs(float(measured[k])-expected_metrics[k])>1e-6 for k in measured):
                    raise RuntimeError('Archived prediction metric mismatch')
            rows.append(result);state['completed'].append(job);write(OUT/'status.json',state)
            print('TEST',job,result['avg_over_rates'],flush=True)
            del model,test,arrays;gc.collect();torch.cuda.empty_cache()
        groups=[]
        for candidate in (winner,'control'):
            members=[r for r in rows if r['candidate']==candidate]
            keys=members[0]['avg_over_rates']
            mean={k:statistics.mean(r['avg_over_rates'][k] for r in members) for k in keys}
            sd={k:statistics.stdev(r['avg_over_rates'][k] for r in members) for k in keys}
            packs=[np.load(OUT/r['job']/'predictions.npz') for r in members]
            if not all(np.array_equal(packs[0]['labels'],p['labels']) and np.array_equal(packs[0]['sample_ids'],p['sample_ids']) for p in packs):
                raise RuntimeError('Ensemble sample order mismatch')
            per_rate={}
            ensemble_arrays=dict(labels=packs[0]['labels'],sample_ids=packs[0]['sample_ids'])
            for rate in rt.RATES:
                key=f'r{rate:.1f}';pred=np.mean([p[key+'_preds'] for p in packs],axis=0)
                per_rate[key]={k:float(v) for k,v in rt.eval_regression(pred,packs[0]['labels']).items()}
                ensemble_arrays[key+'_preds']=pred
            ensemble_mean={k:float(np.mean([v[k] for v in per_rate.values()])) for k in keys}
            np.savez_compressed(OUT/(candidate+'_ensemble_predictions.npz'),**ensemble_arrays)
            groups.append(dict(candidate=candidate,members=members,metric_mean=mean,sample_std=sd,
                               fixed_prediction_ensemble=dict(seeds=[42,45],weights=[.5,.5],per_rate=per_rate,avg_over_rates=ensemble_mean)))
        write(OUT/'summary.json',dict(validation_winner=winner,groups=groups,
            limitation='Two-seed development evaluation; not independent confirmatory ablation and not a test-chosen best seed/recipe',
            metric_mean_is_not_prediction_ensemble=True))
        state.update(phase='complete',finished_at=datetime.now().astimezone().isoformat());write(OUT/'status.json',state)
    except BaseException:
        state.update(phase='failed',error=traceback.format_exc());write(OUT/'status.json',state);raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
