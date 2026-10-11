"""Default adapter equivalence and archived checkpoint prediction checks."""
import argparse
import os
from pathlib import Path
import time
import torch
import numpy as np
from runtime import *
from scripts.train import evaluate_combos


def parity(track,output):
    prep=read(ROOT/'results/research2_ablation_preparation_20261006/plan.json')
    args=arguments(track,43 if track=='R2-M' else 45)
    device=torch.device('cuda:0');set_seed(args.seed)
    model=UMRMv3(args).to(device)
    loader=MMDataLoader(args,'train');batch=next(iter(loader))
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    if track=='R2-T':args.token_train_fixed_rate=.6;args.token_train_rate_cap=.7
    # A nonzero feature rate checks the current padding-aware implicit mask branch.
    if model.curriculum_scheduler is not None:model.curriculum_scheduler.current_epoch=30
    native=[]
    for adapted in [False,True]:
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in initial:p.copy_(initial[n].to(device))
        set_seed(args.seed+101)
        restore,raw=(interventions(model,{}) if adapted else (lambda:None,[]))
        opt=optimizer(model,args)
        value=train_epoch(model,[batch],TaskLoss(args.task_type,args.num_classes),opt,args,device,
                          31 if track=='R2-M' else 3,60)
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        native.append((value,grads));restore();del opt;model.zero_grad(set_to_none=True)
    assert native[0][0]==native[1][0],'Default total loss differs from native'
    assert native[0][1].keys()==native[1][1].keys()
    grad_error=max(float((g-native[1][1][n]).abs().max()) for n,g in native[0][1].items())
    assert grad_error<1e-7,grad_error
    del initial,native
    args.token_train_fixed_rate=None
    if track=='R2-M':
        ckpt=prep['evidence']['modality']['representative_checkpoint']['checkpoint']
        expected=read(ROOT/'results/umrmv3_kd0.3_recon_inj_rbl_rbp_lora_bi_uni0.1_g0.05_rcd0.1_cdkd_bal_ch_MOSI_seed43.json')
    else:
        ckpt=str(ROOT/'results/token_input_retrain_20261006/seed45/best_Acc7.pth')
        expected=read(ROOT/'results/token_input_retrain_20261006/seed45/test.json')
    payload=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(payload['model'],strict=True);del payload
    test_args=deepcopy(args);test_args.batch_size=8 if track=='R2-M' else 16
    test=MMDataLoader(test_args,'test')
    begin=time.time()
    if track=='R2-M':
        result=evaluate_m(model,test,device,all_combos=True,archive=output/'modality_archive_recomputed.npz')
        differences={c:{k:result['per_combo'][c][k]-expected[c][k] for k in METRICS} for c in result['per_combo']}
        maximum=max(abs(v) for row in differences.values() for v in row.values())
        write(output/'modality_archive_diagnostics.json',dict(actual=result,expected=expected,differences=differences,checkpoint=ckpt))
        native_reference=evaluate_combos(model,test,args,device)
        native_error=max(abs(result['per_combo'][c][k]-native_reference[c][k]) for c in result['per_combo'] for k in METRICS)
        assert native_error<1e-4,('Current native evaluator mismatch',native_error)
        # Historical JSON has no individual predictions. Preserve a discrepancy
        # explicitly; a freshly trained controlled cohort uses the verified native path.
        numeric_errors=[abs(v) for row in differences.values() for k,v in row.items() if k not in ['Acc7']]
        assert max(numeric_errors)<1e-4,'Historical numeric differences need investigation'
        counts={c:round((result['per_combo'][c]['Acc7']-expected[c]['Acc7'])*686) for c in result['per_combo']}
        assert all(abs(v)<=1 for v in counts.values()),'Unexpected multiple-sample Acc7 discrepancy'
        archive_exact=maximum<1e-4
    else:
        args.token_mask_seed=20261005
        artifact=output/'token_archive_recomputed.npz'
        result=evaluate_curve(model,test,args,device,RATES,prediction_archive=artifact)
        with np.load(artifact) as current,np.load(ROOT/'results/token_input_retrain_20261006/seed45/test.predictions.npz') as old:
            assert np.array_equal(current['labels'],old['labels'])
            assert np.array_equal(current['sample_ids'],old['sample_ids'])
            maximum=max(float(np.max(np.abs(current[f'r{r:.1f}_preds']-old[f'r{r:.1f}_preds']))) for r in RATES)
            mean_error=max(float(np.mean(np.abs(current[f'r{r:.1f}_preds']-old[f'r{r:.1f}_preds']))) for r in RATES)
            # GPU1 recomputation differs slightly from GPU0 floating-point outputs.
            # Require both small numeric error and EXACT classification metrics.
            assert all(np.allclose(current[f'r{r:.1f}_preds'],old[f'r{r:.1f}_preds'],atol=5e-4,rtol=1e-5) for r in RATES)
        assert maximum<5e-4 and mean_error<2e-5,(track,maximum,mean_error)
        archive_exact=True;native_error=0.;counts={}
        for r in result['per_rate']:
            for key in METRICS:
                assert abs(result['per_rate'][r][key]-expected['per_rate'][r][key])<2e-6,(r,key)
                if key.startswith(('Acc','F1')):
                    assert result['per_rate'][r][key]==expected['per_rate'][r][key],(r,key)
    row=dict(status='passed',native_default_loss=value,default_gradient_max_abs_error=grad_error,
             archive_checkpoint=ckpt,archive_max_abs_error=maximum,archive_seconds=time.time()-begin,
             archive_exact_metric_parity=archive_exact,current_native_evaluator_max_abs_error=native_error,
             historical_acc7_sample_count_differences=counts,
             historical_training_source_proven=False if track=='R2-M' else True,
             training_mask_contract='current padding-aware implicit feature masks, all variants same baseline' if track=='R2-M' else 'original input_valid_aware')
    write(output/(track+'.json'),row)
    print('GATE_PASSED',track,row,flush=True)
    del model;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);cli=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.cuda.set_device(0)
    cli.output.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'results/research2_ablations_gpu1_20261006_v2'
    reused=False
    if (previous/'gates/R2-M.json').exists():
        old=read(previous/'plan.json');current=read(cli.output.parent/'plan.json')
        keys=[k for k in old['source_sha256'] if '/snapshot/' in k or k.endswith(('runtime.py','smoke_gpu1.py','plan.py'))]
        row=read(previous/'gates/R2-M.json')
        proof=previous/'gates/R2-M.json'
        if (all(current['source_sha256'].get(k)==old['source_sha256'][k] for k in keys)
            and current['data_sha256']==old['data_sha256'] and row['status']=='passed'
            and Path(row['archive_checkpoint']).stat().st_mtime<=proof.stat().st_mtime):
            row['reused_from']=str(proof);write(cli.output/'R2-M.json',row);reused=True
            print('REUSED_UNCHANGED_NATIVE_MODALITY_GATE',flush=True)
    if not reused:parity('R2-M',cli.output)
    parity('R2-T',cli.output)
    write(cli.output/'summary.json',dict(status='passed',tracks=['R2-M','R2-T'],
          historical_modality_mask_uncertainty_disclosed=True,all_current_variants_share_frozen_mask_policy=True))


if __name__=='__main__':main()
