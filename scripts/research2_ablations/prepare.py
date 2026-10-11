"""Produce an evidence-backed R2 plan without GPU allocation or model training."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from plan import ROOT,METRICS,SEEDS,variants,jobs,reference_contract


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def audit_token_archives():
    sys.path.insert(0,str(ROOT))
    from utils.metrics import eval_regression
    base=ROOT/'results/token_input_retrain_20261006'
    run=read(base/'run.json')
    assert all(digest(ROOT/p)==h for p,h in run['source_sha256'].items()),'Original Token source changed'
    members=[];identity=None;curves={}
    for seed in [44,45,46]:
        folder=base/f'seed{seed}';result=read(folder/'test.json')
        with np.load(folder/'test.predictions.npz') as archive:
            labels=archive['labels'];ids=archive['sample_ids']
            assert len(labels)==686 and len(set(ids))==686
            if identity is None:identity=(labels.copy(),ids.copy())
            else:assert np.array_equal(identity[0],labels) and np.array_equal(identity[1],ids)
            for rate,reported in result['per_rate'].items():
                actual=eval_regression(archive[f'r{rate}_preds'],labels)
                assert all(abs(actual[k]-reported[k])<1e-10 for k in METRICS),(seed,rate)
            curves[seed]=result['per_rate']
        point=read(folder/'selection.json')['Acc7']
        members.append(dict(seed=seed,selected_epoch=point['epoch'],metrics=result['avg_over_rates'],
                            config=str(folder/'config.json'),checkpoint=str(folder/'best_Acc7.pth')))
    values={k:np.asarray([r['metrics'][k] for r in members]) for k in METRICS}
    return dict(source_hashes_verified=True,predictions_recomputed=True,members=members,
        individual_mean={k:float(v.mean()) for k,v in values.items()},
        individual_sample_std={k:float(v.std(ddof=1)) for k,v in values.items()},
        prediction_ensemble=read(base/'ensemble.json')['avg_over_rates'],
        checkpoint_args=read(base/'seed45/config.json'),run=run,
        best_display_high_missing={k:float(np.mean([curves[45][r][k] for r in ['0.7','0.8','0.9']])) for k in METRICS})


def audit_modality():
    import torch
    full=read(ROOT/'results/NEW3_TAV_s44s42s43_MAEckpt.json')
    missing=read(ROOT/'results/NEW3_missing_s43s45s48_MAEckpt.json')
    paths=list(dict.fromkeys(full['checkpoints']+missing['checkpoints']))
    members=[];reference=None
    for path in paths:
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        args=payload['args'].copy();del payload
        comparable={k:v for k,v in args.items() if k!='seed'}
        if reference is None:reference=comparable
        else:assert comparable==reference,'Reported modality members have different saved configs'
        members.append(dict(checkpoint=path,args=args))
    names=['TA','TV','AV','T','A','V']
    summary={k:float(np.mean([missing[n][k] for n in names])) for k in METRICS}
    assert abs(summary['MAE']-missing['missing_avg_MAE'])<1e-10
    assert abs(summary['Acc7']-missing['missing_avg_Acc7'])<1e-10
    with_text={k:float(np.mean([missing[n][k] for n in ['TA','TV','T']])) for k in METRICS}
    without_text={k:float(np.mean([missing[n][k] for n in ['AV','A','V']])) for k in METRICS}
    return dict(saved_configs_consistent=True,members=members,full_reported=full['TAV'],
                missing_summary=summary,missing_per_combo={n:missing[n] for n in names},
                with_text_mean=with_text,without_text_mean=without_text,
                full_and_missing_members_differ=full['checkpoints']!=missing['checkpoints'],
                archive_mixes_bins_and_clip_directories=True,
                evaluation_definition='main Acc7 values are clip+round; secondary Acc7_bins is excluded',
                representative_checkpoint=next(m for m in members if m['args']['seed']==43))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/research2_ablation_preparation_20261006')
    cli=parser.parse_args()
    if cli.output.exists():raise FileExistsError('Use a fresh preparation directory')
    token=audit_token_archives();modality=audit_modality()
    hpo=read(ROOT/'results/token_seed42_hpo_20261006/summary.json')
    manifest=dict(created_at=datetime.now().isoformat(timespec='seconds'),phase='prepared_not_launched',
        gpu_used=False,formal_training_started=False,gpu_smoke_passed=False,
        ready_for_training=False,
        tracks={t:dict(reference_contract=reference_contract(t),variants=variants(t),seeds=SEEDS[t]) for t in SEEDS},
        reference_args={'R2-M':modality['representative_checkpoint']['args'],'R2-T':token['checkpoint_args']},
        all_jobs=jobs(),core_jobs=jobs(core_only=True),
        counts=dict(modality_settings=18,token_settings=15,total_settings=33,
                    full_training_jobs=len(jobs()),core_training_jobs=len(jobs(core_only=True))),
        evidence=dict(modality=modality,token=token,
                      hpo_guard_not_met=hpo['validation_selection'].get('guard_not_met',False)),
        gates_pending=['isolated native trainer implementation, with baseline-default equivalence',
                       'modality historical implicit-mask source parity; do not silently retain new padding-aware mask policy',
                       'fixed historical checkpoint GPU prediction parity for both tracks',
                       'same model/init/masks one-step loss-components and gradient parity',
                       'KD-off must preserve nonzero RCD student gradient and teacher clean targets',
                       'feature reconstruction-off must preserve contrastive gradients',
                       'full 160-batch training epoch and fixed validation pressure test per track',
                       'every intervention GPU smoke and source/config freeze before training'],
        policy='prepare only; never append to or mutate the running R1 queue; no test-based retuning or seed dropping',
        protected_R1_sources=read(ROOT/'results/mosi_ablations_p0b_0919_gpu0_20261006_v2/plan.json')['source_sha256'])
    sources=[ROOT/'scripts/train.py',ROOT/'scripts/train_token_input.py',ROOT/'models/umrm_v3.py',ROOT/'models/umrm_v2.py',
             ROOT/'dataset/data_loader.py',ROOT/'losses/task_losses.py',ROOT/'utils/token_protocol.py',ROOT/'utils/token_evaluation.py',ROOT/'utils/metrics.py',
             Path(__file__),Path(__file__).with_name('plan.py')]
    manifest['source_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sources}
    cli.output.mkdir(parents=True)
    (cli.output/'plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('R2 PREPARED: 18 modality settings + 15 Token settings; 99 proposed jobs (48 core); NO GPU/TRAINING')
    print(cli.output/'plan.json')


if __name__=='__main__':main()
