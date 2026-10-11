"""Physical GPU0 serial smoke-all then full 37x3-seed train/validation/test queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from plan import ROOT,SEEDS,METRICS,variants,jobs,sources,sha,atomic_json,assert_frozen,reference_arguments


def make_manifest(seeds,checkpoint_root):
    references=reference_arguments()
    paths=[Path(references['R1'][k]) for k in ['featurePath','feature_A','feature_V']]
    paths.append(Path(references['R2']['dataset_root_dir'])/references['R2']['featurePath'])
    return dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S'),seeds=seeds,device='physical GPU0',
                variants=variants(),jobs=jobs(seeds),source_sha256=sources(),checkpoint_root=str(checkpoint_root),
                reference_args=references,data_sha256={str(p):sha(p) for p in paths},
                validation_mask_seed=20261006,test_mask_seed=20261005,eval_batch=16,
                selection='R1 validation Non0 Acc2; R2-M six-combo MAE; R2-T Token10 Acc7; ties frozen in plan.py',
                precision_R1='bf16 autocast / FP32 parameters / gradient checkpointing',
                stages=['all 37 real-model smoke tests','37 seed42 jobs','37 seed44 jobs','37 seed45 jobs'],
                no_test_driven_variant_selection=True,no_HPO_following=True,
                changes_from_historical_reference=['R1 effective-length masking and current unimodal deep supervision',
                    'R2-M eval-mode detached clean targets and six-combo validation selection',
                    'separate input-mask RNG, no nonfinite-batch skipping'],
                live_sources={str(p):sha(p) for p in [ROOT/'models/umrm_v3.py',ROOT/'models/umrm_v2.py',ROOT/'scripts/train.py',ROOT/'scripts/train_token_input.py',
                    Path('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/models/classifier.py'),Path('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/layers/ism.py')]})


def aggregate(output,manifest):
    rows=[]
    for variant in manifest['variants']:
        members=[]
        for seed in manifest['seeds']:
            file=output/'train'/f'{variant["id"]}_seed{seed}'/'test.json'
            if file.exists():members.append(json.loads(file.read_text()))
        if not members:continue
        values={k:[m['summary'][k] for m in members] for k in METRICS}
        baseline_id=variant['track']+'-00';deltas={k:[] for k in METRICS}
        for m in members:
            reference=output/'train'/f'{baseline_id}_seed{m["seed"]}'/'test.json'
            if reference.exists():
                ref=json.loads(reference.read_text())['summary']
                for k in METRICS:deltas[k].append(m['summary'][k]-ref[k])
        rows.append(dict(variant=variant,seeds=[m['seed'] for m in members],n=len(members),
            mean={k:float(np.mean(x)) for k,x in values.items()},
            sample_std={k:float(np.std(x,ddof=1)) if len(x)>1 else None for k,x in values.items()},
            paired_delta_mean={k:float(np.mean(x)) if x else None for k,x in deltas.items()}))
    atomic_json(output/'summary.json',dict(status='complete' if all(r['n']==len(manifest['seeds']) for r in rows) and len(rows)==37 else 'partial',
                                          rows=rows,interpretation='mean/std of individual seed metrics; no prediction ensemble; all predeclared variants retained'))


def run_worker(output,checkpoint_root,manifest_path,row,stage):
    name=f'{row["id"]}_seed{row["seed"]}'
    folder=output/stage/name;ckpt=checkpoint_root/stage/name
    log=output/'logs'/f'{stage}_{name}.log'
    cmd=[sys.executable,str(ROOT/'scripts/ablations/worker.py'),'--job',row['id'],'--seed',str(row['seed']),
         '--stage',stage,'--output',str(folder),'--checkpoints',str(ckpt),'--manifest',str(manifest_path)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
    with log.open('x') as stream:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(output/'active_worker.json',dict(pid=process.pid,stage=stage,job=name,log=str(log),command=cmd))
        code=process.wait()
    if code:raise RuntimeError(f'{stage} failed: {name} exit={code}; see {log}')
    result=folder/('smoke.json' if stage=='smoke' else 'test.json')
    if not result.exists():raise RuntimeError('Worker ended without verified result')
    record=json.loads(result.read_text())
    if stage=='smoke':assert record['status']=='passed'
    else:assert record['audited']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    cli=p.parse_args();output=cli.output.resolve();ckpt=cli.checkpoints.resolve()
    if cli.seeds!=SEEDS:raise ValueError('Formal seed list is frozen to 42,44,45')
    if len(output.parts)<4 or len(ckpt.parts)<4:raise ValueError('Use explicit scoped result/checkpoint paths')
    if not cli.resume:
        if output.exists() or ckpt.exists():raise FileExistsError('Use exclusive new directories')
        needed=(8 if cli.smoke_only or cli.prepare_only else 200)*1024**3
        if shutil.disk_usage(ckpt.parent).free<needed:raise RuntimeError('Insufficient space for scoped checkpoint directory')
        output.mkdir(parents=True);ckpt.mkdir(parents=True);(output/'logs').mkdir()
        atomic_json(output/'plan.json',make_manifest(cli.seeds,ckpt))
    manifest_path=output/'plan.json';manifest=json.loads(manifest_path.read_text());assert_frozen(manifest)
    if str(ckpt)!=manifest['checkpoint_root'] or cli.seeds!=manifest['seeds']:raise ValueError('Resume arguments differ from frozen plan')
    if cli.prepare_only:
        print(manifest_path,flush=True)
        return
    # Lock physical GPU0 across queue processes. Never open GPU1.
    lock=open('/tmp/mmer_ablation_physical_gpu0.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(pid=os.getpid(),phase='smoke',started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                completed_smoke=[],completed_train=[],total_settings=37,total_training_jobs=111)
    try:
        for stage,items in [('smoke',[dict(v,seed=42) for v in manifest['variants']]),('train',manifest['jobs'])]:
            if stage=='train' and cli.smoke_only:break
            for row in items:
                name=f'{row["id"]}_seed{row["seed"]}'
                marker=output/stage/name/('smoke.json' if stage=='smoke' else 'test.json')
                if marker.exists():
                    value=json.loads(marker.read_text())
                    if (stage=='smoke' and value.get('status')!='passed') or (stage=='train' and not value.get('audited')):
                        raise RuntimeError(f'Unverified existing result: {marker}')
                else:
                    assert_frozen(manifest)
                    for path,expected in manifest['live_sources'].items():
                        if sha(path)!=expected:raise RuntimeError(f'Live source changed during isolation: {path}')
                    status.update(phase=stage,current_job=name);atomic_json(output/'status.json',status)
                    print('START',stage,name,flush=True)
                    run_worker(output,ckpt,manifest_path,row,stage)
                status['completed_'+stage].append(name);atomic_json(output/'status.json',status)
                if stage=='train':aggregate(output,manifest)
        status.update(phase='smoke_complete' if cli.smoke_only else 'complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        atomic_json(output/'status.json',status);print(status['phase'],flush=True)
    except BaseException as exc:
        status.update(phase='failed',error=str(exc),traceback=traceback.format_exc())
        atomic_json(output/'status.json',status);raise


if __name__=='__main__':main()
