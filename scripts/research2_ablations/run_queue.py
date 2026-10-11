"""Physical GPU1: preflight then a fixed 99-job serial R2 ablation queue."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import numpy as np
from runtime import read,write,sha,sources,assert_frozen,ROOT,SNAP,METRICS
from plan import SEEDS,variants,reference_contract


def task_order():
    tasks=[]
    for group in ['core','mechanism','policy']:
        for index in range(3):
            # Complete both references first, then the remaining core comparisons.
            for variant_index in range(18):
                for track in ['R2-M','R2-T']:
                    for v in variants(track):
                        if v['group']==group and int(v['id'].rsplit('-',1)[1])==variant_index:
                            seed=SEEDS[track][index]
                            tasks.append(dict(v,seed=seed,job_id=f'{v["id"]}_seed{seed}'))
    assert len(tasks)==99 and len({x['job_id'] for x in tasks})==99
    return tasks


def aggregate(output,manifest):
    rows=[]
    for track in ['R2-M','R2-T']:
        for variant in variants(track):
            members=[];paired=[]
            for seed in SEEDS[track]:
                p=output/'train'/f'{variant["id"]}_seed{seed}'/'test.json'
                if not p.exists():continue
                d=read(p)
                value=d['missing_avg'] if track=='R2-M' else d['avg_over_rates']
                extra=d.get('full_metrics')
                members.append(dict(seed=seed,metrics=value,full_metrics=extra))
                base=output/'train'/f'{track}-00_seed{seed}'/'test.json'
                if base.exists():
                    b=read(base);bv=b['missing_avg'] if track=='R2-M' else b['avg_over_rates']
                    paired.append({k:value[k]-bv[k] for k in METRICS})
            if not members:continue
            mean={k:float(np.mean([m['metrics'][k] for m in members])) for k in METRICS}
            std={k:float(np.std([m['metrics'][k] for m in members],ddof=1)) if len(members)>1 else None for k in METRICS}
            row=dict(variant=variant['id'],title=variant['title'],completed_repeats=len(members),
                     complete=len(members)==3,mean=mean,sample_std=std,members=members,
                     paired_delta_mean={k:float(np.mean([m[k] for m in paired])) for k in METRICS} if paired else None)
            if track=='R2-M':
                row['full_mean']={k:float(np.mean([m['full_metrics'][k] for m in members])) for k in METRICS}
            rows.append(row)
    write(output/'aggregate.json',dict(rows=rows,metric_protocol='Acc2/F1 Has0 and Non0; weighted F1; Acc7 clip+round',
                selection_frozen=True,no_test_based_retuning=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,required=True);cli=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    lockfile=(ROOT/'results/research2_gpu1.lock').open('a')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in [cli.output,cli.checkpoints]:
        if p.exists():raise FileExistsError(p)
    if shutil.disk_usage(cli.checkpoints.parent).free<180*1024**3:
        raise RuntimeError('At least 180 GiB free space required for 99 best checkpoints')
    smoke=read(ROOT/'results/research2_smoke_gpu1_20261006_v2/status.json')
    assert smoke['phase']=='complete' and len(smoke['passed'])==33
    expected=read(ROOT/'results/research2_smoke_gpu1_20261006_v2/source_sha256.json')
    for p,h in expected.items():
        rel=Path(p).relative_to('experiments/research2_smoke_gpu1_20261006/snapshot')
        assert sha(SNAP/rel)==h,'Formal snapshot differs from smoked native model/trainer'
    cli.output.mkdir(parents=True);cli.checkpoints.mkdir(parents=True);(cli.output/'logs').mkdir()
    prep=read(ROOT/'results/research2_ablation_preparation_20261006/plan.json')
    dataset=Path(prep['reference_args']['R2-M']['dataset_root_dir'])/prep['reference_args']['R2-M']['featurePath']
    contracts={t:reference_contract(t) for t in ['R2-M','R2-T']}
    contracts['R2-M'].update(corruption='frozen current padding-aware implicit post-ISM masks; same across new reference and variants',
                 historical_source_confidence='historical mask implementation not fully recoverable; new controlled cohort explicitly uses current smoked implementation',eval_batch_size=8)
    manifest=dict(jobs=task_order(),checkpoint_root=str(cli.checkpoints.resolve()),source_sha256=sources(),
                  data_sha256={str(dataset):sha(dataset)},contracts=contracts,physical_gpu=1,
                  created_at=datetime.now().isoformat(timespec='seconds'),core_jobs=48,total_jobs=99,
                  historical_modality_training_equivalence_not_claimed=True,
                  baseline='all variants compared to newly trained same-seed same-protocol reference; archived scores not substituted')
    write(cli.output/'plan.json',manifest)
    state=dict(phase='gates',pid=os.getpid(),physical_gpu=1,completed=[],total_jobs=99,
               started_at=datetime.now().isoformat(timespec='seconds'),formal_training_started=False)
    write(cli.output/'status.json',state)
    try:
        subprocess.run([sys.executable,str(Path(__file__).with_name('gates.py')),'--output',str(cli.output/'gates')],check=True)
        state.update(phase='train',formal_training_started=True,gates_passed=True)
        write(cli.output/'status.json',state)
        for job in manifest['jobs']:
            assert_frozen(manifest)
            state.update(current_job=job['job_id'],phase='train');write(cli.output/'status.json',state)
            print('START',job['job_id'],job['title'],flush=True)
            with (cli.output/'logs'/(job['job_id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('worker.py')),
                                '--output',str(cli.output),'--job',job['job_id']],stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(cli.output/'train'/job['job_id']/'status.json')['phase']=='complete'
            state['completed'].append(job['job_id']);aggregate(cli.output,manifest)
            write(cli.output/'status.json',state);print('DONE',job['job_id'],flush=True)
        state.update(phase='complete',finished_at=datetime.now().isoformat(timespec='seconds'))
        write(cli.output/'status.json',state)
    except Exception:
        state.update(phase='failed',traceback=traceback.format_exc());write(cli.output/'status.json',state);raise


if __name__=='__main__':main()
