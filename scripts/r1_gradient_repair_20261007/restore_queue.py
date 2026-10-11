"""Reuse audited R1 results, then detach the isolated GPU0 repair queue.

No old result, checkpoint or source is deleted or overwritten. Completed jobs
remain linked to their original provenance. The failed R1-04 is NOT reused.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'experiments/ablation_p0b_0919_gradient_repair_20261007/runner'
sys.path.insert(0,str(RUNNER))
from plan import assert_frozen,atomic_json,sha
from worker import configure


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    cli=parser.parse_args()
    old=ROOT/'results/mosi_ablations_p0b_0919_gpu0_20261006_v2'
    output=ROOT/'results/mosi_ablations_p0b_0919_gradient_repair_20261007'
    plan=json.loads((output/'plan.json').read_text())
    assert_frozen(plan)
    old_plan=json.loads((old/'plan.json').read_text())
    assert plan['reference_args']==old_plan['reference_args']
    assert plan['data_sha256']==old_plan['data_sha256']
    unchanged_gpu1=json.loads((ROOT/'results/research2_ablations_gpu1_20261006_v3/plan.json').read_text())
    for path,expected in unchanged_gpu1['source_sha256'].items():
        assert sha(ROOT/path)==expected,f'GPU1 frozen source changed: {path}'
    completed=json.loads((old/'status.json').read_text())['completed_train']
    assert completed==[f'R1-{i:02d}_seed2024' for i in range(4)]
    for stage in ['reference','smoke']:
        target=output/stage
        if not target.exists():target.symlink_to(old/stage,target_is_directory=True)
        assert target.resolve()==old/stage
    (output/'train').mkdir(exist_ok=True)
    for name in completed:
        result=json.loads((old/'train'/name/'test.json').read_text())
        assert result['audited'] and result['dataset']['n_samples']==686
        job=next(v for v in plan['variants'] if v['id']==result['variant']['id'])
        assert job==result['variant']
        original_args=json.loads((old/'train'/name/'config.json').read_text())['args']
        current_args=vars(configure('R1',2024,120,job['changes'],plan['reference_args']))
        assert original_args==current_args, name
        target=output/'train'/name
        if not target.exists():target.symlink_to(old/'train'/name,target_is_directory=True)
        assert target.resolve()==old/'train'/name
    failed_args=json.loads((old/'train/R1-04_seed2024/config.json').read_text())['args']
    restarted_args=vars(configure('R1',2024,120,{'gcmn':False},plan['reference_args']))
    assert failed_args==restarted_args
    provenance=dict(original_queue=str(old),original_plan=str(old/'plan.json'),
        retained_completed_jobs=completed,retained_via='read-only provenance links',
        old_outputs_overwritten=False,restart_job='R1-04_seed2024',
        restart_from='fresh initialization and original seed; not epoch-28 resume',
        reason='old run lacks last model/optimizer/scheduler/RNG state',
        original_training_arguments_identical=True,dataset_checksums_identical=True,
        GPU1_frozen_sources_unchanged=True,
        numerical_change='native normal path unchanged; verified same-forward emergency backward and robust norm clipping')
    atomic_json(output/'recovery_provenance.json',provenance)
    if not cli.launch:
        print(json.dumps(provenance,ensure_ascii=False,indent=2));return
    for name in ['normal','injected']:
        smoke=json.loads((output/'repair_checks'/name/'smoke.json').read_text())
        assert smoke['status']=='passed' and smoke['train']['skipped_batches']==0
        assert smoke['train']['reference_backward_retries']==(1 if name=='injected' else 0)
        assert smoke['parameter_updated'] and smoke['checkpoint_roundtrip']
    log=ROOT/'logs/mosi_ablations_p0b_0919_gradient_repair_20261007.log'
    command=[sys.executable,str(RUNNER/'run_queue.py'),'--output',str(output),
             '--checkpoints',plan['checkpoint_root'],'--resume']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                               stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,command=command,log=str(log),output=str(output),
                checkpoints=plan['checkpoint_root'])
    atomic_json(log.with_suffix('.launch.json'),record)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
