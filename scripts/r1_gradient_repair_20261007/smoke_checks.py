"""Sequential real-model GPU0 normal and one-time-NaN smoke regressions."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/mosi_ablations_p0b_0919_gradient_repair_20261007'
WORKER=ROOT/'experiments/ablation_p0b_0919_gradient_repair_20261007/runner/worker.py'


def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    for name in ['normal','injected']:
        command=[sys.executable,str(WORKER),'--job','R1-04','--seed','2024','--stage','smoke',
            '--manifest',str(OUTPUT/'plan.json'),'--output',str(OUTPUT/'repair_checks'/name),
            '--checkpoints',str(OUTPUT/'repair_check_weights'/name)]
        if name=='injected':command.append('--inject-gradient-failure')
        print('SMOKE START',name,flush=True)
        with (OUTPUT/'logs'/f'repair_smoke_{name}.log').open('x') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if process.wait():raise RuntimeError(f'{name} smoke failed; inspect its log')
        report=json.loads((OUTPUT/'repair_checks'/name/'smoke.json').read_text())
        assert report['status']=='passed' and report['train']['skipped_batches']==0
        assert report['train']['reference_backward_retries']==(1 if name=='injected' else 0)
        print('SMOKE PASSED',name,'seconds',report['seconds'],'peak_MB',report['peak_gpu_mb'],flush=True)


if __name__=='__main__':main()
