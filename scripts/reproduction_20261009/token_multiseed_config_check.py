"""CPU-only proof: all six effective configs differ from completed45 only by seed."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/research2_legacy_replay_20261009'
sys.path[:0]=[str(BASE),str(BASE/'source/scripts'),str(BASE/'source')]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);cli=ap.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
spec=importlib.util.spec_from_file_location('checked_native_token',BASE/'source/scripts/train_token_input.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
previous=json.loads((ROOT/'results/research2_legacy_replay_20261009/run_01/token/seed45/config.json').read_text())
args=SimpleNamespace(epochs=60,batch_size=8,eval_batch_size=16,lr=.0003,min_epochs=20,patience=15)
configs={}
for seed in [42,43,44,45,46,47]:
    now=vars(module.arguments(args,seed))
    delta={k:[previous.get(k),now.get(k)] for k in set(previous)|set(now) if previous.get(k)!=now.get(k)}
    assert set(delta)<=set(['seed']),delta
    configs[str(seed)]={'difference_vs_completed45':delta,'effective_config':now}
assert not module.torch.cuda.is_initialized()
report={'passed':True,'cuda_initialized':False,'configs':configs,'native_source_modified':False}
with cli.output.open('x') as stream:json.dump(report,stream,indent=2)
print('CPU_CONFIG_CHECK_PASSED: seed is the only effective configuration difference',flush=True)
