"""Read-only CPU audit of archived R2 best checkpoints versus native replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,re
import torch

ROOT=Path(__file__).resolve().parents[2]
EVID=ROOT/'experiments/research2_legacy_replay_20261009/historical_evidence'
NEW=ROOT/'results/research2_legacy_replay_20261009/run_01'
OLDTOKEN=ROOT/'results/token_input_retrain_20261006'
OUT=ROOT/'results/reproduction_20261009/r2_parameter_audit.json'
def read(p):return json.loads(p.read_text())
def diff(a,b):return {k:dict(old=a.get(k),new=b.get(k)) for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
def mean_missing(r):return {k:sum(r[c][k] for c in ['TA','TV','AV','T','A','V'])/6 for k in r['TA']}
def metrics(a,b):return {k:dict(old=a[k],new=b[k],change=b[k]-a[k]) for k in ['MAE','Acc2_nonzero','Acc2_has0','F1_nonzero','F1_has0','Acc7']}
report={'scope':'CPU metadata/log audit only; no evaluation or training', 'checkpoints':{},'metrics':{}}
paths={}
for kind in ['full','missing']:
    old=read(EVID/f'{kind}_ensemble.json');new=read(NEW/f'{kind}_ensemble.json')
    get=(lambda x:x['TAV']) if kind=='full' else mean_missing
    report['metrics'][kind]=metrics(get(old),get(new))
    report[kind+'_members']={'historical':old['checkpoints'],'replay':new['checkpoints']}
    for p in old['checkpoints']:paths[int(re.search(r'_seed(\d+)_',p).group(1))]=Path(p)
for seed,p in sorted(paths.items()):
    q=NEW/f'modality_seed{seed}/checkpoints'/p.name
    try:
        a=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(q,map_location='cpu',weights_only=False,mmap=True)
        aa=a['args'];bb=b['args']
        if not isinstance(aa,dict):aa=vars(aa)
        if not isinstance(bb,dict):bb=vars(bb)
        report['checkpoints'][str(seed)]={'historical_path':str(p),'new_path':str(q),
            'old_arg_count':len(aa),'new_arg_count':len(bb),'differences':diff(aa,bb),
            'state_names_shapes_equal':{k:list(v.shape) for k,v in a['model'].items()}=={k:list(v.shape) for k,v in b['model'].items()},
            'old_args':aa,'new_args':bb}
        del a,b
    except Exception as e:
        report['checkpoints'][str(seed)]={'historical_path':str(p),'error':str(e)}

a=read(OLDTOKEN/'seed45/config.json');b=read(NEW/'token/seed45/config.json')
ar=read(OLDTOKEN/'run.json');br=read(NEW/'token/run.json')
report['token']={'old_arg_count':len(a),'new_arg_count':len(b),'config_differences':diff(a,b),
    'run_differences':diff(ar,br),'old_args':a,'recorded_core_hashes_equal':ar['source_sha256']==br['source_sha256'],
    'current_frozen_core_matches':{name:hashlib.sha256((ROOT/'experiments/research2_legacy_replay_20261009/source'/name).read_bytes()).hexdigest()==value for name,value in ar['source_sha256'].items()}}
ah=read(OLDTOKEN/'seed45/history.json');bh=read(NEW/'token/seed45/history.json')
report['token']['history']={'old_epochs':len(ah),'new_epochs':len(bh),
    'first_old':ah[0],'first_new':bh[0],
    'rate_histogram_difference_epochs':[a['epoch'] for a,b in zip(ah,bh) if a['train']['rate_histogram']!=b['train']['rate_histogram']],
    'learning_rate_difference_epochs':[a['epoch'] for a,b in zip(ah,bh) if a['learning_rates']!=b['learning_rates']],
    'old_selection':read(OLDTOKEN/'seed45/selection.json'),'new_selection':read(NEW/'token/seed45/selection.json')}
report['metrics']['token']=metrics(read(OLDTOKEN/'seed45/test.json')['avg_over_rates'],read(NEW/'token/seed45/test.json')['avg_over_rates'])
report['environment']=read(NEW/'environment_at_launch.json')
report['historical_source_limits']=read(ROOT/'experiments/research2_legacy_replay_20261009/manifest.json')['unknowns']
multi=ROOT/'results/token_multiseed_42_47_20261009/run_01'
report['ongoing_multiseed']={'status':read(multi/'status.json'),
    'config_differences_vs_completed45':{s:v['difference_vs_completed45'] for s,v in read(multi/'configuration_check.json')['configs'].items()}}
report['cuda_initialized']=torch.cuda.is_initialized()
assert not report['cuda_initialized']
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
summary={k:v for k,v in report.items() if k in ['metrics','cuda_initialized']}
summary['checkpoints']={s:{k:v for k,v in r.items() if k not in ['old_args','new_args','historical_path','new_path']} for s,r in report['checkpoints'].items()}
summary['token']={k:v for k,v in report['token'].items() if k not in ['old_args','history']}
summary['history']={k:v for k,v in report['token']['history'].items() if k not in ['first_old','first_new','old_selection','new_selection']}
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(OUT)
