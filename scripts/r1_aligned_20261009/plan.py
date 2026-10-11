"""Prespecified R1 native-pipeline ablations, never selected by test results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/p0B_legacy_replay_20261008'
OUTPUT = ROOT / 'results/r1_native_aligned_ablations_20261009/run_01'
TOKEN = ROOT / 'results/token_multiseed_42_47_20261009/run_01'
PYTHON = '/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python'
SEEDS = [2024, 42, 1111]
SETTINGS = [
    ('00', '完整同管路对照', {}),
    ('01', 'K/V不再读取其他模态，源路由等权', {'source_mode': 'self', 'uniform': True}),
    ('02', '等权源路由', {'uniform': True}),
    ('03', '去掉GLCE全局和局部提取，保留shortcut与第二归一化', {'global': False, 'local': False}),
    ('04', '去掉GCMN', {'gcmn': False}),
    ('05', '去掉BSSM门控', {'bssm': False}),
    ('06', '去掉对话上下文', {'context': False}),
    ('07', '去掉GLCE全局分支', {'global': False}),
    ('08', '去掉GLCE局部分支', {'local': False}),
    ('09', 'K候选仅读取目标模态', {'source_mode': 'key_self'}),
    ('10', 'V采用目标投影x_default', {'source_mode': 'value_self'}),
]
REPORT_KEY = 'secondary_Acc2'
AUTHORITY = '2026-10-09 human: 将研究内容一的消融实验配置与最优的配置进行对齐，并且进行重新消融实验训练。挂载到GPU1下面。'


def jobs():
    return [dict(id='R1-'+i, job_id=f'R1-{i}_seed{seed}', title=title,
                 seed=seed, changes=changes) for seed in SEEDS for i,title,changes in SETTINGS]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def config_for(args, output, changes):
    for name,folder in [('checkpoints_dir','checkpoints'),('logs_dir','logs'),('results_dir','results')]:
        setattr(args,name,str(output/folder))
    if changes.get('context') is False:
        args.use_context = False
    return args


def check_config(actual, reference, changes):
    allowed = {'checkpoints_dir','logs_dir','results_dir'}
    if changes.get('context') is False:
        allowed.add('use_context')
    diffs = {k:dict(reference=reference.get(k), actual=actual.get(k))
             for k in set(reference)|set(actual) if reference.get(k)!=actual.get(k)}
    if set(diffs)-allowed:
        raise RuntimeError('Unexpected native config differences: '+str(diffs))
    if changes.get('context') is False and actual.get('use_context') is not False:
        raise RuntimeError('Context intervention was not applied')
    return diffs
