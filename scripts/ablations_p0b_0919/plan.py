"""Predeclared MOSI ablations. No model imports, test-based ranking or HPO."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/ablation_p0b_0919_20261006'
SEEDS = [2024, 42, 1111]
R2_SEEDS = [42, 44, 45]
RATES = [i / 10 for i in range(10)]
METRICS = ['Acc2_has0', 'Acc2_nonzero', 'F1_has0', 'F1_nonzero', 'Acc7', 'MAE', 'Corr']

R1 = [
    ('00', '完整参考', {}),
    ('01', '去掉跨模态状态写入', {'cell_source_mode': 'self', 'uniform_routing': True}),
    ('02', '等权源路由', {'uniform_routing': True}),
    ('03', '去掉GLCE', {'global_glce': False, 'local_glce': False}),
    ('04', '去掉GCMN', {'gcmn': False}),
    ('05', '去掉BSSM门控', {'bssm_gate': False}),
    ('06', '去掉对话上下文', {'use_context': False}),
    ('07', '去掉GLCE全局分支', {'global_glce': False}),
    ('08', '去掉GLCE局部分支', {'local_glce': False}),
    ('09', 'K候选仅来自目标模态', {'cell_source_mode': 'key_self'}),
    ('10', 'V仅来自目标模态', {'cell_source_mode': 'value_self'}),
]
R2 = [
    ('00', '完整参考', {}),
    ('01', '去掉UME', {'ume': False}),
    ('02', '去掉Enhance门控残差', {'enhance': False}),
    ('03', '去掉全部重建监督', {'text_rec_weight': 0.0, 'feature_rec_weight': 0.0}),
    ('04', '去掉预测KD保留RCD', {'kd_weight': 0.0}),
    ('05', '去掉RCD保留预测KD', {'rcd_weight': 0.0}),
    ('06', '去掉增强特征旁路', {'enhanced_bypass': False}),
    ('07', '仅去掉文本重建', {'text_rec_weight': 0.0}),
    ('08', '仅去掉基础特征重建', {'feature_rec_weight': 0.0}),
    ('09', '去掉预测KD和RCD', {'kd_weight': 0.0, 'rcd_weight': 0.0}),
    ('10', '去掉文本相似度注入', {'text_injection': False}),
    ('11', '去掉反向融合', {'reverse_fusion': False}),
    ('12', '仅去掉gap均衡', {'gap_weight': 0.0}),
]


def variants():
    rows = []
    for track, settings in [('R1', R1), ('R2-M', R2), ('R2-T', R2)]:
        for i, title, changes in settings:
            rows.append(dict(id=f'{track}-{i}', track=track, title=title,
                             changes=deepcopy(changes), batch='core' if int(i) <= 6 else 'mechanism'))
    return rows


def jobs(seeds=SEEDS):
    # Core rows across all tracks first, then mechanism decomposition; repeated for every seed.
    settings = variants()
    return [dict(v, seed=(s if v['track']=='R1' else R2_SEEDS[i]),
                 job_id=f'{v["id"]}_seed{s if v["track"]=="R1" else R2_SEEDS[i]}')
            for i,s in enumerate(seeds) for batch in ['core','mechanism']
            for v in settings if v['batch']==batch]


def selector(track, metrics):
    if track == 'R1':
        return (metrics['Acc2_nonzero'],)
    if track == 'R2-M':
        return (-metrics['MAE'], metrics['Acc7'])
    if track == 'R2-T':
        return (metrics['Acc7'], -metrics['MAE'])
    raise ValueError(track)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def sources():
    files = list((ROOT / 'scripts/ablations_p0b_0919').glob('*.py'))
    for name in ['snapshot_r1', 'snapshot_r2']:
        # Do not traverse the shared, read-only Mamba link or caches.
        files += [p for p in (EXPERIMENT/name).rglob('*.py') if 'mamba' not in p.relative_to(EXPERIMENT/name).parts]
        files += list((EXPERIMENT/name/'configs').glob('*.json'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def assert_frozen(manifest):
    for relative, expected in manifest['source_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise RuntimeError(f'Frozen experiment source changed: {relative}')


def reference_arguments():
    origin=Path('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3')
    log=(origin/'logs/MSA_MOSI_20260918_233755.log').read_text()
    r1,_=json.JSONDecoder().raw_decode(log[log.index('Args: ')+6:])
    for key in ['feature_A','feature_V']:r1[key]=str(origin/r1[key])
    r2=json.loads((ROOT/'results/token_input_retrain_20261006/seed45/config.json').read_text())
    return dict(R1=r1,R2=r2)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)
