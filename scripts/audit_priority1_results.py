"""Verify archived MOSI predictions, rebuild PPT snapshot and audit report."""
import json
import argparse
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.metrics import eval_regression

BASE=ROOT/'results/priority1_20261006/baselines'
PAIR=ROOT/'results/priority1_20261006/token_protocol_paired_test_v2.json'
FIELDS={'mae':'MAE','corr':'Corr','acc2_has0':'Acc2_has0','acc2_nonzero':'Acc2_nonzero',
        'f1_has0':'F1_has0','f1_nonzero':'F1_nonzero','acc7':'Acc7'}


def ppt_row(metrics,model='MMER-Missing',upper=False):
    return dict(model=model,venue='本研究',**{k:round(float(metrics[v if upper else k])*(1 if k in ('mae','corr') else 100),4 if k in ('mae','corr') else 2)
                                           for k,v in FIELDS.items() if (v if upper else k) in metrics})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    cli=parser.parse_args()
    snapshot_path=ROOT/'docs/bjtu_ppt_metrics_20261005.json'
    current=json.loads(snapshot_path.read_text())
    audited={}
    for path in BASE.glob('*_clipround.json'):
        d=json.loads(path.read_text())
        archive=np.load(d['metric_protocol']['predictions'],allow_pickle=False)
        section='conditions' if 'conditions' in d else 'rates'
        for key,value in d[section].items():
            preds,labels=archive[f'{key}_preds'],archive[f'{key}_labels']
            assert len(preds)==len(labels)==686,(path,key,len(preds))
            measured=eval_regression(preds,labels)
            for lower,upper in FIELDS.items():
                assert abs(value[lower]-measured[upper])<1e-10,(path,key,lower)
        audited[(d['model'],section)]=d
    assert len(audited)==8,f'Expected 7 baselines and LNLN Token, got {list(audited)}'
    for name,section in [('full_reproduced','TAV'),('missing_reproduced','missing_avg')]:
        for old in current[name]:
            d=audited[(old['model'],'conditions')]
            metrics=d['conditions'][section] if section=='TAV' else d[section]
            old.update({k:v for k,v in ppt_row(metrics).items() if k not in ('model','venue')})
    current['token_reproduced'][0].update({k:v for k,v in ppt_row(audited[('LNLN','rates')]['avg_over_rates']).items() if k not in ('model','venue')})
    pair=json.loads(PAIR.read_text())
    assert pair['n_samples']==686 and not pair['smoke_only']
    archive=np.load(pair['prediction_archive'],allow_pickle=False)
    for name,value in pair['protocols'].items():
        for rate,metrics in value['per_rate'].items():
            measured=eval_regression(archive[f'ensemble_{name}_r{rate}_preds'],archive['labels'])
            for key in FIELDS.values():
                assert abs(metrics[key]-measured[key])<1e-10,(name,rate,key)
    legacy=json.loads((ROOT/'results/NEW3_token_s44s45s46_Acc7ckpt_fixedmask_clipround.json').read_text())
    for key in FIELDS.values():
        assert abs(legacy['avg_over_rates'][key]-pair['protocols']['legacy_input']['avg_over_rates'][key])<0.00011,key
    current['mmer_token']=ppt_row(pair['protocols']['input_valid_aware']['avg_over_rates'],upper=True)
    single=next(m for m in pair['members'] if re.search(r'_seed44_',m['checkpoint']))
    current['mmer_token_single']=ppt_row(single['protocols']['input_valid_aware']['avg_over_rates'],upper=True)
    current['mmer_token_feature_valid']=ppt_row(pair['protocols']['feature_valid']['avg_over_rates'],upper=True)
    current['mmer_token_legacy']=ppt_row(pair['protocols']['legacy_input']['avg_over_rates'],upper=True)
    current['generated_from'].update(baseline_root=str(BASE),mmer_token=str(PAIR)+'; protocols.input_valid_aware; frozen seed44/45/46',
                                    mmer_token_single=str(PAIR)+'; seed44 input_valid_aware')
    current['protocol']['token']='输入级有效位置缺失；r=0.0–0.9；实际 CLS/SEP 保留，padding 排除；真实可用率 mask 传递到 UME/Enhance；固定 seed20261005'
    current['protocol']['acc7']='所有复现主列由同一个 utils/metrics.py 计算 clip[-3,3]+numpy.round；论文原值未逐篇重新计算'
    current['notes']=[n for n in current['notes'] if not n.startswith('Token 主表')]+[
        'Token 主表采用修正输入级有效长度和 mask 传递的固定权重重评；后 ISM 特征缺失仅在协议附录展示，不与输入级论文值混排。',
        '七个基线的完整/六组合结果与 LNLN 十率结果已重评 clip+round，并保存逐样本预测；原生特征与缺失位置仍分别标注。',
        '当前冻结 seed44/45/46 权重用于配对诊断；正式验证集选模与新训练策略收益属于后续工作。']
    if cli.check_only:
        saved=json.loads(snapshot_path.read_text())
        for section in ['full_reproduced','missing_reproduced','token_reproduced']:
            for expected,actual in zip(current[section],saved[section]):
                for key in FIELDS:
                    if key in expected:
                        assert expected[key]==actual.get(key),(section,expected['model'],key)
        for key in ['mmer_token','mmer_token_single','mmer_token_feature_valid']:
            assert current[key]==saved[key],key
        print('AUDIT PASSED: 8 baseline evaluations, paired predictions, legacy control, and PPT snapshot')
        return
    snapshot_path.write_text(json.dumps(current,ensure_ascii=False,indent=2))
    ledger_path=ROOT/'docs/bjtu_reproduced_uniform_metrics.json'
    old_ledger=json.loads(ledger_path.read_text())
    legacy_path=ROOT/'docs/bjtu_reproduced_uniform_metrics_20261004_legacy_bins.json'
    if 'interval bins' in str(old_ledger.get('protocol',{}).get('acc7','')) and not legacy_path.exists():
        legacy_path.write_text(json.dumps(old_ledger,ensure_ascii=False,indent=2))
    ledger={'dataset':'CMU-MOSI','protocol':current['protocol'],'missing_six':[], 'token_ten':[],
            'audit':'All current fields recomputed from archived sample predictions, 2026-10-06'}
    for (model,section),d in audited.items():
        metric=d['missing_avg' if section=='conditions' else 'avg_over_rates']
        row=ppt_row(metric,model)
        row.pop('venue',None)
        row.update(source=str(BASE/f'{model}_{"missing7" if section=="conditions" else "token"}_clipround.json'),
                   predictions=d['metric_protocol']['predictions'],checkpoint=d['ckpt'])
        ledger['missing_six' if section=='conditions' else 'token_ten'].append(row)
    ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
    lines=['# MOSI 优先级 1 核验记录（2026-10-06）','',
           '已完成七个基线 Acc7 clip+round 和 Acc2/F1 双口径重评，LNLN Token 十档重评，MMER 固定三权重的四协议配对评测。所有原始指标由存档预测重新计算核查；686 个测试样本，无 checkpoint 重训。','',
           '| 基线 | 完整 Acc7 | 缺失六组合 Acc7 | F1 Has0 | F1 Non0 |','|---|---:|---:|---:|---:|']
    for (model,section),d in audited.items():
        if section=='conditions':
            avg=d['missing_avg']
            lines.append(f"| {model} | {100*d['conditions']['TAV']['acc7']:.2f} | {100*avg['acc7']:.2f} | {100*avg['f1_has0']:.2f} | {100*avg['f1_nonzero']:.2f} |")
    lines+=['','F1 两列为六种缺失组合平均；所有复现 Acc7 为 clip+round。论文报告值继续保留原口径，不反推。','',
            '| MMER 三模型固定权重协议 | MAE | Acc2 Non0 | F1 Non0 | Acc7 |','|---|---:|---:|---:|---:|']
    for name,d in pair['protocols'].items():
        m=d['avg_over_rates']
        lines.append(f"| {name} | {m['MAE']:.4f} | {100*m['Acc2_nonzero']:.2f} | {100*m['F1_nonzero']:.2f} | {100*m['Acc7']:.2f} |")
    lines+=['','协议含义：',
            '- legacy_input：复现旧脚本，含 padding/特殊 token 问题，仅作控制。',
            '- input_valid_blind：只删除实际有效位置，正确保护 CLS/SEP，内部 mask 仍全 1。',
            '- input_valid_aware：与上一行相同损坏输入，将实际可用率投影到 50 个特征位置并显式传递；分母排除 padding。原始 A/V 帧聚合后的可用率可以是小数，主模型会再次将该 mask 施加到 ISM 后表示。',
            '- feature_valid：在 ISM 后对有效位置作二值缺失，与训练阶段一致；编码前看过完整输入，不能作为原始输入级缺失论文对比。','',
            '旧协议平均指标已与原 fixedmask JSON 交叉核对（差异仅为原先逐率四位舍入）。各协议 r=0 分类结果一致；跨 CUDA 重评的预测最大偏差约 8.87e-5，在 1e-4 浮点容差内。padding、实际 SEP 及有效帧可用率分母测试通过；一个真实 LoRA/Mamba 训练批次的梯度、参数更新、有效长度和三档评测冒烟通过。','',
            '训练入口现在在特征层缺失时排除 padding，并接收数据集真实 A/V 长度。该修改没有追溯改变历史权重。下一阶段仍需冻结验证集选模规则后正式训练/消融；若研究目标为输入级缺失，还需进行相应输入级训练对照。','',
            f'配对结果与预测：`{PAIR}`，`{pair["prediction_archive"]}`。基线存档：`{BASE}`。']
    report=ROOT/'docs/MOSI_优先级1核验_20261006.md'
    report.write_text('\n'.join(lines)+'\n')
    print('AUDIT PASSED; updated',snapshot_path,'and',report)


if __name__=='__main__':main()
