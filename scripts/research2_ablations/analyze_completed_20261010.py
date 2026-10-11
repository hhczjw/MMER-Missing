"""CPU-only audit and report of the frozen completed R2 cohort."""
import json
import statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/research2_ablations_gpu1_20261006_v3'
read = lambda p: json.loads(p.read_text())
rows = read(RUN / 'aggregate.json')['rows']
metrics = ['Acc2_has0', 'Acc2_nonzero', 'F1_has0', 'F1_nonzero', 'Acc7', 'MAE', 'Corr']
lines = ['# 研究二消融结果分析（2026-10-10）', '',
         '## 范围与口径', '',
         'MOSI：18 组模态轨道配置 × 3 seed（43、45、48），15 组 Token 轨道配置 × 3 seed（45、44、46），共 99 次训练。以下为单次指标的三 seed 算术均值，不是预测集成。', '',
         '模态缺失取 TA/TV/AV/T/A/V 六组合等权均值；完整模态是同一模态轨道模型的 TAV 测试；Token 取 0.0–0.9 十档等权均值（包含无缺失档）。所有测试样本数 686。', '',
         'Has0：全样本，标签和预测以 >=0 二分；Non0：排除标签为零的样本，以 >0 二分；F1 为 weighted F1；Acc7 为 numpy.round(numpy.clip(x,-3,3))，包含零标签。百分制指标的差值单位为百分点。', '',
         'M 以验证集完整模态 MAE 选权重，T 以验证集 Token10 Acc7 选权重。两轨道不共用对照；不能用历史最佳单模型或集成值替换本批消融对照。']
count = 0
identity = None
details = {}
for row in rows:
    track = row['variant'][:4]
    baseline = next(r for r in rows if r['variant'] == track + '-00')
    seed_details = []
    for member in row['members']:
        d = RUN / 'train' / f"{row['variant']}_seed{member['seed']}"
        t = read(d / 'test.json')
        c = read(d / 'config.json')
        base_c = read(RUN / 'train' / f"{track}-00_seed{member['seed']}" / 'config.json')
        allowed = {'kd_weight', 'rcd_weight', 'text_recon_weight', 'uni_weight', 'gap_weight', 'use_cd_kd'}
        differences = {k: (base_c['args'].get(k), v) for k, v in c['args'].items() if v != base_c['args'].get(k)}
        assert set(differences) <= allowed, (d.name, differences)
        z = np.load(d / 'test.predictions.npz')
        y = z['labels']
        assert len(y) == t['sample_count'] == 686
        if identity is None:
            identity = (y.copy(), z['sample_ids'].copy())
        assert np.array_equal(y, identity[0]) and np.array_equal(z['sample_ids'], identity[1])
        scenes = t['per_combo'] if track == 'R2-M' else t['per_rate']
        for scene, saved in scenes.items():
            pred = z[(scene if track == 'R2-M' else 'r' + scene) + '_preds']
            assert np.isfinite(pred).all()
            mask = y != 0
            actual = dict(Acc2_has0=np.mean((pred >= 0) == (y >= 0)),
                          Acc2_nonzero=np.mean((pred[mask] > 0) == (y[mask] > 0)),
                          F1_has0=f1_score(y >= 0, pred >= 0, average='weighted', zero_division=0),
                          F1_nonzero=f1_score(y[mask] > 0, pred[mask] > 0, average='weighted', zero_division=0),
                          Acc7=np.mean(np.round(np.clip(pred, -3, 3)) == np.round(np.clip(y, -3, 3))),
                          MAE=float(np.mean(np.abs(pred-y))))
            for k, v in actual.items():
                assert abs(v - saved[k]) < 1e-6, (d.name, scene, k)
        summary = t['missing_avg'] if track == 'R2-M' else t['avg_over_rates']
        for k in metrics:
            assert abs(summary[k] - member['metrics'][k]) < 1e-10
        base_m = next(m for m in baseline['members'] if m['seed'] == member['seed'])
        seed_details.append(dict(seed=member['seed'], selected_epoch=t['selected_epoch'], epochs_run=t['epochs_run'],
                                 acc7=summary['Acc7'], mae=summary['MAE'],
                                 delta_acc7=summary['Acc7'] - base_m['metrics']['Acc7'], args_differences=differences))
        count += 1
    for k in metrics:
        assert abs(statistics.mean(m['metrics'][k] for m in row['members']) - row['mean'][k]) < 1e-10
    details[row['variant']] = seed_details

lines += ['', '核验：99 份测试预测均为有限值，样本标签与顺序一致；逐场景重算 Acc2 Has0/Non0、F1 Has0/Non0、Acc7、MAE 与保存结果一致；三 seed 汇总一致。配置 args 差异仅出现在设计中的损失开关，结构干预另由 changes/hooks 实施。此核验不等于已证明历史训练可严格复现。']
for title, subset, full in [('完整模态 TAV（模态轨道）', rows[:18], True),
                             ('整模态缺失（六组合平均）', rows[:18], False),
                             ('Token 缺失（十档平均）', rows[18:], False)]:
    lines += ['', '## ' + title, '', '| 配置 | Acc2 Has0 | Acc2 Non0 | F1 Has0 | F1 Non0 | Acc7 | MAE | Corr | Acc7 Δ |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    base = subset[0]['full_mean' if full else 'mean']
    for r in subset:
        v = r['full_mean' if full else 'mean']
        values = [f'{v[k]*100:.2f}' if k not in ['MAE', 'Corr'] else f'{v[k]:.4f}' for k in metrics]
        lines.append('| ' + r['variant'] + ' ' + r['title'] + ' | ' + ' | '.join(values) + f" | {(v['Acc7']-base['Acc7'])*100:+.2f} |")
lines += ['', '## 三 seed 波动与训练选择', '', '| 配置 | Acc7 均值 ± 样本标准差 | 删除后 Acc7 提高的 seed 数 | 各 seed 最佳轮/实际轮数 |', '|---|---:|---:|---|']
for r in rows:
    ds = details[r['variant']]
    lines.append(f"| {r['variant']} | {r['mean']['Acc7']*100:.2f} ± {r['sample_std']['Acc7']*100:.2f} | {sum(x['delta_acc7']>0 for x in ds)}/3 | " + '; '.join(f"{x['seed']}: {x['selected_epoch']}/{x['epochs_run']}" for x in ds) + ' |')
lines += ['', '## 解释与结论', '',
 '1. 整模态缺失：Enhance、联合 KD/RCD 的平均贡献最大，但需要同时说明稳定性。关闭 Enhance 的三个 seed Acc7 均降低，其中 seed48 降幅最大；联合关闭 KD/RCD 是 2/3 seed 降低，seed45 明显退化，因此不能把平均降幅全部解读为稳定的模块贡献，更不能仅凭组合删除幅度证明严格协同效应。', '',
 '2. 完整模态：本批 M-00 的 Acc7=50.83、MAE=0.5853。所有结构/损失/策略变体的平均 Acc7 均低于对照，MAE 均高于对照；关闭 Enhance 和联合蒸馏退化明显。它是模态轨道的 TAV 评估，不是研究一模型的消融。', '',
 '3. 重建存在任务及指标权衡：M-07 关闭文本重建后，Acc7 从33.83降至32.60，但 Acc2 Non0 从70.66升至71.71，MAE 从1.0339降至1.0230。不能写成文本重建改善全部指标。M-01 关闭 UME 后缺失平均改善，TAV 反而下降；当前结果不支持 UME 对所有场景普遍有效。', '',
 '4. Token 对照训练质量是首要疑点：三个 seed 均在第2轮取得最佳验证 Acc7，第20轮早停；最佳轮还处于 warmup 且训练缺失率上限仅0.5。随后上限在第4轮达到0.9，学习率第5轮达到3e-4，验证指标未恢复。训练难度与学习率同时上升，是需要隔离验证的机制假设，不是已经确定的根因。不能认定只要延长训练就必然改善。', '',
 '5. Token 的14种删除/策略变体平均 Acc7 全部高于对照。去文本重建 T-07 三 seed 为37.04/35.50/35.19，均高于对应对照25.07/31.66/28.41，不能仅归因于单个坏 seed。联合去 KD/RCD、去 gap 也有一致方向的 Acc7 改善。当前完整损失组合与训练方案存在适配/优化问题的证据较强，但这些结果尚不足以证明相关模块在所有合理训练设置下无效。', '',
 '6. Token 当前最高平均 Acc7 是 T-07（35.91）；最高 Acc2 Non0 是 T-12（74.23）；最低 MAE 是 T-09（0.9924）。三个来自不同配置，不能拼成同一模型。这里只是描述测试观察，不可据此测试集选参并宣称无偏新最佳。', '',
 '7. 口径边界：M 以 TAV 验证 MAE 选模，不是按缺失六组合 MAE 选模；T 以 Token10 验证 Acc7 选模。两者本就会产生不同指标倾向。三 seed 规模较小，本文不宣称统计显著。部分删除还改变优化路径，消融测量的是本训练方案下的整体效应。', '',
 '## 建议与汇报方式', '',
 '- 可汇报整模态缺失与 TAV 的本批同条件消融，保留三 seed 均值/标准差，优先展示完整对照、去 Enhance、去重建、去 KD、去 RCD、联合去蒸馏。不要只挑能支持创新点的指标。',
 '- Token 暂按“本轮控制实验观察”呈现，不能用历史最佳补对照、掩盖当前完整模型偏弱。优先核对当前与历史原生训练循环、学习率调度状态、随机数路径及训练记录；尤其核对早期过峰与高缺失阶段退化。',
 '- 后续若要调整：在验证集上预先固定对照复验方案，分别隔离学习率峰值、缺失率课程速度、文本重建权重/联合蒸馏权重；先恢复稳定对照，再给受影响消融使用同一训练预算和选模规则。不能只延长对照而保留其他组短训结果。',
 '- 本次仅做 CPU 结果审查，没有启动训练、改动训练代码或修改 PPT。', '',
 '数据源：`results/research2_ablations_gpu1_20261006_v3/aggregate.json`、`train/*/{config,history,selection,test}.json`、`test.predictions.npz`；机制定义：`scripts/research2_ablations/{runtime,worker,smoke_gpu1,plan}.py`。']
out = ROOT / 'docs/研究二消融结果分析_20261010.md'
out.write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(audited_jobs=count, report=str(out), token_baseline=details['R2-T-00']), ensure_ascii=False, indent=2))
