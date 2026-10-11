"""
评估指标
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def eval_regression(preds, labels):
    """回归评估。

    二分类统一口径：
    - Non0：剔除真实标签为 0 的样本，再按正/负号计算；作为 Acc2/F1 主口径。
    - Has0：保留全部样本，真实值和预测值均以 >= 0 判为非负类。
    Acc7、MAE 和 Corr 始终使用全部样本。Acc7 主口径与全模态项目、
    LNLN/TF-Mamba/MPLMM 一致：先 clip 到 [-3, 3]，再使用 np.round。
    旧区间分箱结果保留为 Acc7_bins，不能与 Acc7 混用。

    ``Acc2`` / ``F1`` 保留为向后兼容别名，均指向 Non0。
    """
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    mae = mean_absolute_error(labels, preds)
    corr = np.corrcoef(preds, labels)[0, 1] if len(preds) > 1 else 0.0

    # Acc-2 / F1：同时报告 Has0 与 Non0；论文主口径使用 Non0。
    has0_true = labels >= 0
    has0_pred = preds >= 0
    acc2_has0 = accuracy_score(has0_true, has0_pred)
    f1_has0 = f1_score(has0_true, has0_pred, average='weighted', zero_division=0)

    nonzero = labels != 0
    if np.any(nonzero):
        non0_true = labels[nonzero] > 0
        non0_pred = preds[nonzero] > 0
        acc2_nonzero = accuracy_score(non0_true, non0_pred)
        f1_nonzero = f1_score(non0_true, non0_pred, average='weighted', zero_division=0)
    else:
        acc2_nonzero = 0.0
        f1_nonzero = 0.0

    # Acc-7: 论文常用七分类口径，clip 后按 NumPy 的 round 规则取整。
    acc7 = accuracy_score(
        np.round(np.clip(labels, -3.0, 3.0)),
        np.round(np.clip(preds, -3.0, 3.0)),
    )

    # 兼容已有实验：旧整数区间分箱，单独报告，绝不作为 Acc7 别名。
    def to_7class(x):
        if x < -2: return 0
        elif x < -1: return 1
        elif x < 0: return 2
        elif x < 1: return 3
        elif x < 2: return 4
        elif x < 3: return 5
        else: return 6
    pred7 = [to_7class(p) for p in preds]
    label7 = [to_7class(l) for l in labels]
    acc7_bins = accuracy_score(label7, pred7)

    return {
        'MAE': mae,
        'Corr': corr,
        'Acc2': acc2_nonzero,
        'F1': f1_nonzero,
        'Acc2_nonzero': acc2_nonzero,
        'Acc2_has0': acc2_has0,
        'F1_nonzero': f1_nonzero,
        'F1_has0': f1_has0,
        'Acc7': acc7,
        'Acc7_bins': acc7_bins,
    }


def eval_classification(preds, labels, num_classes=4):
    """分类评估：Accuracy, F1"""
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted' if num_classes > 2 else 'binary')

    return {'Accuracy': acc, 'F1': f1}
