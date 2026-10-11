"""Acc7 must match the full-modal project's clip+round convention."""

import numpy as np

from utils.metrics import eval_regression


def test_acc7_clip_round_and_legacy_bins_are_separate():
    labels = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    preds = np.array([-2.6, -0.6, 0.6, 1.4, 3.4])
    result = eval_regression(preds, labels)
    expected = np.mean(np.round(np.clip(preds, -3, 3)) == np.round(np.clip(labels, -3, 3)))
    assert result['Acc7'] == expected
    assert result['Acc7'] != result['Acc7_bins']


def test_acc7_uses_numpy_ties_to_even():
    result = eval_regression(np.array([0.5, 1.5, -0.5]), np.array([0.0, 2.0, 0.0]))
    assert result['Acc7'] == 1.0
