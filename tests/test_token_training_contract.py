"""CPU regressions for fractional availability and canonical Acc7 labels."""
import unittest
import numpy as np
import torch


class TokenTrainingContract(unittest.TestCase):
    def test_round_before_class_offset(self):
        labels = torch.tensor([-4., -2.5, -1.5, -.5, .5, 1.5, 2.5, 4.])
        actual = (labels.clamp(-3, 3).round() + 3).long().numpy()
        expected = np.round(np.clip(labels.numpy(), -3, 3)).astype(int) + 3
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, (labels + 3).round().clamp(0, 6).numpy()))

    def test_fractional_missing_weights(self):
        mask = torch.tensor([[1., .75, .25, 0.]])
        torch.testing.assert_close((1-mask).clamp(0,1), torch.tensor([[0., .25, .75, 1.]]))
        self.assertEqual((mask >= 1).tolist(), [[True,False,False,False]])


if __name__ == '__main__':
    unittest.main()
