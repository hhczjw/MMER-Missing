import unittest
import torch
from utils.token_protocol import keep_mask, validity, native_corruption, feature_corruption_masks, aligned_availability


class TokenProtocolTests(unittest.TestCase):
    def test_padding_and_actual_sep_are_preserved(self):
        valid = torch.tensor([[1,1,1,1,0,0],[0,0,0,0,0,0]],dtype=torch.bool)
        keep, eligible = keep_mask(valid,1.,protect_edges=True)
        self.assertEqual(keep[0].tolist(),[1,0,0,1,1,1])
        self.assertTrue(bool(keep[1].all()))
        self.assertEqual(int(eligible.sum()),2)

    def test_native_missing_uses_true_av_lengths(self):
        t=torch.tensor([[[0,12,13,2,1,1],[1,1,1,1,0,0],[0,0,0,0,0,0]]])
        a=torch.ones(1,8,2);v=torch.ones(1,10,3)
        valid=validity(t,a,v,torch.tensor([3]),torch.tensor([4]))
        generator=torch.Generator().manual_seed(8)
        (tt,aa,vv),keeps,_=native_corruption(t,a,v,valid,1.,generator,3)
        self.assertEqual(tt[0,0].tolist(),[0,3,3,2,1,1])
        self.assertEqual(tt[0,1].tolist(),t[0,1].tolist())
        self.assertFalse(bool(aa[0,:3].any()))
        self.assertTrue(bool(aa[0,3:].all()))
        mask,_=feature_corruption_masks(valid,6,0.,torch.Generator().manual_seed(8))
        self.assertTrue(bool(mask.all()))

    def test_fixed_masks_ignore_model_rng(self):
        valid=torch.ones(2,50,dtype=torch.bool)
        first,_=keep_mask(valid,.5,torch.Generator().manual_seed(123),True)
        torch.rand(3000)
        second,_=keep_mask(valid,.5,torch.Generator().manual_seed(123),True)
        self.assertTrue(torch.equal(first,second))

    def test_pooling_does_not_count_padding_as_observed(self):
        valid=torch.tensor([[1,1,1,0,0,0]],dtype=torch.bool)
        keep=torch.tensor([[1,0,0,1,1,1]],dtype=torch.float32)
        result=aligned_availability([keep]*3,2,[valid]*3)
        self.assertAlmostEqual(float(result[0,0,0]),1/3,places=6)
        self.assertEqual(float(result[0,0,1]),1.)


if __name__=='__main__':
    unittest.main()
