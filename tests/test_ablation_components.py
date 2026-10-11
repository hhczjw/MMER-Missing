"""CPU tests of actual adapter functions, without importing GPU kernels/models."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]


def actual_definitions(path,names):
    tree=ast.parse(path.read_text())
    selected=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert {x.name for x in selected}==set(names)
    scope=dict(torch=torch,F=F,nn=torch.nn)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return scope


class AblationComponents(unittest.TestCase):
    def test_right_padding_is_not_count_nonzero(self):
        scope=actual_definitions(ROOT/'scripts/ablations/r1_backend.py',['valid_frames'])
        x=torch.tensor([[[1.],[0.],[2.],[0.]]])
        self.assertTrue(torch.equal(scope['valid_frames'](x),torch.tensor([[True,True,True,False]])))

    def test_masked_pool_excludes_padding_denominator(self):
        scope=actual_definitions(ROOT/'scripts/ablations/r1_backend.py',['masked_align'])
        x=torch.tensor([[[2.],[0.],[0.],[0.]]]);mask=torch.tensor([[True,False,False,False]])
        value,valid=scope['masked_align'](x,mask,2)
        self.assertTrue(torch.equal(value,torch.tensor([[[2.],[0.]]])))
        self.assertTrue(torch.equal(valid,torch.tensor([[True,False]])))

    def test_rcd_keeps_student_gradient_and_detaches_teacher(self):
        scope=actual_definitions(ROOT/'scripts/ablations/r2_backend.py',['classification_relation'])
        torch.manual_seed(3)
        student=torch.randn(3,7,requires_grad=True);teacher=torch.randn(3,7,requires_grad=True)
        loss=scope['classification_relation'](student,teacher,torch.tensor([-.5,.5,9.]))
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()),0)
        self.assertIsNone(teacher.grad)

    def test_matched_relations_have_zero_loss(self):
        scope=actual_definitions(ROOT/'scripts/ablations/r2_backend.py',['classification_relation'])
        torch.manual_seed(3);x=torch.randn(3,7)
        loss=scope['classification_relation'](x,x,torch.tensor([-5.,0.,.5]))
        self.assertLess(abs(float(loss)),1e-6)

    def test_feature_reconstruction_disabled_without_disabling_contrastive(self):
        scope=actual_definitions(ROOT/'experiments/ablation_20261006/snapshot_r2/models/umrm_v2.py',
                                 ['MultiGranularityContrastiveReconstruction'])
        torch.manual_seed(3);module=scope['MultiGranularityContrastiveReconstruction'](4)
        fused=torch.randn(3,6,4,requires_grad=True);targets=[torch.randn(3,6,4) for _ in range(3)]
        tm=torch.tensor([[[1.,1.,0.,0.,1.,1.]]*3]*3);mm=torch.ones(3,3)
        enabled=module(fused,targets,tm,mm)
        rec=module.last_components['feature_reconstruction']
        module.ablation_reconstruction_weight=0.
        disabled=module(fused,targets,tm,mm)
        self.assertTrue(torch.allclose(enabled-disabled,rec,atol=1e-5))
        disabled.backward()
        self.assertGreater(float(module.token_proj[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or not bool((p.grad!=0).any()) for p in module.reconstruct_decoder.parameters()))


if __name__=='__main__':unittest.main()
