import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from skip_policy import install
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/r1_aligned_20261009'))
from plan import BASE,read,write


def native():
    sys.path.insert(0,str(BASE/'source'))
    spec=importlib.util.spec_from_file_location('skip_test_native',BASE/'source/trainer.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.text_encoder=nn.Linear(4,4);self.proj_audio=nn.Linear(4,4);self.head=nn.Linear(4,1)
    def forward(self,text,audio,video,**kw):
        return dict(logits=self.head(self.text_encoder(text)+self.proj_audio(audio)+video))


class Tests(unittest.TestCase):
    def test_finite_path_exact(self):
        m=native();clip=torch.nn.utils.clip_grad_norm_
        args=SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
        logger=SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)
        a=Model();b=deepcopy(a);na=m.Trainer(deepcopy(args),a,logger)
        batches=[dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),labels={'M':torch.randn(4)}) for _ in range(4)]
        rng=torch.get_rng_state().clone();loss=na.train_one_epoch(batches,1);after=torch.get_rng_state().clone()
        try:
            with tempfile.TemporaryDirectory() as temp:
                install(m,temp,write);nb=m.Trainer(deepcopy(args),b,logger)
                torch.set_rng_state(rng);other=nb.train_one_epoch(batches,1)
                self.assertEqual(loss,other);self.assertTrue(torch.equal(after,torch.get_rng_state()))
                self.assertTrue(all(torch.equal(p,b.state_dict()[n]) for n,p in a.state_dict().items()))
                self.assertEqual(na.scheduler.state_dict(),nb.scheduler.state_dict())
                self.assertEqual(nb.numerical_guard.progress()['skipped_batches'],0)
        finally:torch.nn.utils.clip_grad_norm_=clip

    def test_backward_bad_batch_only_is_skipped(self):
        m=native();clip=torch.nn.utils.clip_grad_norm_
        args=SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
        model=Model();a=deepcopy(model);calls=[0]
        def bad_once(grad):
            calls[0]+=1
            return grad*float('nan') if calls[0]==1 else grad
        batches=[dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),labels={'M':torch.randn(4)}) for _ in range(3)]
        logger=SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)
        expected=m.Trainer(deepcopy(args),a,logger);expected.train_one_epoch(batches[1:],1)
        try:
            with tempfile.TemporaryDirectory() as temp:
                install(m,temp,write);obs=m.Trainer(deepcopy(args),model,logger)
                model.head.weight.register_hook(bad_once)
                obs.train_one_epoch(batches,1)
                self.assertEqual(obs.global_step,2)
                self.assertEqual(obs.scheduler.state_dict(),expected.scheduler.state_dict())
                self.assertTrue(all(torch.equal(p,model.state_dict()[n]) for n,p in a.state_dict().items()))
                self.assertEqual(read(Path(temp)/'skipped_batches.json')['count'],1)
        finally:torch.nn.utils.clip_grad_norm_=clip

    def test_forward_nan_skips_and_next_batch_trains(self):
        m=native();clip=torch.nn.utils.clip_grad_norm_
        args=SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
        model=Model();calls=[0]
        def bad_once(module,inputs,output):
            calls[0]+=1
            if calls[0]==1:return dict(logits=output['logits']*float('nan'))
            return output
        batches=[dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),labels={'M':torch.randn(4)}) for _ in range(2)]
        model.register_forward_hook(bad_once)
        try:
            with tempfile.TemporaryDirectory() as temp:
                install(m,temp,write);obs=m.Trainer(args,model,SimpleNamespace(info=lambda *a:None,warning=lambda *a:None))
                obs.train_one_epoch(batches,1)
                self.assertEqual(obs.global_step,1);self.assertEqual(obs.numerical_guard.progress()['skipped_batches'],1)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        finally:torch.nn.utils.clip_grad_norm_=clip


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assert not torch.cuda.is_initialized()
    raise SystemExit(0 if r.wasSuccessful() else 1)
