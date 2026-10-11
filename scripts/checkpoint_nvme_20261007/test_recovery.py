import builtins
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from data_cache import install_cache
from training_state import save_training_state, restore_training_state


class RecoveryTests(unittest.TestCase):
    def test_full_state_resume_next_update_identical(self):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.5),torch.nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1./(step+1))
        def step(model,optimizer,scheduler):
            optimizer.zero_grad(set_to_none=True)
            x=torch.randn(5,3);y=torch.randn(5,1)
            loss=(model(x)-y).square().mean();loss.backward();optimizer.step();scheduler.step()
            return loss.detach().clone(),random.random(),float(np.random.random())
        step(model,optimizer,scheduler)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'resume.pth'
            save_training_state(file,model,optimizer,scheduler,torch.device('cpu'),epoch=1,best={'epoch':1},stale=0)
            expected=step(model,optimizer,scheduler)
            expected_parameters=[p.detach().clone() for p in model.parameters()]
            clone=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.5),torch.nn.Linear(4,1))
            opt=torch.optim.AdamW(clone.parameters(),lr=.001,foreach=False)
            schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1./(step+1))
            state=torch.load(file,map_location='cpu',weights_only=False)
            restore_training_state(state,clone,opt,schedule,torch.device('cpu'))
            actual=step(clone,opt,schedule)
            self.assertTrue(torch.equal(expected[0],actual[0]))
            self.assertEqual(expected[1:],actual[1:])
            for before,after in zip(expected_parameters,clone.parameters()):self.assertTrue(torch.equal(before,after))
            self.assertEqual(scheduler.state_dict(),schedule.state_dict())

    def test_dataset_redirect_is_read_only(self):
        original_open,original_path_open=builtins.open,Path.open
        original_exists,original_isfile=os.path.exists,os.path.isfile
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source.pkl';local=root/'cache.pkl'
                source.write_bytes(b'original');local.write_bytes(b'verified copy')
                index=root/'index.json';index.write_text(json.dumps({'phase':'ready','mapping':{str(source):{'local':str(local)}}}))
                self.assertTrue(install_cache(index))
                self.assertEqual(source.read_bytes(),b'verified copy')
                with open(source,'rb') as f:self.assertEqual(f.read(),b'verified copy')
                with open(source,'wb') as f:f.write(b'write to original')
                self.assertEqual(local.read_bytes(),b'verified copy')
                with original_open(source,'rb') as f:self.assertEqual(f.read(),b'write to original')
        finally:
            builtins.open,Path.open=original_open,original_path_open
            os.path.exists,os.path.isfile=original_exists,original_isfile


if __name__=='__main__':unittest.main()
