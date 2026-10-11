"""CPU regression checks for the audited p0B cohort; no GPU model import."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import importlib.util
import numpy as np
import logging
import torch

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'scripts/ablations_p0b_0919'
spec=importlib.util.spec_from_file_location('p0b_plan',DIRECTORY/'plan.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)


class P0BReferenceTests(unittest.TestCase):
    def test_logged_hyperparameters(self):
        refs=plan.reference_arguments();a=refs['R1']
        expected=dict(batch_size=4,learning_rate=.0004,bert_learning_rate=.00001,
                      dropout=.6,ism_depth=1,d_state=64,num_layers=2,epochs=120,
                      early_stop=30,use_context=True,use_gcmn_gate=True,use_bssm_gate=True,
                      grad_clip=.1,warmup_ratio=.15,multi_task=True,sub_loss_lambda=.3)
        for k,v in expected.items():self.assertEqual(a[k],v,k)

    def test_config_retains_inactive_auxiliary_and_fp32(self):
        tree=ast.parse((DIRECTORY/'worker.py').read_text())
        scope=dict(json=json,SimpleNamespace=SimpleNamespace)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'configure','exec'),scope)
        a=scope['configure']('R1',2024,120,{},plan.reference_arguments())
        self.assertFalse(a.amp);self.assertFalse(a.pretokenize_text)
        self.assertEqual(a.num_workers,4);self.assertEqual(a.KeyEval,'MAE')
        self.assertEqual(a.effective_sub_loss_weight,0)
        self.assertEqual(a.sub_loss_lambda,.3)
        b=scope['configure']('R1',2024,120,{'use_context':False},plan.reference_arguments())
        self.assertFalse(b.use_context)

    def test_no_adapter_overrides_forward(self):
        tree=ast.parse((DIRECTORY/'r1_backend.py').read_text())
        self.assertFalse(any(isinstance(n,ast.ClassDef) for n in tree.body))
        self.assertIn('MSAClassifier(', (DIRECTORY/'r1_backend.py').read_text())
        self.assertNotIn('masked_align',(DIRECTORY/'r1_backend.py').read_text())

    def test_loss_does_not_backpropagate_auxiliary_heads(self):
        main=torch.randn(4,requires_grad=True);aux=torch.randn(4,requires_grad=True)
        y=torch.tensor([-1.,0.,1.,2.])
        torch.nn.functional.smooth_l1_loss(main,y,beta=.5).backward()
        self.assertIsNotNone(main.grad);self.assertIsNone(aux.grad)
        source=(DIRECTORY/'worker.py').read_text()
        self.assertIn('loss=task',source)
        self.assertNotIn('loss=task+args.sub_loss_lambda',source)

    def test_inactive_auxiliary_not_presented_as_ablation(self):
        r1=[r for r in plan.variants() if r['track']=='R1']
        self.assertEqual(len(r1),11)
        self.assertEqual(r1[6]['changes'],{'use_context':False})
        self.assertTrue(all('sub_loss_weight' not in r['changes'] for r in r1))

    def test_seed_pairing_and_strict_secondary_selection(self):
        jobs=[r for r in plan.jobs() if r['track']=='R1']
        self.assertEqual(len(jobs),33)
        self.assertEqual([jobs[i]['seed'] for i in [0,11,22]],[2024,42,1111])
        self.assertEqual(plan.selector('R1',dict(Acc2_nonzero=.8,MAE=.5)),
                         plan.selector('R1',dict(Acc2_nonzero=.8,MAE=.6)))

    def test_safety_gates_and_no_other_gpu(self):
        source=(DIRECTORY/'run_queue.py').read_text()
        self.assertIn("('reference',[dict(manifest['variants'][0],seed=2024)])",source)
        self.assertIn("if v['track']=='R1'",source)
        self.assertIn("CUDA_VISIBLE_DEVICES='0'",source)
        self.assertIn('deferred_tracks',source)
        source=(DIRECTORY/'worker.py').read_text()
        self.assertIn('full_training_epoch_pressure=stress',source)
        self.assertIn('foreach=False',source)
        self.assertIn("primary_mae=validation['summary']['MAE']",source)

    def test_parse_isolated_sources(self):
        for path in DIRECTORY.glob('*.py'):ast.parse(path.read_text(),filename=str(path))

    def test_historical_context_is_row_shift_not_new_clip_order(self):
        source=plan.EXPERIMENT/'snapshot_r1/dataset/data_loader.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MMDataset')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_init_context')
        scope=dict(np=np,logger=logging.getLogger('test'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'context','exec'),scope)
        t=np.arange(12).reshape(3,2,2);a=np.arange(6).reshape(3,2,1).astype(np.float32)
        obj=SimpleNamespace(text=t,audio=a,vision=a+10,use_bert=True)
        scope['_init_context'](obj,{'text_bert':t},'train')
        self.assertTrue(np.array_equal(obj.context_text,t[[0,0,1]]))
        self.assertTrue(np.array_equal(obj.context_audio,a[[0,0,1]]))
        self.assertNotIn('_previous_context_indices',ast.unparse(node))


if __name__=='__main__':unittest.main()
