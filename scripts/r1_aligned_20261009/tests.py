"""CPU-only safety/config/intervention tests; never initializes CUDA."""
import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from plan import BASE, SETTINGS, jobs, config_for, check_config, read
import interventions as intervention
import run_queue as q


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        layer=nn.Module()
        layer.weight_nets=nn.ModuleDict({'audio':nn.Linear(12,2)})
        layer.cells=nn.ModuleDict()
        layer.gcmn=nn.Linear(4,4)
        self.layers=nn.ModuleList([layer])
        for name in ('ism_text','ism_audio','ism_video'):
            enc=nn.Module();block=nn.Module()
            block.global_extractor=nn.Linear(4,4)
            block.local_extractor=nn.Linear(4,4)
            block.use_bssm_gate=True;block.bssm_gate=nn.Linear(4,4)
            enc.layers=nn.ModuleList([block]);setattr(self,name,enc)


def forward_source():
    text=(BASE/'source/models/coupled_mamba3_fork.py').read_text()
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CrossMamba3Cell')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    import textwrap
    return textwrap.dedent(ast.get_source_segment(text,fn))


class FakeCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_inner=4;self.nheads=2;self.headdim=2;self.num_rope_angles=1
        self.mimo_rank=1;self.num_bc_heads=1;self.v_self_ratio=0
        self.in_proj_tgt=nn.Linear(4,15);self.c_proj_tgt=nn.Linear(4,2)
        self.b_projs=nn.ModuleDict({'a':nn.Linear(4,2),'v':nn.Linear(4,2)})
        self.v_projs=nn.ModuleDict({'a':nn.Linear(4,4),'v':nn.Linear(4,4)})
        self.B_norm=nn.Identity();self.C_norm=nn.Identity()
        self.dt_bias=nn.Parameter(torch.zeros(2));self.A_floor=1e-4
        self.B_bias=nn.Parameter(torch.zeros(1,1,2));self.C_bias=nn.Parameter(torch.zeros(1,1,2))
        self.D=nn.Parameter(torch.ones(2));self.is_mimo=False;self.is_outproj_norm=False
        self.chunk_size=64;self.rotary_dim_divisor=2;self.out_proj=nn.Linear(4,4)


def compile_cell(source):
    def fake_kernel(**kw):
        # Exercise the actual native Python K/V assembly; not a CUDA kernel test.
        scale=(kw['K']*kw['Q']).sum((-1,-2))[...,None,None]
        return kw['V']*(scale+1)
    scope=dict(torch=torch,F=F,rearrange=rearrange,Tuple=tuple,mamba3_siso_combined=fake_kernel)
    exec(compile(source,'<cpu-native-cell>','exec'),scope)
    return scope['forward']


class Tests(unittest.TestCase):
    def test_plan_33_with_paired_full(self):
        js=jobs();self.assertEqual(len(js),33)
        self.assertEqual([j['seed'] for j in js if j['id']=='R1-00'],[2024,42,1111])
        self.assertEqual(len({j['job_id'] for j in js}),33)
        self.assertEqual(q.REPORT_KEY,'secondary_Acc2')

    def test_config_only_intervention_and_paths(self):
        raw=read(BASE/'source/configs/config.json');ref={}
        for section in ('common','model'):ref.update(raw[section])
        ref.update(raw['datasets']['MOSI']);ref['dataset_root_dir']=raw['dataset_root_dir']
        ref['dataset_name']='mosi'
        for _,_,changes in SETTINGS:
            args=config_for(SimpleNamespace(**deepcopy(ref)),Path('/tmp/r1-test'),changes)
            diff=check_config(vars(args),ref,changes)
            self.assertEqual(set(diff),{'checkpoints_dir','logs_dir','results_dir'}|({'use_context'} if changes.get('context') is False else set()))
        for k,v in [('batch_size',4),('epochs',120),('grad_clip',.1),('weight_decay',1e-4),('warmup_ratio',.15),('learning_rate',4e-4),('bert_learning_rate',1e-5),('num_workers',4),('ism_mixer_type','bimamba')]:
            self.assertEqual(ref[k],v)
        with self.assertRaisesRegex(RuntimeError,'Unexpected'):
            check_config(dict(ref,batch_size=8),ref,{})

    def test_empty_intervention_identity_and_rng(self):
        m=Tiny();before={n:p.clone() for n,p in m.state_dict().items()};rng=torch.get_rng_state().clone()
        methods=[type(layer).forward for layer in m.modules()]
        self.assertEqual(intervention.apply(m,{}),[])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(methods,[type(layer).forward for layer in m.modules()])
        self.assertTrue(all(torch.equal(v,m.state_dict()[n]) for n,v in before.items()))
        self.assertTrue(all(not module._forward_hooks for module in m.modules()))

    def test_uniform_routes_and_rng(self):
        m=Tiny();x=torch.randn(2,3,12);rng=torch.get_rng_state().clone()
        intervention.apply(m,{'uniform':True})
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        weights=m.layers[0].weight_nets['audio'](x).softmax(-1)
        self.assertTrue(torch.equal(weights,torch.full_like(weights,.5)))
        self.assertTrue(all(not p.requires_grad for p in m.layers[0].weight_nets.parameters()))

    def test_glce_only_requested_branch(self):
        for mode,stem in [('global','global_extractor'),('local','local_extractor')]:
            m=Tiny();intervention.apply(m,{mode:False});x=torch.randn(2,3,4,requires_grad=True)
            for name in ('ism_text','ism_audio','ism_video'):
                block=getattr(m,name).layers[0];y=getattr(block,stem)(x)
                self.assertTrue(torch.equal(y,torch.zeros_like(y)))
                self.assertTrue(all(not p.requires_grad for p in getattr(block,stem).parameters()))
                other=getattr(block,'local_extractor' if mode=='global' else 'global_extractor')
                self.assertFalse(torch.equal(other(x),torch.zeros_like(y)))

    def test_gcmn_and_bssm_removed_after_initialization(self):
        m=Tiny();intervention.apply(m,{'gcmn':False,'bssm':False})
        self.assertIsNone(m.layers[0].gcmn)
        self.assertFalse(m.ism_text.layers[0].use_bssm_gate)
        self.assertTrue(all(not p.requires_grad for p in m.ism_text.layers[0].bssm_gate.parameters()))

    def test_actual_native_cell_kv_interventions(self):
        original=forward_source();native=compile_cell(original)
        torch.manual_seed(90);cell=FakeCell()
        target=torch.randn(2,3,4);s0=torch.randn(2,3,4);s1=torch.randn(2,3,4)
        weights=torch.full((2,3,2),.5)
        original_out=native(cell,target,s0,s1,weights,('a','v'))
        for mode in ('key_self','value_self','self'):
            fn=compile_cell(intervention.cell_source(original,mode))
            out=fn(cell,target,s0,s1,weights,('a','v'))
            self.assertEqual(out.shape,(2,3,4))
            self.assertFalse(torch.equal(original_out,out))
            cell.zero_grad();out.sum().backward()
            if mode in ('self','value_self'):
                self.assertTrue(all(p.grad is None for p in cell.v_projs.parameters()))
                self.assertIsNotNone(cell.in_proj_tgt.weight.grad)
            else:self.assertTrue(all(p.grad is not None for p in cell.v_projs.parameters()))
            if mode=='self':
                self.assertTrue(torch.equal(out,fn(cell,target,s0+7,s1-3,weights,('a','v'))))
        with self.assertRaisesRegex(RuntimeError,'source changed'):
            intervention.cell_source(original.replace('B0 =','B0  ='),'key_self')

    def test_token_in_progress_waits(self):
        with patch.object(q,'alive',return_value=True):
            self.assertFalse(q.dependency_ready(dict(phase='training',pid=10)))

    def test_failure_or_loss_blocks(self):
        with self.assertRaisesRegex(RuntimeError,'failed'):
            q.dependency_ready(dict(phase='failed'))
        with patch.object(q,'alive',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'lost'):
                q.dependency_ready(dict(phase='training',pid=10))

    def test_complete_still_waits_parent_exit(self):
        with patch.object(q,'alive',return_value=True):
            self.assertFalse(q.dependency_ready(dict(phase='complete',pid=10)))

    def test_only_gpu1_lock_gpu0_can_remain_busy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'results').mkdir()
            manifest={'current_gpu1_uuid':'GPU-one','expected_current_environment':{'driver':'580'}}
            with patch.object(q,'ROOT',root),patch.object(q,'write'),patch.object(q.fcntl,'flock'),patch.object(q.subprocess,'check_output',side_effect=['1, GPU-one, RTX, 580','100, GPU-zero']):
                lock=q.acquire_gpu1(manifest);self.assertEqual(Path(lock.name).name,'research2_gpu1.lock');lock.close()
            with patch.object(q,'ROOT',root),patch.object(q.fcntl,'flock'),patch.object(q.subprocess,'check_output',side_effect=['1, GPU-one, RTX, 580','100, GPU-one']):
                with self.assertRaises(BlockingIOError):q.acquire_gpu1(manifest)

    def test_pinned_helper_mutation_blocks(self):
        with patch.object(q.legacy,'verify_sources'),patch.object(q,'sha',return_value='changed'):
            with self.assertRaisesRegex(RuntimeError,'helper changed'):
                q.verify_pin(dict(frozen_native_manifest={},helpers={'file':'original'}))

    def test_native_trainer_two_groups_and_passive_gradient_observer(self):
        import importlib.util
        import sys
        sys.path.insert(0,str(BASE/'source'))
        spec=importlib.util.spec_from_file_location('cpu_native_trainer',BASE/'source/trainer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder=nn.Linear(4,4);self.proj_audio=nn.Linear(4,4)
                self.sub_fc=nn.Linear(4,1);self.head=nn.Linear(4,1)
            def forward(self,text,audio,video,**kw):
                x=self.text_encoder(text)+self.proj_audio(audio)+video
                return dict(logits=self.head(x),sub_T=self.sub_fc(x),sub_A=self.sub_fc(x),sub_V=self.sub_fc(x))
        args=SimpleNamespace(**read(BASE/'source/configs/config.json')['common'])
        m=SmallModel();other=deepcopy(m);logger=SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)
        baseline=module.Trainer(args,m,logger);observed=module.Trainer(args,other,logger)
        self.assertEqual(len(baseline.optimizer.param_groups),2)
        self.assertEqual([g['initial_lr'] for g in baseline.optimizer.param_groups],[1e-5,4e-4])
        self.assertTrue(all(g['weight_decay']==1e-4 and g['foreach'] is None for g in baseline.optimizer.param_groups))
        self.assertEqual((baseline.warmup_steps,baseline.total_steps),(5778,38520))
        self.assertEqual(baseline.criterion.task_weights,{'M':1.0})
        rows=[]
        observed.optimizer.register_step_pre_hook(lambda *a:rows.append(intervention.verify_gradients(other,{})))
        batches=[dict(text=torch.randn(4,4),audio=torch.randn(4,4),vision=torch.randn(4,4),labels={'M':torch.randn(4)}) for _ in range(3)]
        rng=torch.get_rng_state().clone()
        loss1=baseline.train_one_epoch(batches,1);after1=torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        loss2=observed.train_one_epoch(batches,1);after2=torch.get_rng_state().clone()
        self.assertEqual(loss1,loss2);self.assertTrue(torch.equal(after1,after2))
        self.assertTrue(all(torch.equal(p,other.state_dict()[n]) for n,p in m.state_dict().items()))
        self.assertEqual(baseline.scheduler.state_dict(),observed.scheduler.state_dict())
        self.assertEqual(baseline.global_step,3);self.assertEqual(len(rows),3)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if torch.cuda.is_initialized():raise RuntimeError('CPU test initialized CUDA')
    raise SystemExit(0 if result.wasSuccessful() else 1)
