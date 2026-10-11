"""Native-function GPU smoke only: no epoch queue, best selection or test scores."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from plan import ROOT,variants,apply_loss_changes

SNAP=Path(os.environ.get('R2_SNAPSHOT_ROOT',str(ROOT/'experiments/research2_smoke_gpu1_20261006/snapshot')))
sys.path.insert(0,str(SNAP))
import _path_setup
import numpy as np
import torch
from models import UMRMv3
from dataset.data_loader import MMDataLoader
from scripts.train import train_epoch,evaluate,TaskLoss
from utils.seed import set_seed
from utils.token_evaluation import evaluate_curve


def write(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['models','utils','scripts','dataset','losses']
            for p in (SNAP/folder).rglob('*.py')}


class ProbeAdam(torch.optim.AdamW):
    def __init__(self,model,args):
        main=[p for n,p in model.named_parameters() if p.requires_grad and
              not (n.startswith('bert_encoder') and 'lora_' not in n)]
        other=[p for n,p in model.named_parameters() if p.requires_grad and
               n.startswith('bert_encoder') and 'lora_' not in n]
        groups=[{'params':main}]
        if other:groups.append({'params':other,'lr':args.bert_lr})
        super().__init__(groups,lr=args.learning_rate,weight_decay=args.weight_decay,foreach=False)
        self.probes={k:[p for n,p in model.named_parameters() if n.startswith(prefix)]
                     for k,prefix in {'rcd':'aux_cls_head','decoder':'contrastive_reconstruction.reconstruct_decoder',
                                      'contrastive':'contrastive_reconstruction.token_proj',
                                      'audio':'proj_audio','ume':'missing_encoder'}.items()}
        self.last_gradients={}
        self.steps=0

    def step(self,closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
        self.last_gradients={k:sum(float(p.grad.abs().sum()) for p in pp if p.grad is not None)
                             for k,pp in self.probes.items()}
        result=super().step(closure)
        self.steps+=1
        return result


def interventions(model,changes):
    """Intervene after full initialization; restore modules and hooks on exit."""
    hooks=[]
    retained={k:getattr(model,k) for k in ['inj_audio','inj_vision','core_fusion_rev','combo_heads']}
    model.use_balanced_missing=changes.get('balanced_missing',model.use_balanced_missing)
    if changes.get('ume') is False:
        hooks.append(model.missing_encoder.register_forward_hook(
            lambda m,ins,out:(ins[0],torch.zeros_like(out[1]))))
    if changes.get('enhance') is False:
        for m in model.missing_enhancers:
            hooks.append(m.register_forward_hook(lambda m,ins,out:m.layer_norm(ins[0])))
    if changes.get('enhanced_bypass') is False:
        def zero_bypass(m,ins):
            value=ins[0].clone();d=model.args.d_model
            for start in [0,2*d,4*d]:value[:,start:start+d]=0
            return (value,)
        hooks.append(model.fusion_norm.register_forward_pre_hook(zero_bypass))
    if changes.get('text_injection') is False:model.inj_audio=model.inj_vision=None
    if changes.get('reverse_fusion') is False:model.core_fusion_rev=None
    if changes.get('combo_heads') is False:model.combo_heads=None
    if 'fixed_feature_rate' in changes:
        def fixed_rate(m,ins,kwargs):
            if torch.is_grad_enabled():m.token_missing_rate=changes['fixed_feature_rate']
        hooks.append(model.register_forward_pre_hook(fixed_rate,with_kwargs=True))
    rec=model.contrastive_reconstruction
    original_rec=rec.reconstruction_loss
    weight=changes.get('feature_rec_weight',1.)
    rec_calls=[]
    def rec_probe(*aa,**kw):
        val=original_rec(*aa,**kw);rec_calls.append(float(val.detach()))
        return val if weight==1. else val*weight
    rec.reconstruction_loss=rec_probe
    def restore():
        for h in hooks:h.remove()
        for k,v in retained.items():setattr(model,k,v)
        rec.reconstruction_loss=original_rec
    return restore,rec_calls


def check_forward(model,batch,args,device):
    model.eval()
    with torch.no_grad():
        t,a,v=[batch[k].to(device) for k in ['text','audio','vision']]
        mm=torch.ones(t.shape[0],3,device=device)
        out=model(t,a,v,modality_mask=mm,token_mask=None)['logits']
    assert torch.isfinite(out).all()
    return out.detach().cpu()


def run_track(track,cli,status):
    prep=json.loads((ROOT/'results/research2_ablation_preparation_20261006/plan.json').read_text())
    reference=deepcopy(prep['reference_args'][track])
    reference.update(bert_pretrained='FacebookAI/roberta-large',num_workers=0,lora_r=8,lora_alpha=16)
    if track=='R2-T':reference['token_train_protocol']='input_valid_aware'
    else:reference['token_train_protocol']='feature_valid'
    args=SimpleNamespace(**reference)
    device=torch.device('cuda:0')
    set_seed(args.seed)
    train_loader=MMDataLoader(args,'train')
    assert len(train_loader)==160 and train_loader.drop_last
    batch=next(iter(train_loader))
    eval_args=deepcopy(args);eval_args.batch_size=16
    valid_loader=MMDataLoader(eval_args,'valid')
    short_valid=[next(iter(valid_loader))]
    model=UMRMv3(args).to(device)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    original_balanced=model.use_balanced_missing
    folder=cli.output/track;folder.mkdir()
    results=[]
    for variant in variants(track):
        changes=variant['changes']
        args=SimpleNamespace(**apply_loss_changes(reference,changes))
        args.use_kd=True
        args.use_cd_kd=changes.get('cd_kd',reference.get('use_cd_kd',False))
        args.token_train_rate_cap=changes.get('fixed_token_rate_cap',.7)
        # Diagnostic fixed corruption activates reconstruction; policy contrast samples its cap.
        args.token_train_fixed_rate=None if 'fixed_token_rate_cap' in changes else .6
        model.args=args
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in initial:p.copy_(initial[n].to(device))
        model.use_balanced_missing=original_balanced
        if model.curriculum_scheduler is not None:model.curriculum_scheduler.current_epoch=30
        set_seed(args.seed)
        restore,rec_calls=interventions(model,changes)
        optimizer=ProbeAdam(model,args)
        seen=[]
        def capture(m,ins,kwargs,out):
            seen.append(dict(teacher=not torch.is_grad_enabled(),training=m.training,
                target_present=kwargs.get('reconstruction_targets') is not None,
                targets_detached=all(not v.requires_grad for v in kwargs.get('reconstruction_targets',[])),
                modality=out['modality_mask'].detach().cpu().tolist(),
                token_min=float(out['token_mask'].min()),
                aux=out.get('aux_logits'),logits=out['logits']))
        hook=model.register_forward_hook(capture,with_kwargs=True)
        start=time.time();torch.cuda.reset_peak_memory_stats()
        before=model.proj_audio.weight.detach().clone()
        try:
            loss=train_epoch(model,[batch],TaskLoss(args.task_type,args.num_classes),optimizer,args,device,31 if track=='R2-M' else 3,args.epochs)
            hook.remove()
            assert np.isfinite(loss) and optimizer.steps==1
            assert not torch.equal(before,model.proj_audio.weight)
            teacher,student=seen
            assert teacher['teacher'] and not teacher['logits'].requires_grad
            assert teacher['training']==(track=='R2-M')
            assert student['target_present']==(track=='R2-T')
            assert student['targets_detached']
            if track=='R2-T':assert np.all(np.asarray(student['modality'])==1)
            if changes.get('kd_weight')==0 and changes.get('rcd_weight',.1)>0:
                assert optimizer.last_gradients['rcd']>0,'RCD killed by KD-off'
            if changes.get('rcd_weight')==0:assert optimizer.last_gradients['rcd']==0
            if changes.get('feature_rec_weight')==0:
                assert optimizer.last_gradients['decoder']==0
                assert optimizer.last_gradients['contrastive']>0,'Feature reconstruction switch killed Token contrastive gradient'
            assert rec_calls
            if track=='R2-M':metrics=evaluate(model,short_valid,args,device)
            else:
                args.token_mask_seed=20261006
                metrics=evaluate_curve(model,short_valid,args,device,[0.,.5,.9])
            def finite(v):
                if isinstance(v,dict):return all(finite(x) for x in v.values())
                if isinstance(v,(float,int)):return np.isfinite(v)
                return True
            assert finite(metrics),'Nonfinite evaluation'
            row=dict(id=variant['id'],status='passed',loss=loss,seconds=time.time()-start,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                optimizer_update=True,gradients=optimizer.last_gradients,
                teacher_train_mode=teacher['training'],teacher_stop_gradient=True,
                clean_teacher_targets=student['target_present'],feature_reconstruction_raw=rec_calls,
                eval_scenes=['TAV','TA','TV','AV','T','A','V'] if track=='R2-M' else ['Token0','Token0.5','Token0.9'],
                changes=changes,diagnostic_metrics_only=True)
            write(folder/(variant['id']+'.json'),row)
            results.append(row);status['passed'].append(variant['id'])
            status.update(current=variant['id'],phase='smoke');write(cli.output/'status.json',status)
            print('PASSED',variant['id'],'loss',loss,'seconds',round(row['seconds'],1),flush=True)
        finally:
            hook.remove();restore();del optimizer,seen,before
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    # Full native-size epoch and full validation pressure, with baseline restored.
    args=SimpleNamespace(**reference);model.args=args
    model.use_balanced_missing=original_balanced
    if model.curriculum_scheduler is not None:model.curriculum_scheduler.current_epoch=30
    if track=='R2-T':args.token_train_rate_cap=.7
    set_seed(args.seed)
    optimizer=ProbeAdam(model,args)
    start=time.time();torch.cuda.reset_peak_memory_stats()
    status.update(phase='pressure',current=track);write(cli.output/'status.json',status)
    print('PRESSURE_START',track,flush=True)
    loss=train_epoch(model,train_loader,TaskLoss(args.task_type,args.num_classes),optimizer,args,device,31 if track=='R2-M' else 3,args.epochs)
    assert optimizer.steps==160 and np.isfinite(loss)
    del optimizer;model.zero_grad(set_to_none=True)
    if track=='R2-M':metrics=evaluate(model,valid_loader,args,device)
    else:
        args.token_mask_seed=20261006
        metrics=evaluate_curve(model,valid_loader,args,device,[i/10 for i in range(10)])
    probe=check_forward(model,short_valid[0],args,device)
    checkpoint_folder=cli.checkpoints/track
    checkpoint_folder.mkdir(parents=True,exist_ok=False)
    ckpt=checkpoint_folder/'smoke_checkpoint.pth'
    torch.save({'model':model.state_dict(),'args':vars(args),'smoke_only':True},ckpt)
    saved=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    assert torch.equal(probe,check_forward(model,short_valid[0],args,device))
    pressure=dict(status='passed',batches=160,loss=loss,validation_samples=229,
        validation_scenes=7 if track=='R2-M' else 10,
        checkpoint_strict_roundtrip=True,checkpoint=str(ckpt),seconds=time.time()-start,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
        diagnostic_metrics_only=True,metrics=metrics)
    write(folder/'pressure.json',pressure)
    status['pressure_passed'].append(track);write(cli.output/'status.json',status)
    print('PRESSURE_PASSED',track,'seconds',round(pressure['seconds'],1),flush=True)
    del model,initial;torch.cuda.empty_cache()
    return results,pressure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path)
    cli=ap.parse_args()
    if cli.checkpoints is None:
        cli.checkpoints=Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints')/cli.output.name
    if cli.checkpoints.exists():raise FileExistsError(cli.checkpoints)
    cli.output.mkdir(parents=True,exist_ok=False)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','Must target physical GPU1 only'
    torch.cuda.set_device(0)
    status=dict(phase='starting',pid=os.getpid(),physical_gpu=1,started_at=datetime.now().isoformat(),
        passed=[],pressure_passed=[],formal_training_started=False)
    write(cli.output/'status.json',status)
    source=hashes();write(cli.output/'source_sha256.json',source)
    try:
        for track in ['R2-M','R2-T']:run_track(track,cli,status)
        assert hashes()==source
        status.update(phase='complete',finished_at=datetime.now().isoformat(),gpu_smoke_passed=True,
            ready_for_formal_training=False,
            remaining_gate='Historical modality source/mask equivalence and fixed archived-checkpoint prediction parity still pending')
        write(cli.output/'status.json',status)
        print('ALL_SMOKES_PASSED',len(status['passed']),flush=True)
    except Exception:
        status.update(phase='failed',traceback=traceback.format_exc());write(cli.output/'status.json',status)
        raise


if __name__=='__main__':main()
