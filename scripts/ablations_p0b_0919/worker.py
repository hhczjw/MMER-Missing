"""One exclusive ablation worker; isolated imports; validation-only checkpoint selection."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from plan import ROOT,EXPERIMENT,RATES,METRICS,variants,selector,atomic_json,assert_frozen
from metrics import eval_regression


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value


def configure(track,seed,epochs,changes,references):
    if track=='R1':
        data=json.loads(json.dumps(references['R1']))
        data.update(seed=seed,epochs=epochs,bert_pretrained='FacebookAI/roberta-large',
                    pretokenize_text=False,feature_dims=[1024,768,18],
                    sub_loss_lambda=.3,effective_sub_loss_weight=0.,use_context=changes.get('use_context',True),
                    # These are the actual 50-position arrays used by the reference loader.
                    contrastive_weight=0.,aux_cls_weight=0.,amp=False,KeyEval='MAE')
    else:
        data=json.loads(json.dumps(references['R2']))
        data.update(seed=seed,epochs=epochs,lora_r=8,lora_alpha=16,num_workers=0)
        if track=='R2-M':
            data.update(use_curriculum=True,use_cd_kd=True,use_balanced_missing=True,
                        modality_missing_rate=.3,token_missing_rate_max=.6,token_train_protocol='feature_valid',KeyEval='missing_avg_MAE')
        else:data['KeyEval']='Token10_Acc7'
    return SimpleNamespace(**data)


def load_backend(track):
    root=EXPERIMENT/('snapshot_r1' if track=='R1' else 'snapshot_r2')
    sys.path.insert(0,str(root))
    import _path_setup
    if track=='R1':
        from r1_backend import build_model,forward_batch
        from dataset.data_loader import MMDataset,_collate_fn
        return build_model,forward_batch,MMDataset,_collate_fn
    from r2_backend import build_model,loss_batch
    from dataset.data_loader import MMDataset,mm_collate
    return build_model,loss_batch,MMDataset,mm_collate


def make_loader(args,split,Dataset,collate,track,batch_size,smoke=False):
    data=Dataset(args,split)
    expected={'train':1284,'valid':229,'test':686}[split]
    if len(data)!=expected:raise ValueError(f'{split} N={len(data)}, expected {expected}')
    ids=np.asarray([str(x) for x in data.ids])
    if len(set(ids))!=len(ids):raise ValueError(f'Duplicate {split} sample IDs')
    labels=np.asarray(data.labels['M'] if track=='R1' else data.labels).reshape(-1)
    if not np.isfinite(labels).all():raise ValueError('Nonfinite dataset labels')
    for name in ['audio','vision']:
        x=getattr(data,name)
        if not np.isfinite(x).all():raise ValueError(f'Nonfinite {split}/{name}')
    stats=dict(split=split,n_samples=len(data),sample_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
               label_zero_count=int((labels==0).sum()),shapes={k:list(getattr(data,k).shape) for k in ['text','audio','vision']})
    if smoke:data=Subset(data,range(min(len(data),batch_size*2)))
    generator=None if track=='R1' else torch.Generator().manual_seed(args.seed+31337)
    loader=DataLoader(data,batch_size=batch_size,shuffle=split=='train',drop_last=False,
                      generator=generator,num_workers=args.num_workers,pin_memory=True,collate_fn=collate)
    return loader,stats


def finite_metrics(metrics):
    # Constant predictions have undefined Pearson correlation; report a flag and finite sentinel.
    if not math.isfinite(float(metrics['Corr'])):metrics['Corr']=0.
    if not all(math.isfinite(float(x)) for x in metrics.values()):raise FloatingPointError('Nonfinite metric')
    return {k:float(x) for k,x in metrics.items()}


@torch.no_grad()
def evaluate(model,loader,args,track,device,forward,mask_seed,rates,archive=None):
    model.eval()
    arrays={};metrics={}
    if track=='R2-T':
        from utils.token_evaluation import evaluate_curve
        args.token_mask_seed=mask_seed
        result=evaluate_curve(model,loader,args,device,rates,prediction_archive=archive)
        result['per_rate']={k:finite_metrics(v) for k,v in result['per_rate'].items()}
        result['avg_over_rates']={k:float(np.mean([m[k] for m in result['per_rate'].values()])) for k in next(iter(result['per_rate'].values()))}
        result['summary']=result['avg_over_rates']
        return result
    combos=[('TAV',[1,1,1])] if track=='R1' else list(zip(['TAV','TA','TV','AV','T','A','V'],
                                                        [[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1]]))
    for name,pattern in combos:
        preds,labels,ids=[],[],[]
        for batch in loader:
            if track=='R1':
                out=forward(model,batch,device)
                y=batch['labels']['M'];sample_ids=batch['ids']
            else:
                t,a,v=[batch[k].to(device) for k in ['text','audio','vision']]
                if not pattern[0]:
                    t=torch.zeros_like(t);t[:,0]=model.bert_encoder.tokenizer.pad_token_id
                if not pattern[1]:a=torch.zeros_like(a)
                if not pattern[2]:v=torch.zeros_like(v)
                mm=torch.tensor(pattern,device=device).float().expand(t.shape[0],-1)
                tm=torch.ones(t.shape[0],3,t.shape[-1],device=device)
                out=model(t,a,v,token_mask=tm,modality_mask=mm,
                          audio_lengths=batch['audio_valid_lengths'].to(device),vision_lengths=batch['vision_valid_lengths'].to(device))
                y=batch['label'];sample_ids=batch['id']
            p=out['logits'].reshape(-1).float().cpu().numpy()
            if not np.isfinite(p).all():raise FloatingPointError(f'Nonfinite predictions: {name}')
            preds.append(p);labels.append(y.numpy().reshape(-1));ids.extend(str(x) for x in sample_ids)
        p,y=np.concatenate(preds),np.concatenate(labels)
        if 'labels' in arrays:
            assert np.array_equal(y,arrays['labels']) and np.array_equal(ids,arrays['sample_ids'])
        arrays.update(labels=y,sample_ids=np.asarray(ids));arrays[name+'_preds']=p
        metrics[name]=finite_metrics(eval_regression(p,y))
    if archive is not None:np.savez_compressed(archive,**arrays)
    summary=metrics['TAV'] if track=='R1' else {k:float(np.mean([metrics[c][k] for c in ['TA','TV','AV','T','A','V']])) for k in metrics['TAV']}
    return dict(combos=metrics,summary=summary)


def train_epoch(model,loader,args,track,changes,device,forward,optimizer,epoch,scheduler=None,smoke=False):
    model.train();losses=[];component_sum={};first_grads={};rates=[]
    rng=torch.Generator(device=device).manual_seed(args.seed+epoch*1000003)
    for step,batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        if track=='R1':
            out=forward(model,batch,device);y=batch['labels']['M'].to(device).reshape(-1)
            task=torch.nn.functional.smooth_l1_loss(out['logits'].reshape(-1),y,beta=.5)
            loss=task
            components=dict(task=float(task.detach()),unimodal=0.)
        else:
            loss=forward(model,batch,args,track,changes,device,epoch,rng,fixed_rate=(.5 if step==0 else .9) if smoke else None)
            diagnostic=model.ablation_last_diagnostics
            components=diagnostic['weighted'];rates.append(diagnostic['sampled_rate'])
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
        if step==0:
            first_grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None and
                         any(k in n for k in ['proj_audio.weight','sub_fc','missing_encoder','missing_aware_gate','reconstruct_decoder','aux_cls_head','global_extractor','local_extractor','bssm_gate'])}
        optimizer.step()
        if scheduler is not None:scheduler.step()
        losses.append(float(loss.detach()))
        for key,value in components.items():component_sum[key]=component_sum.get(key,0.)+value
    if not losses:raise RuntimeError('Empty train epoch')
    return dict(loss=float(np.mean(losses)),loss_components={k:v/len(losses) for k,v in component_sum.items()},
                gradient_probe=first_grads,sampled_rates=rates,batches=len(losses),skipped_batches=0)


def optimizer_for(model,args,track):
    text,main=[],[]
    for n,p in model.named_parameters():
        if p.requires_grad:
            (text if track=='R1' and n.startswith('text_encoder.') else main).append(p)
    groups=[]
    if text:groups.append(dict(params=text,lr=args.bert_learning_rate))
    groups.append(dict(params=main,lr=args.learning_rate))
    return torch.optim.AdamW(groups,weight_decay=args.weight_decay,foreach=False)


def verify_intervention(model,row,train):
    c=row['changes'];track=row['track'];g=train['gradient_probe'];losses=train['loss_components']
    if not any('proj_audio.weight' in k and v>0 for k,v in g.items()):raise AssertionError('Audio projection has no gradient')
    if track=='R1':
        assert losses['unimodal']==0 and not any('sub_fc' in k for k in g)
        for enabled,stem in [('global_glce','global_extractor'),('local_glce','local_extractor'),('bssm_gate','bssm_gate')]:
            if c.get(enabled,True) is False:assert not any(stem in k for k in g)
        if c.get('gcmn',True) is False:assert all(x.gcmn is None for x in model.layers)
        if c.get('uniform_routing',False):assert all(x.ablation_uniform_routing for x in model.layers)
    else:
        for change,component in [('text_rec_weight','text_reconstruction'),('feature_rec_weight','feature_reconstruction'),
                                 ('kd_weight','prediction_kd'),('rcd_weight','rcd'),('gap_weight','gap')]:
            if c.get(change,None)==0:assert losses[component]==0,(change,losses)
        if not c.get('ume',True):assert not any('missing_encoder' in k for k in g)
        if not c.get('enhance',True):assert not any('missing_aware_gate' in k for k in g)
        if c.get('kd_weight',None)==0 and c.get('rcd_weight',.1)>0:
            assert any('aux_cls_head' in k and v>0 for k,v in g.items()),'RCD was accidentally disabled with KD'
        if not c.get('reverse_fusion',True):assert model.core_fusion_rev is None
        if not c.get('text_injection',True):assert model.inj_audio is None and model.inj_vision is None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--stage',choices=['smoke','train','reference'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    cli=p.parse_args();manifest=json.loads(cli.manifest.read_text());assert_frozen(manifest)
    row=next(x for x in variants() if x['id']==cli.job);track=row['track'];changes=row['changes']
    epochs=120 if track=='R1' else 60
    torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_device(device)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Worker must be restricted to physical GPU0')
    cli.output.mkdir(parents=True,exist_ok=False);cli.checkpoints.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    args=configure(track,cli.seed,epochs,changes,manifest['reference_args'])
    builder,forward,Dataset,collate=load_backend(track)
    from utils.seed import set_seed
    set_seed(cli.seed)
    stress=cli.stage=='smoke' and row['id']=='R1-00'
    train_loader,train_data=make_loader(args,'train',Dataset,collate,track,args.batch_size,smoke=cli.stage=='smoke' and not stress)
    valid_loader,valid_data=make_loader(args,'valid',Dataset,collate,track,args.batch_size if track=='R1' else 16,smoke=cli.stage=='smoke' and not stress)
    # The formal test loader is created only after selection below.
    model=builder(args,changes).to(device)
    optimizer=optimizer_for(model,args,track) if cli.stage!='reference' else None
    config=dict(track=track,variant=row,seed=cli.seed,args=vars(args),train_data=train_data,valid_data=valid_data,
                initialization='fresh model; fixed pretrained text only; interventions applied after reference initialization',
                new_cohort=True,validation_selection={'R1':'MAE early stopping; strict validation Acc2 Non0 checkpoint selection','R2-M':'six-combo mean MAE, Acc7 tie','R2-T':'Token10 mean Acc7, lower MAE tie'}[track],
                parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                source_manifest=str(cli.manifest),source_sha256=manifest['source_sha256'],
                precision='FP32; foreach=False AdamW; preserved-RNG text activation recomputation' if track=='R1' else 'FP32; foreach=False AdamW',
                R2_M_note='new reference uses detached eval clean teacher and separated losses; not a historical-score ablation' if track=='R2-M' else '')
    atomic_json(cli.output/'config.json',config)
    if cli.stage=='reference':
        if row['id']!='R1-00' or cli.seed!=2024:raise ValueError('Only fixed archived p0B is allowed')
        checkpoint=Path('/media/zjw/951FB31A9E1EB7E0/H-DCD/coupled_bi_mamba3/references/codebases/coupled_BI_Mamba3/checkpoints/MOSI_p0B_seed2024_best_Acc2.pt')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True);del payload
        test_loader,test_data=make_loader(args,'test',Dataset,collate,track,args.batch_size)
        result=evaluate(model,test_loader,args,track,device,forward,20261005,RATES,cli.output/'predictions.npz')
        expected=json.loads((Path('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/results/MOSI_p0B_seed2024_test.json')).read_text())['secondary_Acc2']
        mapping={'Acc2_nonzero':'Acc2','F1_nonzero':'F1','Acc2_has0':'Acc2_has0','F1_has0':'F1_has0','Acc7':'Acc7','MAE':'MAE','Corr':'Corr'}
        differences={k:result['summary'][k]-expected[v] for k,v in mapping.items()}
        assert all(abs(v)<1e-5 for v in differences.values()),differences
        atomic_json(cli.output/'reference.json',dict(status='passed',checkpoint=str(checkpoint),test=result,differences=differences,strict_state_dict=True))
        print('ARCHIVED P0B REFERENCE PARITY PASSED',result['summary'],flush=True)
        return
    if cli.stage=='smoke':
        start=time.time();before=model.proj_audio.weight.detach().clone()
        scheduler=None
        if stress:
            steps=args.epochs*len(train_loader);warmup=int(args.warmup_ratio*steps)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:n/max(1,warmup) if n<warmup else .5*(1+math.cos(math.pi*(n-warmup)/max(1,steps-warmup))))
        train=train_epoch(model,train_loader,args,track,changes,device,forward,optimizer,1,scheduler,smoke=True)
        assert not torch.equal(before,model.proj_audio.weight.detach())
        verify_intervention(model,row,train)
        # Free optimizer moments before checkpoint I/O, especially full RoBERTa R1.
        del optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='smoke_roundtrip_',dir=cli.checkpoints) as temp:
            file=Path(temp)/'roundtrip.pth';torch.save(model.state_dict(),file)
            state=torch.load(file,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state
        metrics=evaluate(model,valid_loader,args,track,device,forward,20261006,[0.,.5,.9])
        report=dict(status='passed',stage='smoke',job=row,seed=cli.seed,train=train,validation=metrics,
                    parameter_updated=True,checkpoint_roundtrip=True,intervention_assertions=True,
                    full_training_epoch_pressure=stress,training_batches=len(train_loader),
                    peak_gpu_mb=torch.cuda.max_memory_allocated(device)/1024**2,seconds=time.time()-start)
        atomic_json(cli.output/'smoke.json',report);print('SMOKE PASSED',row['id'],flush=True)
        return
    # R1 per-step warmup+cosine; R2 epoch warmup+plateau. Same rule for all variants in each track.
    if track=='R1':
        steps=epochs*len(train_loader);warmup=max(1,int(.15*steps))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:n/warmup if n<warmup else .5*(1+math.cos(math.pi*(n-warmup)/max(1,steps-warmup))))
    else:scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min' if track=='R2-M' else 'max',factor=.5,patience=5,min_lr=1e-6)
    base_lrs=[g['lr'] for g in optimizer.param_groups];best=None;stale=0;history=[];primary_mae=float('inf');primary_epoch=None
    patience=30 if track=='R1' else 15;checkpoint=cli.checkpoints/'best_validation.pth'
    for epoch in range(1,epochs+1):
        assert_frozen(manifest);start=time.time()
        if track!='R1' and epoch<=5:
            for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr*epoch/5
        train=train_epoch(model,train_loader,args,track,changes,device,forward,optimizer,epoch,scheduler if track=='R1' else None)
        validation=evaluate(model,valid_loader,args,track,device,forward,20261006,RATES)
        improved=best is None or selector(track,validation['summary'])>selector(track,best['validation']['summary'])
        if improved:
            best=dict(epoch=epoch,validation=validation,checkpoint=str(checkpoint))
            temporary=checkpoint.with_suffix('.pth.tmp')
            torch.save(dict(model=model.state_dict(),selection=best,config=config),temporary);temporary.replace(checkpoint)
            if track!='R1':stale=0
            atomic_json(cli.output/'selection.json',best)
        elif track!='R1':stale+=1
        if track=='R1':
            if validation['summary']['MAE']<primary_mae:
                primary_mae=validation['summary']['MAE'];primary_epoch=epoch;stale=0
            else:stale+=1
        record=dict(epoch=epoch,train=train,validation=validation,seconds=time.time()-start,
                    early_stop_primary_MAE=primary_mae if track=='R1' else None,primary_epoch=primary_epoch,
                    lr=[g['lr'] for g in optimizer.param_groups],peak_gpu_mb=torch.cuda.max_memory_allocated(device)/1024**2)
        history.append(record);atomic_json(cli.output/'history.json',history)
        atomic_json(cli.output/'status.json',dict(phase='training',pid=os.getpid(),job=row['id'],seed=cli.seed,epoch=epoch,max_epochs=epochs,best_epoch=best['epoch'],epoch_seconds=record['seconds']))
        print(f'TRAIN {row["id"]} seed{cli.seed} E{epoch} loss={train["loss"]:.5f} val={validation["summary"]} bestE={best["epoch"]} sec={record["seconds"]:.1f}',flush=True)
        if track!='R1' and epoch>=5:scheduler.step(validation['summary']['MAE' if track=='R2-M' else 'Acc7'])
        if (track=='R1' or epoch>=20) and stale>=patience:break
    atomic_json(cli.output/'frozen_test_selection.json',best)
    del optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True);del payload
    test_loader,test_data=make_loader(args,'test',Dataset,collate,track,args.batch_size if track=='R1' else 16)
    archive=cli.output/'test.predictions.npz'
    test=evaluate(model,test_loader,args,track,device,forward,20261005,RATES,archive)
    with np.load(archive) as saved:
        assert len(saved['labels'])==686 and len(set(saved['sample_ids']))==686
        for key in (test['per_rate'] if track=='R2-T' else test['combos']):
            pred=saved[(f'r{key}' if track=='R2-T' else key)+'_preds']
            recomputed=finite_metrics(eval_regression(pred,saved['labels']))
            metrics=test['per_rate'][key] if track=='R2-T' else test['combos'][key]
            assert all(abs(metrics[k]-recomputed[k])<1e-10 for k in METRICS)
    test.update(selection=best,prediction_archive=str(archive),dataset=test_data,variant=row,seed=cli.seed,epochs_run=len(history),audited=True)
    atomic_json(cli.output/'test.json',test)
    atomic_json(cli.output/'status.json',dict(phase='complete',job=row['id'],seed=cli.seed,epochs_run=len(history),best_epoch=best['epoch']))
    print('TEST AUDITED',row['id'],cli.seed,test['summary'],flush=True)


if __name__=='__main__':main()
