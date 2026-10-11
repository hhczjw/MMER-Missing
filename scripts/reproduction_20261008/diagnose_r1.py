"""Controlled full-modality parity against a recovered legacy trainer candidate."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'experiments/ablation_p0b_0919_gradient_repair_20261007/runner'
sys.path.insert(0,str(ROOT/'scripts/checkpoint_nvme_20261007'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from data_cache import install_cache
install_cache(ROOT/'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
import torch
sys.path.insert(0,str(RUNNER))
from worker import configure,load_backend,make_loader,train_epoch
from queue_guard import acquire_idle_gpus


def digest(t):
    t=t.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def probe(model,grad=False):
    result={}
    for n,p in model.named_parameters():
        if not p.requires_grad or (grad and p.grad is None):continue
        # Full non-text tensors and representative first/last text layers.
        if not n.startswith('text_encoder.') or any(k in n for k in
            ['.layer.0.','.layer.23.','embeddings.LayerNorm']):
            result[n]=(p.grad if grad else p).detach().cpu().clone()
    return result


def compare(a,b):
    if a.keys()!=b.keys():return dict(names_equal=False)
    rows=[dict(name=n,max_abs=float((a[n]-b[n]).abs().max()),
               changed=int(torch.count_nonzero(a[n]-b[n]))) for n in a]
    rows.sort(key=lambda r:r['max_abs'],reverse=True)
    return dict(names_equal=True,max_abs=rows[0]['max_abs'],changed=sum(r['changed'] for r in rows),top=rows[:10])


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--gpu0-concurrent-r2',action='store_true',
                   help='Explicit GPU0 diagnostic while the confined GPU1 R2 baseline runs')
    p.add_argument('--mode-set',choices=['original','single_tensor','repeat_only'],default='original',
                   help='Single-tensor controls fit GPU0; explicitly do not test default foreach')
    p.add_argument('--attention-math',action='store_true',
                   help='Diagnostic-only: select SDPA math backend to test repeatability')
    p.add_argument('--reference-ism-backward',action='store_true',
                   help='Diagnostic-only: native scan/conv forward, reference backward')
    p.add_argument('--wait-worker',type=int);cli=p.parse_args()
    if cli.gpu0_concurrent_r2:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise RuntimeError('Concurrent diagnostic must be confined to physical GPU0')
        sys.path.insert(0,str(ROOT/'scripts/reproduction_20261009'))
        from verify_r1_checkpoint import acquire_gpu0_validation
        gpu_locks=acquire_gpu0_validation()
    else:
        gpu_locks=acquire_idle_gpus()
    cli.output.mkdir(parents=True,exist_ok=False)
    write(cli.output/'status.json',dict(phase='waiting_for_gpu',pid=os.getpid(),wait_worker=cli.wait_worker))
    if cli.wait_worker:
        until=time.monotonic()+3600;proc=Path('/proc')/str(cli.wait_worker)/'stat'
        while proc.exists() and proc.read_text().split(') ',1)[1][0]!='Z':
            if time.monotonic()>until:raise TimeoutError('Preceding diagnostic did not finish')
            time.sleep(3)
    torch.cuda.set_device(0);device=torch.device('cuda:0')
    if cli.attention_math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    manifest=json.loads((ROOT/'results/mosi_ablations_p0b_0919_gradient_repair_20261007/plan.json').read_text())
    args=configure('R1',2024,120,{},manifest['reference_args']);args.batch_size=cli.batch_size
    args.gradient_diagnostic_dir=str(cli.output/'gradient_diagnostics')
    builder,forward,Dataset,collate=load_backend('R1')
    from legacy_r1_trainer import Trainer
    from utils.seed import set_seed
    set_seed(2024)
    loader,stats=make_loader(args,'train',Dataset,collate,'R1',cli.batch_size)
    model=builder(args,{}).to(device)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    it=iter(loader);batches=[next(it) for _ in range(cli.steps)];del it
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    report=dict(seed=2024,batch_size=cli.batch_size,steps=cli.steps,data=stats,
        physical_gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),
        attention_implementation=model.text_encoder.transformer.config._attn_implementation,
        attention_backend='math_only' if cli.attention_math else 'default',
        legacy_candidate='Git index 0006b54: two-group AdamW and MultiTaskLoss; not a complete dated September snapshot',
        batches=[dict(ids=list(map(str,b['ids'])),hashes={k:digest(b[k]) for k in
                    ['text','audio','vision','context_text','context_audio','context_video']}) for b in batches],
        runs={},comparisons={})
    baseline=None
    modes=[('legacy',False,None,False),('legacy_repeat',False,None,False),
           ('legacy_checkpoint',True,None,False),('legacy_foreach_false',False,False,False),
           ('current',True,False,True)]
    if cli.mode_set in ('single_tensor','repeat_only'):
        modes=[('legacy_foreach_false',False,False,False),
               ('legacy_foreach_false_repeat',False,False,False),
               ('legacy_checkpoint_foreach_false',True,False,False),
               ('current',True,False,True)]
        if cli.mode_set=='repeat_only':modes=modes[:2]
    report['mode_set']=cli.mode_set
    report['reference_ism_backward']=cli.reference_ism_backward
    if cli.reference_ism_backward:
        sys.path.insert(0,str(ROOT/'scripts/reproduction_20261009'))
        from selective_scan_reference_backward import reference_scan_backward
    baseline_name=modes[0][0]
    report['comparison_reference']=baseline_name
    for name,checkpoint,foreach,current in modes:
        write(cli.output/'status.json',dict(phase='diagnosing',mode=name,pid=os.getpid()))
        with torch.no_grad():
            for n,param in model.named_parameters():
                if n in initial:param.copy_(initial[n].to(device))
        model.zero_grad(set_to_none=True)
        if checkpoint:model.text_encoder.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:model.text_encoder.transformer.gradient_checkpointing_disable()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        legacy=Trainer(deepcopy(args),model,logging.getLogger('legacy_probe'))
        opt=legacy.optimizer
        if foreach is False:
            for group in opt.param_groups:group['foreach']=False
        rows=[];states=[];last_forward={}
        def trace(module,inputs,output):
            logits=output['logits'] if isinstance(output,dict) else output
            last_forward.clear();last_forward.update(logits=logits.detach().cpu(),hash=digest(logits))
        hook=model.register_forward_hook(trace)
        original_step=opt.step
        def step(*aa,**kw):
            grads=probe(model,True);value=original_step(*aa,**kw)
            params=probe(model)
            states.append(dict(gradients=grads,parameters=params,logits=last_forward['logits'].clone()))
            rows.append(dict(step=len(rows),logits_hash=last_forward['hash'],
                             lr=[g['lr'] for g in opt.param_groups]))
            return value
        opt.step=step
        try:
            backward_stats={}
            backward_context=(reference_scan_backward(backward_stats,include_conv=True)
                              if cli.reference_ism_backward else nullcontext())
            with backward_context:
                if current:
                    value=train_epoch(model,batches,args,'R1',{},device,forward,opt,1,legacy.scheduler)
                    loss=value['loss']
                else:loss=legacy.train_one_epoch(batches,1)
            row=dict(status='passed',loss=loss,steps=rows,checkpoint=checkpoint,foreach=foreach,
                     backward_substitution=backward_stats,
                     cpu_rng_after=digest(torch.get_rng_state()),cuda_rng_after=digest(torch.cuda.get_rng_state()),
                     peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
            if name==baseline_name:baseline=states
            elif baseline is not None:
                report['comparisons'][name]=[dict(step=i,gradients=compare(a['gradients'],b['gradients']),
                    parameters=compare(a['parameters'],b['parameters']),
                    logits_max_abs=float((a['logits']-b['logits']).abs().max()))
                    for i,(a,b) in enumerate(zip(baseline,states))]
        except torch.OutOfMemoryError:
            row=dict(status='oom',checkpoint=checkpoint,foreach=foreach,
                     note='Diagnostic resource failure only; no old experiment changed')
            print('DIAGNOSTIC_OOM',name,flush=True)
        report['runs'][name]=row
        print('R1_MODE',name,row,flush=True)
        hook.remove();del legacy,opt,states
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        write(cli.output/'report.json',report)
    write(cli.output/'status.json',dict(phase='complete',pid=os.getpid()))


if __name__=='__main__':main()
