"""Frozen native R2 training utilities. No HPO or test-based selection."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import torch
from plan import ROOT,RATES,METRICS,reference_contract,apply_loss_changes
from smoke_gpu1 import SNAP,interventions,UMRMv3,MMDataLoader,train_epoch,TaskLoss,set_seed
from utils.metrics import eval_regression
from utils.token_evaluation import evaluate_curve


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def sources():
    files=list(Path(__file__).parent.glob('*.py'))
    files += [p for p in SNAP.rglob('*.py') if 'mamba' not in p.relative_to(SNAP).parts]
    files += list((SNAP/'configs').glob('*.json'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def assert_frozen(manifest):
    for name,expected in manifest['source_sha256'].items():
        assert sha(ROOT/name)==expected,f'Frozen R2 source changed: {name}'
    for name,expected in manifest['data_sha256'].items():
        assert sha(name)==expected,f'Frozen dataset changed: {name}'


def arguments(track,seed,changes=None):
    ref=read(ROOT/'results/research2_ablation_preparation_20261006/plan.json')['reference_args'][track]
    data=apply_loss_changes(ref,changes or {})
    data.update(seed=seed,bert_pretrained='FacebookAI/roberta-large',num_workers=0,
                lora_r=8,lora_alpha=16,model_name='umrmv3',use_kd=True,epochs=60)
    data['token_train_protocol']='input_valid_aware' if track=='R2-T' else 'feature_valid'
    data['use_cd_kd']=(changes or {}).get('cd_kd',ref.get('use_cd_kd',False))
    data['token_train_fixed_rate']=None
    return SimpleNamespace(**data)


def optimizer(model,args):
    bert=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('bert_encoder') and 'lora_' not in n]
    main=[p for n,p in model.named_parameters() if p.requires_grad and not(n.startswith('bert_encoder') and 'lora_' not in n)]
    groups=[{'params':main}]
    if bert:groups.append({'params':bert,'lr':args.bert_lr})
    return torch.optim.AdamW(groups,lr=args.learning_rate,weight_decay=args.weight_decay)


def loaders(args):
    train=MMDataLoader(args,'train')
    eval_args=deepcopy(args)
    eval_args.batch_size=16 if args.token_train_protocol=='input_valid_aware' else 8
    valid=MMDataLoader(eval_args,'valid')
    assert len(train)==160 and train.drop_last and len(valid.dataset)==229
    return train,valid,eval_args


@torch.no_grad()
def evaluate_m(model,loader,device,all_combos=False,archive=None):
    model.eval();result={};arrays={}
    combos=[('TAV',[1,1,1]),('TA',[1,1,0]),('TV',[1,0,1]),('AV',[0,1,1]),
            ('T',[1,0,0]),('A',[0,1,0]),('V',[0,0,1])]
    for name,pattern in combos if all_combos else combos[:1]:
        predictions=[];labels=[];ids=[]
        for batch in loader:
            t,a,v=[batch[k].to(device) for k in ['text','audio','vision']]
            if not pattern[0]:t=torch.full_like(t,model.bert_encoder.tokenizer.pad_token_id)
            if not pattern[1]:a=torch.zeros_like(a)
            if not pattern[2]:v=torch.zeros_like(v)
            mm=torch.tensor(pattern,device=device).bool().expand(t.shape[0],-1)
            out=model(t,a,v,modality_mask=mm,token_mask=None)['logits']
            predictions.append(out.float().cpu().numpy().reshape(-1))
            labels.append(batch['label'].numpy().reshape(-1));ids.extend(map(str,batch['id']))
        pred,label=np.concatenate(predictions),np.concatenate(labels)
        assert np.isfinite(pred).all()
        result[name]={k:float(v) for k,v in eval_regression(pred,label).items()}
        assert all(np.isfinite(v) for v in result[name].values())
        arrays.update(labels=label,sample_ids=np.asarray(ids));arrays[name+'_preds']=pred
    if archive is not None:np.savez_compressed(archive,**arrays)
    value={'per_combo':result,'full_metrics':result['TAV']}
    if all_combos:
        value['missing_avg']={k:float(np.mean([result[c][k] for c in ['TA','TV','AV','T','A','V']])) for k in result['TAV']}
        value['with_text_avg']={k:float(np.mean([result[c][k] for c in ['TA','TV','T']])) for k in result['TAV']}
        value['without_text_avg']={k:float(np.mean([result[c][k] for c in ['AV','A','V']])) for k in result['TAV']}
    return value


def validation(model,loader,args,track,device):
    if track=='R2-M':return evaluate_m(model,loader,device)
    args.token_mask_seed=20261006
    return evaluate_curve(model,loader,args,device,RATES)


def summary(result,track):
    return result['full_metrics'] if track=='R2-M' else result['avg_over_rates']


def improve(track,current,best):
    if best is None:return True
    if track=='R2-M':return current['MAE']<best['MAE']
    return current['Acc7']>best['Acc7'] or (current['Acc7']==best['Acc7'] and current['MAE']<best['MAE'])


def schedules(opt,args,track):
    if track=='R2-M':
        warm=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:.1+.9*(ep+1)/5 if ep<5 else 1.)
        plate=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=7)
    else:
        warm=None
        plate=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=1e-6)
    return warm,plate


def curriculum(model,args,track,epoch,changes):
    if track=='R2-T':
        args.token_train_rate_cap=changes.get('fixed_token_rate_cap',min(.9,.3+.2*(epoch-1)))
        model.token_missing_rate=args.token_train_rate_cap
    # Modality native train_epoch reads its scheduler; outer loop steps after training.


def checkpoint(path,model,args,epoch,validation,variant):
    tmp=path.with_suffix('.pth.tmp')
    torch.save(dict(model=model.state_dict(),args=vars(args).copy(),epoch=epoch,
                    validation=validation,variant=variant),tmp)
    tmp.replace(path)
