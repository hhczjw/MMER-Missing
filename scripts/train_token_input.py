"""MOSI input-Token retraining: validation selection, then test, then fixed 3-seed ensemble.

No old MMER weights are loaded. Frozen pretrained RoBERTa + trainable LoRA
match the historical architecture. This is a Token specialist, not a replacement
for the whole-modality missing experiment. Every output directory is exclusive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _path_setup
from reeval_token_protocols import configuration
from dataset.data_loader import MMDataLoader
from models import UMRMv3
from scripts.train import train_epoch, TaskLoss
from utils.seed import set_seed
from utils.metrics import eval_regression
from utils.token_evaluation import evaluate_curve
from utils.token_protocol import validity, native_corruption, aligned_availability

RATES = [i / 10 for i in range(10)]


def write_json(path, value):
    """Atomic status/config writes; never publish an incomplete JSON."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def arguments(cli, seed):
    args = configuration()
    for k,v in dict(seed=seed, model_name='umrmv3', epochs=cli.epochs,
                    batch_size=cli.batch_size, learning_rate=cli.lr,
                    use_curriculum=False, use_cd_kd=False,
                    use_balanced_missing=False, modality_missing_rate=0.0,
                    token_missing_rate_max=0.9, token_train_protocol='input_valid_aware',
                    token_eval_protocol='input_valid_aware', token_mask_seed=20261005,
                    KeyEval='Acc7', num_workers=0).items():
        setattr(args,k,v)
    return args


def make_optimizer(model, args):
    bert = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('bert_encoder') and 'lora_' not in n]
    main = [p for n,p in model.named_parameters() if p.requires_grad and not (n.startswith('bert_encoder') and 'lora_' not in n)]
    groups = [{'params': main}]
    if bert:
        groups.append({'params':bert, 'lr':args.bert_lr})
    return torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)


def smoke(cli, device):
    args = arguments(cli, 44)
    args.batch_size = 2
    args.token_train_fixed_rate = 1.0
    set_seed(args.seed)
    model = UMRMv3(args).to(device)
    batch = next(iter(MMDataLoader(args,'train')))
    seen = []
    def capture(m, inputs, kwargs, out):
        seen.append((tuple(x.detach().cpu() for x in inputs[:3]), kwargs, out))
    hook = model.register_forward_hook(capture, with_kwargs=True)
    probe = model.proj_audio.weight
    before = probe.detach().clone()
    start = time.time()
    loss = train_epoch(model,[batch],TaskLoss(args.task_type,args.num_classes),make_optimizer(model,args),args,device,1,cli.epochs)
    hook.remove()
    clean, corrupted = seen
    assert not torch.equal(before,probe.detach()), 'No optimizer update'
    assert not clean[2]['logits'].requires_grad, 'Clean teacher retained gradients'
    assert corrupted[2]['logits'].requires_grad, 'Student detached'
    assert torch.equal(corrupted[1]['text_reconstruction_target'],clean[2]['text_target'])
    assert all(torch.equal(x,y) for x,y in zip(corrupted[1]['reconstruction_targets'],clean[2]['reconstruction_targets']))
    t,a,v = [batch[k].to(device) for k in ('text','audio','vision')]
    valid = validity(t,a,v,batch['audio_valid_lengths'],batch['vision_valid_lengths'])
    expected,keeps,_ = native_corruption(t,a,v,valid,1.0,torch.Generator(device=device).manual_seed(10),model.bert_encoder.tokenizer.unk_token_id)
    assert all(torch.equal(x.cpu(),y) for x,y in zip(expected,corrupted[0])), 'Corruption was not before the encoder'
    assert torch.equal(corrupted[2]['token_mask'],aligned_availability(keeps,t.shape[-1],valid))
    assert torch.equal(corrupted[0][0][:,1:],batch['text'][:,1:]), 'Text attention/padding modified'
    assert bool((corrupted[2]['modality_mask']==1).all()), 'Token run dropped entire modalities'
    assert np.isfinite(loss)
    args.token_train_fixed_rate = None
    metrics = evaluate_curve(model,[batch],args,device,[0.,.5,.9])
    assert all(np.isfinite(x) for x in metrics['avg_over_rates'].values())
    # Binary legacy behavior is unchanged; fractional missing windows have supervision.
    from models.umrm_v2 import MultiGranularityContrastiveReconstruction
    rec = MultiGranularityContrastiveReconstruction(4).to(device)
    r = torch.ones(1,2,4,device=device)
    y = torch.zeros_like(r)
    assert float(rec.reconstruction_loss(r,y,torch.tensor([[1.,.5]],device=device))) == 1.
    result = {'status':'passed','smoke_only':True,'loss':loss,'training_seconds':time.time()-start,
              'input_corrupted_before_encoder':True,'clean_detached_reconstruction_targets':True,
              'padding_and_actual_special_tokens_preserved':True,'fractional_av_supervision':True,
              'parameter_updated':True,'rates_evaluated':[0.,.5,.9], 'metrics':metrics}
    write_json(cli.output/'smoke.json',result)
    print('SMOKE_PASSED',json.dumps(result,ensure_ascii=False),flush=True)
    del seen,model
    torch.cuda.empty_cache()


def train_seed(cli, seed, device, status):
    args = arguments(cli,seed)
    set_seed(seed)
    folder = cli.output / f'seed{seed}'
    folder.mkdir()
    write_json(folder/'config.json',vars(args))
    train_loader = MMDataLoader(args,'train')
    args.batch_size = cli.eval_batch_size
    valid_loader = MMDataLoader(args,'valid')
    args.batch_size = cli.batch_size
    model = UMRMv3(args).to(device)
    optimizer = make_optimizer(model,args)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5,min_lr=1e-6)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    best_acc, best_mae, stagnant, history = -float('inf'),float('inf'),0,[]
    checkpoints = {k:folder/f'best_{k}.pth' for k in ('Acc7','MAE')}
    chosen = {}
    for epoch in range(1,cli.epochs+1):
        start = time.time()
        if epoch <= args.warmup_epochs:
            for g,lr in zip(optimizer.param_groups,base_lrs):
                g['lr'] = lr * epoch / args.warmup_epochs
        args.token_train_rate_cap = min(.9,.3+.2*(epoch-1))
        model.token_missing_rate = args.token_train_rate_cap
        loss = train_epoch(model,train_loader,TaskLoss(args.task_type,args.num_classes),optimizer,args,device,epoch,cli.epochs)
        args.token_mask_seed = 20261006
        validation = evaluate_curve(model,valid_loader,args,device,RATES)
        avg = validation['avg_over_rates']
        acc_improved = avg['Acc7'] > best_acc or (avg['Acc7'] == best_acc and avg['MAE'] < chosen.get('Acc7',{}).get('avg',{}).get('MAE',float('inf')))
        mae_improved = avg['MAE'] < best_mae
        row = {'epoch':epoch,'loss':loss,'validation_avg':avg,'learning_rates':[g['lr'] for g in optimizer.param_groups],
               'train':args.token_train_last_stats,'seconds':time.time()-start}
        for metric, improved in [('Acc7',acc_improved),('MAE',mae_improved)]:
            if improved:
                chosen[metric] = {'epoch':epoch,'avg':avg}
                tmp = checkpoints[metric].with_suffix('.pth.tmp')
                torch.save({'model':model.state_dict(),'args':vars(args).copy(),'epoch':epoch,
                            'validation':validation,'selection_metric':metric},tmp)
                tmp.replace(checkpoints[metric])
        if acc_improved:
            best_acc,stagnant = avg['Acc7'],0
        else:
            stagnant += 1
        best_mae = min(best_mae,avg['MAE'])
        history.append(row)
        write_json(folder/'history.json',history)
        write_json(folder/'selection.json',chosen)
        status.update(phase='training',seed=seed,epoch=epoch,max_epochs=cli.epochs,
                      best_valid_Acc7=best_acc,best_valid_MAE=best_mae,epoch_seconds=row['seconds'])
        write_json(cli.output/'status.json',status)
        print(f'SEED {seed} EPOCH {epoch:02d} loss={loss:.5f} validAcc7={avg["Acc7"]:.5f} validMAE={avg["MAE"]:.5f} bestAcc7={best_acc:.5f} seconds={row["seconds"]:.1f}',flush=True)
        if epoch >= args.warmup_epochs:
            plateau.step(avg['Acc7'])
        if stagnant >= cli.patience and epoch >= cli.min_epochs:
            print(f'EARLY_STOP seed={seed} epoch={epoch}',flush=True)
            break
    # Load only the validation-selected primary checkpoint, then open test loader.
    state = torch.load(checkpoints['Acc7'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    del state
    args.token_mask_seed = 20261005
    args.batch_size = cli.eval_batch_size
    test_loader = MMDataLoader(args,'test')
    archive = folder/'test.predictions.npz'
    test = evaluate_curve(model,test_loader,args,device,RATES,prediction_archive=archive)
    test.update(checkpoint=str(checkpoints['Acc7']),selection=chosen['Acc7'],prediction_archive=str(archive))
    write_json(folder/'test.json',test)
    print(f'SEED {seed} TEST',json.dumps(test['avg_over_rates']),flush=True)
    status.setdefault('completed_seeds',[]).append(seed)
    write_json(cli.output/'status.json',status)
    del model,optimizer,train_loader,valid_loader,test_loader
    torch.cuda.empty_cache()
    return test


def ensemble(cli, members):
    archives = [np.load(cli.output/f'seed{s}'/'test.predictions.npz') for s in cli.seeds]
    labels,ids = archives[0]['labels'],archives[0]['sample_ids']
    assert all(np.array_equal(labels,a['labels']) and np.array_equal(ids,a['sample_ids']) for a in archives)
    arrays = {'labels':labels,'sample_ids':ids}
    curve = {}
    for rate in RATES:
        key = f'r{rate:.1f}_preds'
        predictions = np.stack([a[key] for a in archives])
        arrays[f'r{rate:.1f}_member_preds'] = predictions
        arrays[key] = predictions.mean(0)
        curve[f'{rate:.1f}'] = {k:float(v) for k,v in eval_regression(arrays[key],labels).items()}
    archive = cli.output/'ensemble.predictions.npz'
    np.savez_compressed(archive,**arrays)
    result = {'dataset':'MOSI','protocol':'input_valid_aware','seeds':cli.seeds,'rates':RATES,
              'weights':[1/len(members)]*len(members),'checkpoint_selection':'per-seed validation Token10 mean Acc7, MAE tie-break',
              'test_mask_seed':20261005,'n_samples':len(labels),'per_rate':curve,
              'avg_over_rates':{k:float(np.mean([x[k] for x in curve.values()])) for k in next(iter(curve.values()))},
              'members':members,'prediction_archive':str(archive),'acc7':'numpy.round(numpy.clip(x,-3,3))'}
    for a in archives:
        a.close()
    old_path = ROOT/'results/priority1_20261006/token_protocol_paired_test_v2.json'
    if old_path.exists():
        old = json.loads(old_path.read_text())['protocols']['input_valid_aware']['avg_over_rates']
        result['old_feature_trained_input_test_avg'] = old
        result['delta_vs_old_same_input_test'] = {k:result['avg_over_rates'][k]-old[k] for k in old}
    write_json(cli.output/'ensemble.json',result)
    print('ENSEMBLE_FINAL',json.dumps(result['avg_over_rates']),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--seeds',nargs='+',type=int,default=[44,45,46])
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--min-epochs',type=int,default=20)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--eval-batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--smoke-only',action='store_true')
    cli = ap.parse_args()
    if len(set(cli.seeds)) != len(cli.seeds) or cli.epochs < 1 or cli.batch_size < 1 or cli.eval_batch_size < 1:
        ap.error('Invalid seeds, epochs, or batch size')
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True,exist_ok=False)
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    status = {'pid':os.getpid(),'phase':'smoke','seeds':cli.seeds,'completed_seeds':[],
              'device':str(device),'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),
              'smoke_only':cli.smoke_only}
    write_json(cli.output/'status.json',status)
    write_json(cli.output/'run.json',dict(vars(cli),output=str(cli.output),
              architecture='UMRMv3 / RoBERTa-large LoRA / bidirectional CoupledMamba3',
              initialization='fresh model for each seed; pretrained RoBERTa only, no old MMER checkpoint',
              training_protocol='input_valid_aware; Token-only all modalities present; clean detached teacher/reconstruction targets',
              sampling='uniform discrete 0..rate_cap per batch; cap .3/.5/.7/.9 in first four epochs, .9 thereafter',
              kd='constant weight 0.3; legacy modality/curriculum KD schedule disabled',
              validation='Token10 fixed masks seed20261006; checkpoint mean Acc7 then MAE tie-break',
              test='only after checkpoint selection; seed20261005; fixed equal seed weights',
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),ROOT/'scripts/train.py',ROOT/'models/umrm_v3.py',
                                       ROOT/'models/umrm_v2.py',ROOT/'utils/token_protocol.py',
                                       ROOT/'utils/token_evaluation.py',ROOT/'utils/metrics.py']}))
    try:
        smoke(cli,device)
        if not cli.smoke_only:
            members = [train_seed(cli,s,device,status) for s in cli.seeds]
            ensemble(cli,members)
        status.update(phase='smoke_passed' if cli.smoke_only else 'complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(cli.output/'status.json',status)
    except BaseException as error:
        status.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(cli.output/'status.json',status)
        raise


if __name__ == '__main__':
    main()
