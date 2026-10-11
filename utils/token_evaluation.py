"""Explicit, fixed-mask evaluation for MMER's feature/input missing contracts."""
import numpy as np
import torch
from .metrics import eval_regression, eval_classification
from .token_protocol import validity, native_corruption, aligned_availability, feature_corruption_masks


def evaluate_curve(model, loader, args, device, rates, available=('text','audio','vision'), prediction_archive=None):
    protocol = getattr(args, 'token_eval_protocol', 'feature_valid')
    if protocol not in ('feature_valid','input_valid_aware','input_valid_blind'):
        raise ValueError(f'Unsupported Token protocol: {protocol}')
    model.eval()
    tokenizer = getattr(getattr(model,'bert_encoder',None),'tokenizer',None)
    unk_id = tokenizer.unk_token_id if tokenizer is not None else 100
    mask_seed = getattr(args,'token_mask_seed',20261005)
    curve = {}
    archive = {}
    for rate in rates:
        predictions,labels,ids = [],[],[]
        with torch.no_grad():
            for batch_idx,b in enumerate(loader):
                t,a,v = [b[k].to(device) for k in ('text','audio','vision')]
                valid = validity(t,a,v,b.get('audio_valid_lengths'),b.get('vision_valid_lengths'))
                L = t.shape[-1] if model.use_bert else t.shape[1]
                seed = mask_seed + int(round(rate*10))*1_000_003 + batch_idx*10_007
                generator = torch.Generator(device=device).manual_seed(seed)
                if protocol == 'feature_valid':
                    inputs = (t,a,v)
                    tm,_ = feature_corruption_masks(valid,L,rate,generator)
                else:
                    inputs,keeps,_ = native_corruption(t,a,v,valid,rate,generator,unk_id)
                    tm = aligned_availability(keeps,L,valid) if protocol == 'input_valid_aware' else None
                    inputs = list(inputs)
                    for i,name in enumerate(('text','audio','vision')):
                        if name not in available:
                            inputs[i] = torch.zeros_like(inputs[i])
                mm = torch.tensor([name in available for name in ('text','audio','vision')],device=device).expand(t.shape[0],-1)
                pred = model(*inputs,token_mask=tm,modality_mask=mm)['logits']
                predictions.append((pred.reshape(-1) if args.task_type=='regression' else pred.argmax(-1)).cpu().numpy())
                labels.append(b['label'].numpy().reshape(-1))
                ids.extend(str(x) for x in b['id'])
        p,y = np.concatenate(predictions),np.concatenate(labels)
        met = eval_regression(p,y) if args.task_type=='regression' else eval_classification(p,y,args.num_classes)
        curve[f'{rate:.1f}'] = {k:float(v) for k,v in met.items()}
        if prediction_archive is not None:
            if 'labels' in archive:
                assert np.array_equal(y, archive['labels']) and np.array_equal(ids, archive['sample_ids'])
            archive.update(labels=y, sample_ids=np.asarray(ids))
            archive[f'r{rate:.1f}_preds'] = p
    if prediction_archive is not None:
        np.savez_compressed(prediction_archive, **archive)
    return {'per_rate':curve,'avg_over_rates':{k:float(np.mean([met[k] for met in curve.values()])) for k in next(iter(curve.values()))},
            'protocol':{'name':protocol,'mask_seed':mask_seed,'acc7':'numpy.round(numpy.clip(x,-3,3))',
                        'feature_valid_note':'Dropout follows ISM; complete encoding is available, not raw-input missing' if protocol=='feature_valid' else ''}}
