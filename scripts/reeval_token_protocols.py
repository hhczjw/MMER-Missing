"""Paired fixed-checkpoint audit of Token missing location and mask visibility."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _path_setup
from config import load_config
from dataset.data_loader import MMDataLoader
from models import UMRMv3
from utils.metrics import eval_regression
from utils.seed import set_seed
from utils.token_protocol import validity, native_corruption, aligned_availability, feature_corruption_masks
from reeval_ensemble_token import masked_batch

PROTOCOLS = ['legacy_input', 'input_valid_blind', 'input_valid_aware', 'feature_valid']
METRICS = ['MAE','Corr','Acc2_nonzero','Acc2_has0','F1_nonzero','F1_has0','Acc7','Acc7_bins']


def configuration():
    args = load_config(str(ROOT / 'configs/umrm_v3_config.json'), 'MOSI')
    for key, value in dict(seed=42, bert_pretrained='FacebookAI/roberta-large', bert_pretokenized=True,
                           featurePath='CMU-MOSI/Processed/unaligned_50_roberta.pkl', bert_finetune='lora',
                           use_uni=True, uni_weight=0.1, gap_weight=0.05, use_rcd=True, rcd_weight=0.1,
                           use_cd_kd=True, use_balanced_missing=True, use_combo_heads=True, use_bi_direction=True,
                           use_text_recon=True, use_text_injection=True, use_kd=True, kd_weight=0.3).items():
        setattr(args, key, value)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-manifest', default=str(ROOT/'results/NEW3_token_s44s45s46_Acc7ckpt_fixedmask_clipround.json'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--split', choices=['valid','test'], default='test')
    ap.add_argument('--mask-seed', type=int, default=20261005)
    ap.add_argument('--rates', nargs='+', type=float, default=[i/10 for i in range(10)])
    ap.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=PROTOCOLS)
    ap.add_argument('--max-batches', type=int, default=0)
    ap.add_argument('--max-members', type=int, default=0)
    ap.add_argument('--output', required=True)
    cli = ap.parse_args()
    if any(r < 0 or r > 1 for r in cli.rates):
        ap.error('Missing rates must be in [0,1]')
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError(output)
    args = configuration()
    set_seed(args.seed)
    device = torch.device(cli.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable; run in an execution environment with GPU devices')
        torch.cuda.set_device(device)
    loader = MMDataLoader(args, mode=cli.split)
    batches, labels, ids = [], [], []
    for i, b in enumerate(loader):
        if cli.max_batches and i >= cli.max_batches:
            break
        batches.append({k: v.to(device) for k,v in b.items() if isinstance(v,torch.Tensor)})
        labels.extend(b['label'].numpy().reshape(-1).tolist())
        ids.extend([str(v) for v in b['id']])
    labels = np.asarray(labels)
    checkpoints = json.loads(Path(cli.checkpoint_manifest).read_text())['checkpoints']
    if cli.max_members:
        checkpoints = checkpoints[:cli.max_members]
    totals = {p: {str(r): None for r in cli.rates} for p in cli.protocols}
    arrays = {'labels':labels, 'sample_ids':np.asarray(ids)}
    members, realized = [], {}
    for member_idx, checkpoint in enumerate(checkpoints):
        model = UMRMv3(args).to(device)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model',state), strict=True)
        del state
        model.eval()
        unk_id = model.bert_encoder.tokenizer.unk_token_id
        member = {'checkpoint':checkpoint, 'protocols':{}}
        print('MEMBER',member_idx, checkpoint, flush=True)
        for rate in cli.rates:
            rate_idx = int(round(rate*10))
            predictions = {p: [] for p in cli.protocols}
            counts = {p: np.zeros((3,2),dtype=np.int64) for p in cli.protocols}
            for batch_idx,b in enumerate(batches):
                t,a,v = [b[k] for k in ['text','audio','vision']]
                valid = validity(t,a,v,b['audio_valid_lengths'],b['vision_valid_lengths'])
                L = t.shape[-1]
                seed = cli.mask_seed + rate_idx*1_000_003 + batch_idx*10_007
                generator = torch.Generator(device=device).manual_seed(seed)
                inputs, keeps, eligible = native_corruption(t,a,v,valid,rate,generator,unk_id)
                avail = aligned_availability(keeps,L,valid)
                feature_gen = torch.Generator(device=device).manual_seed(seed)
                feature_mask, feature_eligible = feature_corruption_masks(valid,L,rate,feature_gen)
                with torch.no_grad():
                    for protocol in cli.protocols:
                        if protocol == 'legacy_input':
                            xx = masked_batch(t,a,v,rate,True,unk_id,cli.mask_seed,rate_idx,batch_idx)
                            tm = None
                        elif protocol == 'feature_valid':
                            xx, tm = (t,a,v), feature_mask
                        else:
                            xx = inputs
                            tm = avail if protocol == 'input_valid_aware' else None
                        out = model(*xx,token_mask=tm,modality_mask=None)
                        if tm is not None:
                            assert torch.equal(out['token_mask'],tm), 'Model discarded the explicit missing mask'
                        pred = out['logits'].reshape(-1).cpu().numpy()
                        assert np.isfinite(pred).all()
                        predictions[protocol].append(pred)
                        if protocol != 'legacy_input':
                            for k in range(3):
                                kk = feature_mask[:,k] if protocol == 'feature_valid' else keeps[k]
                                ee = feature_eligible[:,k] if protocol == 'feature_valid' else eligible[k]
                                counts[protocol][k] += [int(((kk == 0) & ee).sum()),int(ee.sum())]
                if member_idx == 0:
                    for k in range(3):
                        arrays[f'r{rate}_b{batch_idx}_native_keep{k}'] = keeps[k].cpu().numpy().astype(np.uint8)
                    arrays[f'r{rate}_b{batch_idx}_feature_keep'] = feature_mask.cpu().numpy().astype(np.uint8)
            for protocol in cli.protocols:
                pred = np.concatenate(predictions[protocol])
                arrays[f'member{member_idx}_{protocol}_r{rate}_preds'] = pred
                total = totals[protocol][str(rate)]
                totals[protocol][str(rate)] = pred.copy() if total is None else total + pred
                member['protocols'].setdefault(protocol, {'per_rate':{}})['per_rate'][str(rate)] = {
                    k:float(x) for k,x in eval_regression(pred,labels).items()}
                if member_idx == 0 and protocol != 'legacy_input':
                    realized.setdefault(protocol,{})[str(rate)] = [float(c[0]/c[1]) if c[1] else 0 for c in counts[protocol]]
            print('RATE',rate,'complete',flush=True)
        for protocol,value in member['protocols'].items():
            value['avg_over_rates'] = {k:float(np.mean([m[k] for m in value['per_rate'].values()])) for k in METRICS}
        members.append(member)
        del model
        torch.cuda.empty_cache()
    result = {'dataset':'MOSI','split':cli.split,'mask_seed':cli.mask_seed,'checkpoints':checkpoints,
              'rates':cli.rates,'n_samples':len(labels),'smoke_only':bool(cli.max_batches or cli.max_members),
              'members':members,'protocols':{},'realized_missing_rates_TAV':realized,
              'protocol_notes':{
                  'legacy_input':'Exact historical shared-mask input corruption; padding/special-token flaws retained as control',
                  'input_valid_blind':'Only valid native frames/tokens dropped; actual CLS/SEP protected; internal mask remains all ones',
                  'input_valid_aware':'Same corrupted input as input_valid_blind; pooled native keep fractions passed to UME/Enhance',
                  'feature_valid':'Binary dropout after ISM, matching training location; clean encoding sees complete input; padding excluded; not raw-input robustness',
                  'checkpoint_selection':'Frozen historical seed44/45/46 trio for paired diagnostics, no new test-driven selection',
                  'acc7':'numpy.round(numpy.clip(x,-3,3)); Acc7_bins retained only for audit'}}
    for protocol,curves in totals.items():
        curve = {}
        for rate,pred in curves.items():
            pred = pred / len(checkpoints)
            arrays[f'ensemble_{protocol}_r{rate}_preds'] = pred
            curve[rate] = {k:float(v) for k,v in eval_regression(pred,labels).items()}
        result['protocols'][protocol] = {'per_rate':curve,'avg_over_rates':{
            k:float(np.mean([m[k] for m in curve.values()])) for k in METRICS}}
        print(protocol,result['protocols'][protocol]['avg_over_rates'],flush=True)
    if 0.0 in cli.rates:
        base = arrays[f'ensemble_{cli.protocols[0]}_r0.0_preds']
        assert all(np.allclose(base,arrays[f'ensemble_{p}_r0.0_preds'],atol=1e-6) for p in cli.protocols), 'Zero-rate predictions differ across protocols'
    output.parent.mkdir(parents=True,exist_ok=True)
    archive = output.with_suffix('.predictions.npz')
    np.savez_compressed(archive,**arrays)
    result['prediction_archive'] = str(archive)
    result['mask_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('SAVED',output,flush=True)


if __name__ == '__main__':
    main()
