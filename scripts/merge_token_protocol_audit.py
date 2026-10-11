"""Combine the unchanged control arms with the corrected valid-frame mask arm."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/priority1_20261006'


def main():
    old=json.loads((ROOT/'token_protocol_paired_test.json').read_text())
    corrected=json.loads((ROOT/'token_input_aware_validfraction.json').read_text())
    for key in ['dataset','split','mask_seed','checkpoints','rates','n_samples']:
        assert old[key]==corrected[key],key
    a=np.load(old['prediction_archive'],allow_pickle=False)
    b=np.load(corrected['prediction_archive'],allow_pickle=False)
    for key in b.files:
        if 'preds' not in key:
            assert np.array_equal(a[key],b[key]),key
    arrays={key:a[key] for key in a.files}
    arrays.update({key:b[key] for key in b.files if 'input_valid_aware' in key and 'preds' in key})
    base=arrays['ensemble_legacy_input_r0.0_preds']
    # Separate fused-CUDA executions have small float32 reduction drift.
    # Require prediction agreement within 1e-4 and unchanged zero-rate classes.
    assert all(np.allclose(base,arrays[f'ensemble_{name}_r0.0_preds'],atol=1e-4)
               for name in old['protocols']), 'Zero-rate identity failed after correction'
    for name in old['protocols']:
        pred=arrays[f'ensemble_{name}_r0.0_preds']
        assert np.array_equal(base>0,pred>0),name
        assert np.array_equal(np.round(np.clip(base,-3,3)),np.round(np.clip(pred,-3,3))),name
    old['protocols']['input_valid_aware']=corrected['protocols']['input_valid_aware']
    for m,updated in zip(old['members'],corrected['members']):
        assert m['checkpoint']==updated['checkpoint']
        m['protocols']['input_valid_aware']=updated['protocols']['input_valid_aware']
    old['protocol_notes']['input_valid_aware']='Same valid native corrupted inputs as input_valid_blind; availability numerator/denominator only count actual valid frames; all-padding pooled windows are neutral dropout placeholders'
    old['correction']='Exclude padding from pooled availability denominator; control predictions/masks and sample order verified unchanged'
    old['zero_rate_max_prediction_delta']=float(max(np.max(np.abs(base-arrays[f'ensemble_{name}_r0.0_preds'])) for name in old['protocols']))
    output=ROOT/'token_protocol_paired_test_v2.json'
    if output.exists():
        raise FileExistsError(output)
    archive=output.with_suffix('.predictions.npz')
    np.savez_compressed(archive,**arrays)
    old['prediction_archive']=str(archive)
    old['mask_archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    output.write_text(json.dumps(old,ensure_ascii=False,indent=2))
    print('Combined final audit:',output)


if __name__=='__main__':main()
