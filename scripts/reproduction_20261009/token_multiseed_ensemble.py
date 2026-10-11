"""CPU-only fixed six-seed prediction average, using the frozen native metric function."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SEEDS=[42,43,44,45,46,47]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);cli=ap.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
spec=importlib.util.spec_from_file_location('native_metrics',ROOT/'experiments/research2_legacy_replay_20261009/source/utils/metrics.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
plan=json.loads((cli.output/'plan.json').read_text())
assert plan['ensemble']['seeds']==SEEDS and plan['ensemble']['weights']==[1/6]*6
members=[json.loads((cli.output/f'seed{s}/test.json').read_text()) for s in SEEDS]
archives=[np.load(cli.output/f'seed{s}/test.predictions.npz',allow_pickle=False) for s in SEEDS]
try:
    labels=archives[0]['labels'];ids=archives[0]['sample_ids']
    assert all(np.array_equal(a['labels'],labels) and np.array_equal(a['sample_ids'],ids) for a in archives)
    assert len(labels)==686
    arrays={'labels':labels,'sample_ids':ids};curve={}
    for i in range(10):
        rate=f'{i/10:.1f}';key=f'r{rate}_preds'
        predictions=np.stack([a[key] for a in archives])
        assert np.isfinite(predictions).all()
        # Recompute every saved single result before doing the fixed average.
        for j,member in enumerate(members):
            recomputed=metrics.eval_regression(predictions[j],labels)
            assert all(np.isclose(v,member['per_rate'][rate][k],rtol=1e-6,atol=1e-7) for k,v in recomputed.items())
        arrays[f'r{rate}_member_preds']=predictions
        arrays[key]=predictions.mean(0)
        curve[rate]={k:float(v) for k,v in metrics.eval_regression(arrays[key],labels).items()}
    archive=cli.output/'ensemble.predictions.npz'
    if archive.exists() or (cli.output/'ensemble.json').exists():raise FileExistsError('Ensemble output already exists')
    np.savez_compressed(archive,**arrays)
    avg={k:float(np.mean([x[k] for x in curve.values()])) for k in next(iter(curve.values()))}
    report={'dataset':'MOSI','protocol':'input_valid_aware','seeds':SEEDS,'weights':[1/6]*6,
        'checkpoint_selection':'per-seed validation Token10 mean Acc7; MAE tie-break',
        'test_mask_seed':20261005,'n_samples':len(labels),'per_rate':curve,'avg_over_rates':avg,
        'prediction_archive':str(archive),'members':members,'single_metrics_recomputed':True,
        'seed45_reused_from':plan['reuse45'],'test_selected_subset':False}
    with (cli.output/'ensemble.json').open('x') as stream:json.dump(report,stream,indent=2)
    with (cli.output/'single_seed_summary.json').open('x') as stream:json.dump({str(s):m['avg_over_rates'] for s,m in zip(SEEDS,members)},stream,indent=2)
    print('FIXED_SIX_SEED_ENSEMBLE',json.dumps(avg),flush=True)
finally:
    for archive in archives:archive.close()
