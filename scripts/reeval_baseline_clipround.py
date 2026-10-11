"""Run a native baseline evaluation and archive canonical metrics/predictions.

Use each model's existing Python environment and pass its evaluator arguments
after --. Outputs are new files; checkpoint selection and weights are retained.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'references'))
import metrics_clipround as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--kind', choices=['missing7', 'token'], default='missing7')
    cli, native = parser.parse_known_args()
    if native and native[0] == '--':
        native = native[1:]
    if '--out' not in native:
        parser.error('Native arguments must contain --out')
    out = Path(native[native.index('--out') + 1]).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing evaluation: {out}')
    script = ROOT / 'references/codebases' / cli.model / f'eval_{cli.kind}.py'
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    # Support native wrappers which import the legacy module by name.
    sys.modules['unified_metrics'] = common
    sys.argv = [str(script)] + native
    runpy.run_path(str(script), run_name='__main__')
    result = json.loads(out.read_text())
    section = 'conditions' if cli.kind == 'missing7' else 'rates'
    entries = result[section]
    if len(entries) != len(common.RECORDS):
        raise RuntimeError(f'Prediction capture mismatch: {len(entries)} entries / {len(common.RECORDS)} records')
    arrays = {}
    for (key, entry), record in zip(entries.items(), common.RECORDS):
        entry.update(common.lower_metrics(record['metrics']))
        entry['n_samples'] = len(record['labels'])
        arrays[f'{key}_preds'] = record['preds']
        arrays[f'{key}_labels'] = record['labels']
    selected = list(entries) if cli.kind == 'token' else [k for k in entries if k != 'TAV']
    result['avg_over_rates' if cli.kind == 'token' else 'missing_avg'] = {
        metric: float(np.mean([entries[k][metric] for k in selected]))
        for metric in common.lower_metrics(common.RECORDS[0]['metrics'])}
    prediction_path = out.with_suffix('.predictions.npz')
    np.savez_compressed(prediction_path, **arrays)
    result['metric_protocol'] = {
        'acc7': 'numpy.round(numpy.clip(x,-3,3))',
        'binary': 'Has0 >=0 all labels; Non0 >0 after label!=0; weighted F1',
        'acc7_bins': 'legacy audit only', 'predictions': str(prediction_path),
        'sample_order': 'native test-loader order; arrays paired with labels per condition'}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('Canonical metrics and sample predictions saved:', out)


if __name__ == '__main__':
    main()
