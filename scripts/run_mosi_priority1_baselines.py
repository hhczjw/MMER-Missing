"""Re-evaluate existing MOSI baseline weights, with resumable local outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing/references/codebase')
OUT = ROOT / 'results/priority1_20261006/baselines'


def jobs():
    for model, folder, env in [('LNLN', 'lnln', 'noise'), ('HGDNet', 'hgdnet', 'coupled-mamba'),
                               ('IIE', 'IIE', 'dmd'), ('TLRA', 'tlra', 'noise'),
                               ('MPLMM', 'mplmm', 'dmd'), ('FSRF', 'FSRF', 'noise'),
                               ('MITR-DNet', 'mitr', 'dmd')]:
        old = json.loads((EXTERNAL / 'results' / folder / 'MOSI/metrics_missing7.json').read_text())
        ckpt = old['ckpt']
        if model == 'LNLN':
            args = ['--config_file', 'configs/train_mosi.yaml', '--ckpt', ckpt]
        elif model == 'HGDNet':
            args = ['--ckpt', ckpt, '--dataset', 'CMUMOSI']
        elif model == 'IIE':
            args = ['--dataset', 'mosi', '--ckpt', ckpt]
        elif model == 'TLRA':
            args = ['--dataset', 'mosi', '--ckpt', ckpt]
        elif model == 'MPLMM':
            args = ['--dataset', 'mosi', '--ckpt', ckpt, '--data_path',
                    '/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets/CMU-MOSI/Processed/unaligned_50.pkl']
        elif model == 'FSRF':
            path = Path(ckpt)
            args = ['--data', 'mosi', '--checkpoint_dir', str(path.parent),
                    '--checkpoint_name', path.stem.removeprefix('model_'), '--use_bert', 'False']
        else:
            args = ['--datasetName', 'mosi', '--ckpt', ckpt, '--gpu', '0']
        yield model, 'missing7', env, args
    yield 'LNLN', 'token', 'noise', ['--config_file', 'configs/train_mosi.yaml', '--ckpt',
                                   str(EXTERNAL / 'LNLN/MOSI/best_MAE_1111.pth')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    cli = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cli.gpu,
                       FSRF_DATA_ROOT=str(EXTERNAL / 'data_fsrf'), IIE_DEVICE='cuda:0',
                       HGDNET_MODEL_DIR=str(EXTERNAL / 'HGDNet'))
    status = []
    for model, kind, env, args in jobs():
        output = OUT / f'{model}_{kind}_clipround.json'
        if output.exists() and 'metric_protocol' in json.loads(output.read_text()):
            print('ALREADY COMPLETE', model, kind, flush=True)
            status.append({'model': model, 'kind': kind, 'status': 'passed', 'output': str(output)})
            continue
        command = [f'/home/zjw/anaconda3/envs/{env}/bin/python', '-u',
                   str(ROOT / 'scripts/reeval_baseline_clipround.py'), '--model', model,
                   '--kind', kind, '--'] + args + ['--out', str(output)]
        log = OUT / f'{model}_{kind}.log'
        print('START', model, kind, flush=True)
        with log.open('w') as handle:
            result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        status.append({'model': model, 'kind': kind, 'status': 'passed' if result.returncode == 0 else 'failed',
                       'exit_code': result.returncode, 'output': str(output), 'log': str(log)})
        (OUT / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print('FINISH', model, kind, result.returncode, flush=True)
        if result.returncode:
            print(log.read_text()[-3000:], flush=True)
    (OUT / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    if any(s['status'] == 'failed' for s in status):
        sys.exit(1)


if __name__ == '__main__':
    main()
