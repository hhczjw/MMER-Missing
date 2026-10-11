"""Freeze an explicitly authorized native R2 replay candidate, without CUDA."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/research2_legacy_replay_20261009'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    historical = json.loads((ROOT / 'results/token_input_retrain_20261006/run.json').read_text())
    for name, expected in historical['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Historical Token core changed: {name}')
    source = BASE / 'source'
    source.mkdir(parents=True, exist_ok=False)
    files = [ROOT / '_path_setup.py', ROOT / 'config.py']
    for folder in ['models', 'layers', 'utils', 'dataset', 'losses', 'configs']:
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json'))
    scripts = ['train.py', 'train_token_input.py', 'reeval_token_protocols.py',
               'reeval_ensemble_token.py', 'reeval_ensemble.py']
    files.extend(ROOT / 'scripts' / name for name in scripts)
    files.extend((ROOT / 'mamba/mamba_ssm').rglob('*.py'))
    files.extend((ROOT / 'mamba').glob('*.so'))
    captured = {}
    for path in sorted(set(files)):
        name = str(path.relative_to(ROOT))
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(path)
        shutil.copy2(path, target)
        if sha(target) != before or sha(path) != before:
            raise RuntimeError(f'Concurrent source change: {name}')
        captured[name] = dict(sha256=before, origin=str(path))
    cache = ROOT / 'results/checkpoint_nvme_staging_20261007/input_cache/index.json'
    shutil.copy2(cache, BASE / 'input_cache_index.json')
    shutil.copy2(ROOT / 'scripts/checkpoint_nvme_20261007/data_cache.py', BASE / 'data_cache.py')
    evidence = BASE / 'historical_evidence'
    evidence.mkdir()
    for name, path in {
        'token_run.json': ROOT / 'results/token_input_retrain_20261006/run.json',
        'token_seed45_config.json': ROOT / 'results/token_input_retrain_20261006/seed45/config.json',
        'token_seed45_test.json': ROOT / 'results/token_input_retrain_20261006/seed45/test.json',
        'missing_ensemble.json': ROOT / 'results/NEW3_missing_s43s45s48_MAEckpt.json',
        'full_ensemble.json': ROOT / 'results/NEW3_TAV_s44s42s43_MAEckpt.json',
        'missing_launch.sh': ROOT / 'scripts/run_acc7clip_retrain.sh',
        'BEST_CONFIG.md': ROOT / 'references/BEST_CONFIG.md',
    }.items():
        shutil.copy2(path, evidence / name)
    r1 = json.loads((ROOT / 'experiments/p0B_legacy_replay_20261008/recovery_manifest.json').read_text())
    cache_mapping = json.loads(cache.read_text())['mapping']
    data = '/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets/CMU-MOSI/Processed/unaligned_50_roberta.pkl'
    local = Path(cache_mapping[data]['local']) if data in cache_mapping else Path(data)
    hub = Path('/home/zjw/.cache/huggingface/hub/models--FacebookAI--roberta-large')
    revision = (hub / 'refs/main').read_text().strip()
    weights = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size)
               for p in (hub / 'snapshots' / revision).iterdir() if p.is_file()}
    manifest = dict(created_at=datetime.now().astimezone().isoformat(),
        restoration_scope='User-authorized native candidate; not a complete dated historical environment/source snapshot',
        files=captured, helper_sha256={str(p.relative_to(BASE)): sha(p) for p in BASE.glob('*.py')},
        data_sha256={data: sha(local)}, data_local_path=str(local),
        pretrained_files=weights, pretrained_revision=revision,
        historical_token_core_sha256=historical['source_sha256'],
        expected_environment=r1['expected_current_environment'], gpu1_uuid=r1['current_gpu1_uuid'],
        inherited_overrides_must_be_unset=r1['expected_unset_environment'],
        authority='2026-10-09 user: 是的，除了研究一的，研究二的也进行最优结果的重训。',
        unknowns=['Historical physical GPU UUID/complete dependency versions unproven',
                  'R2-M full dated training-source snapshot unavailable; archived best-result snapshot equals current checked core',
                  'R2 replay uses GPU1/current coupled_mamba_3, does not assert exact historical physical-device equality'],
        training='Unmodified native train_token_input.py and scripts/train.py; FP32; LoRA8/16; fresh pretrained text only; no old MMER checkpoint resume',
        token=dict(seed=45, epochs=60, batch=8, eval_batch=16, lr=.0003,
                   min_epochs=20, patience=15, selection='valid Token10 mean Acc7, MAE tie-break'),
        modality=dict(seeds=[43,45,48,42,44], epochs=60, batch=8, eval_batch=8,
                      selection='valid TAV MAE; native warmup, curriculum, KD and early-stop',
                      reporting_only_change='skip-extra-eval omits unrelated Token/joint-grid reporting; training unchanged'),
        ensembles=dict(missing=[43,45,48], full=[44,42,43], weights='fixed equal prediction average, not metric mean'))
    with (BASE / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print('Frozen', len(captured), 'native source/config/binary files;', BASE)


if __name__ == '__main__':
    main()
