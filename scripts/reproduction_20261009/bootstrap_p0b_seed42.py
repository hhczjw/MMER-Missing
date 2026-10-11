"""Same frozen native R1 entry, seed42 and isolated output paths only."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/p0B_legacy_replay_20261008'
OUTPUT = ROOT / 'results/p0B_legacy_replay_20261009_seed42/run_01'


def output_config(args):
    for name, folder in [('checkpoints_dir', 'checkpoints'), ('logs_dir', 'logs'),
                         ('results_dir', 'results')]:
        setattr(args, name, str(OUTPUT / folder))
    return args


def main():
    sys.path.insert(0, str(BASE))
    from data_cache import install_cache
    if not install_cache(BASE / 'input_cache_index.json'):
        raise RuntimeError('Verified input cache unavailable')
    sys.path.insert(0, str(BASE / 'source'))
    import configs
    native_load = configs.load_config
    configs.load_config = lambda dataset: output_config(native_load(dataset))
    sys.argv = [str(BASE / 'source/train.py'), '--dataset', 'MOSI',
                '--seed', '42', '--exp_tag', 'p0B']
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
