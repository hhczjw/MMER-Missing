"""Reuse original seed42 bootstrap; isolated run02 output only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_p0b_seed42 as native

native.OUTPUT = Path(__file__).resolve().parents[2] / 'results/p0B_legacy_replay_20261009_seed42/run_02'

if __name__ == '__main__':
    native.main()
