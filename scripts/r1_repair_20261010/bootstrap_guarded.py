"""Safety wrapper around the pinned original ablation bootstrap; no source edits."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/r1_aligned_20261009'))
from plan import BASE, read, write
from numerical_guard import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    sys.path.insert(0, str(BASE/'source'))
    import trainer
    install(trainer, cli.output, write)
    write(cli.output/'numerical_policy.json', dict(
        finite_path='unchanged native forward/backward/clip/AdamW/scheduler; additional read-only checks',
        repair='fail on nonfinite loss/logits or total gradient norm before parameter update; capture state',
        no_batch_skip=True, no_retry=True, no_reference_backward=True,
        no_lr_batch_precision_optimizer_change=True,
        originating_fault='unresolved; cannot claim a kernel or GLCE removal root cause yet'))
    runpy.run_path(str(ROOT/'scripts/r1_aligned_20261009/bootstrap.py'), run_name='__main__')


if __name__ == '__main__':
    main()
