"""Device-only adapter for the original frozen R2 ablation worker."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'scripts/research2_ablations/worker.py'


def adapted_source(source):
    edits = {
        "assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'":
            "assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'",
        "pid=os.getpid(),physical_gpu=1,": "pid=os.getpid(),physical_gpu=0,",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError('Original device assertion/metadata changed; re-review required')
        source = source.replace(old, new)
    return source


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('GPU0 worker requires physical GPU0 visibility')
    sys.path.insert(0, str(ORIGINAL.parent))
    source = adapted_source(ORIGINAL.read_text())
    namespace = dict(__name__='__main__', __file__=str(ORIGINAL), __package__=None)
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)


if __name__ == '__main__':
    main()
