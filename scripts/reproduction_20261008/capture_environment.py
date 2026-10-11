"""Read-only environment evidence collection; deliberately no CUDA initialization."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    packages = sorted([dict(name=d.metadata['Name'], version=d.version)
                       for d in metadata.distributions()], key=lambda x: x['name'].lower())
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap',
        '--format=csv'
    ], text=True, timeout=10)
    allowed = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTORCH_CUDA_ALLOC_CONF',
               'PYTORCH_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS',
               'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'NVIDIA_TF32_OVERRIDE',
               'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'PYTHONHASHSEED']
    sources = {}
    for folder in [ROOT / 'mamba/mamba_ssm',
                   Path('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/mamba/mamba_ssm')]:
        for path in sorted(folder.rglob('*.py')):
            sources[str(path)] = sha(path)
    conda_history = Path(sys.prefix) / 'conda-meta/history'
    report = dict(captured_at=datetime.now().isoformat(),
                  evidence_scope='current state only, not proof of historical installed versions',
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=packages, torch_cuda_runtime=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(),
                  default_precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                         cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                         cudnn_deterministic=torch.backends.cudnn.deterministic,
                                         cudnn_benchmark=torch.backends.cudnn.benchmark,
                                         algorithms_deterministic=torch.are_deterministic_algorithms_enabled()),
                  note='Fresh CPU-only process defaults: final training settings must be captured after native seed setup.',
                  environment={k:os.environ.get(k) for k in allowed},
                  gpu_inventory=gpu, mamba_source_sha256=sources,
                  conda_history_sha256=sha(conda_history), cuda_initialized=torch.cuda.is_initialized())
    assert not report['cuda_initialized'], 'Environment probe must not initialize CUDA'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(args.output)


if __name__ == '__main__':
    main()
