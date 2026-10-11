"""GPU0-only fixed-input FP32 SDPA repeatability probe; no model training."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from verify_r1_checkpoint import acquire_gpu0_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('GPU0 only')
    locks = acquire_gpu0_validation()
    try:
        if cli.output.exists():
            raise FileExistsError(cli.output)
        torch.manual_seed(2024)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        shape = (4, 16, 50, 64)
        q, k, v = [torch.randn(shape, device='cuda', requires_grad=True) for _ in range(3)]
        upstream = torch.randn(shape, device='cuda')
        mask = torch.zeros(4, 1, 1, 50, device='cuda')
        mask[..., -10:] = -torch.inf
        rng = torch.cuda.get_rng_state()
        report = dict(shape=shape, dtype='FP32', fixed_rng=True, dropout=.1, modes={})
        for mode in ['default', 'math']:
            rows = []
            reference = None
            for trial in range(12):
                torch.cuda.set_rng_state(rng)
                context = sdpa_kernel(SDPBackend.MATH) if mode == 'math' else nullcontext()
                with context:
                    output = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                        attn_mask=mask, dropout_p=.1, is_causal=False)
                    gradients = torch.autograd.grad(output, (q, k, v), upstream)
                values = [x.detach().cpu() for x in (output, *gradients)]
                if reference is None:
                    reference = values
                rows.append(dict(trial=trial, forward_max_abs=float((values[0]-reference[0]).abs().max()),
                    gradient_max_abs=[float((x-y).abs().max()) for x, y in zip(values[1:],reference[1:])]))
            report['modes'][mode] = rows
            print(mode, 'forward_max', max(r['forward_max_abs'] for r in rows),
                  'gradient_max', max(max(r['gradient_max_abs']) for r in rows), flush=True)
        cli.output.write_text(json.dumps(report, indent=2))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
