"""Repeat actual BiMamba selective-scan primitive vs its PyTorch reference."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from verify_r1_checkpoint import acquire_gpu0_validation

ROOT = Path(__file__).resolve().parents[2]


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
        source = ROOT / 'experiments/p0B_legacy_replay_20261008/source'
        sys.path.insert(0, str(source))
        import _path_setup
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        torch.manual_seed(2024)
        batch, channels, length, states = 4, 256, 51, 32
        tensors = dict(u=torch.randn(batch, channels, length, device='cuda')*.1,
            delta=torch.randn(batch, channels, length, device='cuda')*.1,
            A=-torch.exp(torch.randn(channels, states, device='cuda')*.1),
            B=torch.randn(batch, states, length, device='cuda')*.1,
            C=torch.randn(batch, states, length, device='cuda')*.1,
            D=torch.randn(channels, device='cuda')*.1,
            z=torch.randn(batch, channels, length, device='cuda')*.1,
            delta_bias=torch.randn(channels, device='cuda')*.1)
        for value in tensors.values():
            value.requires_grad_()
        upstream = torch.randn(batch, channels, length, device='cuda')
        report = dict(shape=[batch, channels, length, states], dtype='FP32',
                      variable_BC=True, group_count=1, modes={}, reference_agreement={})
        first_native = None
        for name, function in [('cuda', selective_scan_fn), ('torch_reference', selective_scan_ref)]:
            first = None
            rows = []
            for trial in range(12):
                output = function(**tensors, delta_softplus=True)
                gradients = torch.autograd.grad(output, tuple(tensors.values()), upstream)
                values = [value.detach().cpu().clone() for value in (output, *gradients)]
                if first is None:
                    first = values
                    if name == 'cuda':
                        first_native = values
                    else:
                        report['reference_agreement'] = dict(
                            forward_max_abs=float((values[0]-first_native[0]).abs().max()),
                            gradients={key:float((x-y).abs().max()) for key,x,y in
                                       zip(tensors,values[1:],first_native[1:])})
                row = dict(trial=trial, forward_max_abs=float((values[0]-first[0]).abs().max()),
                    gradients={key:float((x-y).abs().max()) for key,x,y in
                               zip(tensors,values[1:],first[1:])})
                rows.append(row)
            report['modes'][name] = rows
            maximum = {key:max(row['gradients'][key] for row in rows) for key in tensors}
            print(name, maximum, flush=True)
        cli.output.write_text(json.dumps(report, indent=2))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
