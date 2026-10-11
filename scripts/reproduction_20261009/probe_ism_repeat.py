"""Fixed-input repeatability of the actual p0B ISM encoder on physical GPU0."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import torch
from verify_r1_checkpoint import acquire_gpu0_validation

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/ablation_p0b_0919_gradient_repair_20261007/snapshot_r1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', choices=['ism','block','mixer','mixer_forward','mixer_reverse'], default='ism')
    parser.add_argument('--reference-scan-backward', action='store_true')
    parser.add_argument('--reference-conv-backward', action='store_true')
    cli = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('GPU0 only')
    locks = acquire_gpu0_validation()
    try:
        if cli.output.exists():
            raise FileExistsError(cli.output)
        sys.path.insert(0, str(SOURCE))
        import _path_setup
        from layers.ism import ISMEncoder
        torch.manual_seed(2024)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = ISMEncoder(d_model=128, seq_len=50, depth=1, d_state=32,
                           dropout=.6, mixer_type='bimamba', use_bssm_gate=True).cuda().train()
        length = 50 if cli.component == 'ism' else 51
        if cli.component == 'block':
            model = model.layers[0]
        elif cli.component.startswith('mixer'):
            model = model.layers[0].mixer
        x = torch.randn(4, length, 128, device='cuda', requires_grad=True)
        upstream = torch.randn_like(x)
        parameters = list(model.named_parameters())
        rng = torch.cuda.get_rng_state()
        reference = None
        rows = []
        substitution = {}
        for trial in range(12):
            torch.cuda.set_rng_state(rng)
            if cli.component == 'ism':
                output, cls = model(x, return_cls=True)
            elif cli.component == 'block':
                output, residual = model(x)
            elif cli.component in ('mixer_forward','mixer_reverse'):
                from einops import rearrange
                from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
                m = model.mamba
                projected = rearrange(m.in_proj.weight @ rearrange(x, 'b l d -> d (b l)'),
                                      'd (b l) -> b d l', l=length)
                reverse = cli.component == 'mixer_reverse'
                if reverse:
                    projected = projected.flip([-1])
                conv = m.conv1d_b if reverse else m.conv1d
                xp = m.x_proj_b if reverse else m.x_proj
                dt = m.dt_proj_b if reverse else m.dt_proj
                A = m.A_b_log if reverse else m.A_log
                D = m.D_b if reverse else m.D
                output = mamba_inner_fn_no_out_proj(projected, conv.weight, conv.bias,
                    xp.weight, dt.weight, -torch.exp(A.float()), None, None, D.float(),
                    delta_bias=dt.bias.float(), delta_softplus=True)
                output = rearrange(output, 'b d l -> b l d')
                if output.shape != upstream.shape:
                    upstream = torch.ones_like(output)
            else:
                output = model(x)
            from selective_scan_reference_backward import reference_scan_backward
            context = reference_scan_backward(substitution, include_conv=cli.reference_conv_backward) if cli.reference_scan_backward else nullcontext()
            with context:
                gradients = torch.autograd.grad(output, [x, *[p for _, p in parameters]],
                                                upstream, allow_unused=True)
            values = {name: grad.detach().cpu().clone() for name, grad in
                      zip(['input', *[n for n, _ in parameters]], gradients) if grad is not None}
            prediction = output.detach().cpu().clone()
            if reference is None:
                reference = prediction, values
            errors = {name: float((value-reference[1][name]).abs().max())
                      for name, value in values.items()}
            row = dict(trial=trial, forward_max_abs=float((prediction-reference[0]).abs().max()),
                       gradient_max_abs=errors)
            rows.append(row)
            print(trial, 'forward', row['forward_max_abs'], 'gradient', max(errors.values()), flush=True)
        cli.output.write_text(json.dumps(dict(shape=list(x.shape), component=cli.component,
            model='p0B ISM depth1 state32 BiMamba BSSM',
            reference_scan_backward=cli.reference_scan_backward, substitution=substitution,
            fixed_rng=True, rows=rows), indent=2))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
