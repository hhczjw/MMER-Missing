"""Diagnostic-only substitution: native forward, PyTorch reference scan gradients."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def reference_scan_backward(stats, include_conv=False):
    from mamba_ssm.ops import selective_scan_interface as interface
    extension = interface.selective_scan_cuda
    original = extension.bwd
    original_conv = interface.causal_conv1d_bwd_function

    def backward(*args):
        native = list(original(*args))
        source = args[:8]  # u, delta, A, B, C, D, z, delta_bias
        if source[2].is_complex():
            raise ValueError('Diagnostic supports the actual real-valued p0B scan only')
        with torch.enable_grad():
            detached = [x.detach().requires_grad_(True) if x is not None else None for x in source]
            output = interface.selective_scan_ref(*detached, delta_softplus=args[12])
            present = [(i,x) for i,x in enumerate(detached) if x is not None]
            gradients = torch.autograd.grad(output, [x for _,x in present], args[8])
        destinations = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:7, 7:6}
        for (i,_), grad in zip(present, gradients):
            index = destinations[i]
            error = float((native[index].float()-grad.float()).abs().max())
            stats['max_native_reference_gradient_error'] = max(
                stats.get('max_native_reference_gradient_error', 0.), error)
            # Preserve preallocated dz aliasing required by the fused caller.
            native[index].copy_(grad)
        stats['calls'] = stats.get('calls',0)+1
        return native

    extension.bwd = backward
    def conv_backward(*args):
        if any(x is not None for x in args[4:7]) or args[8]:
            raise ValueError('Reference convolution supports the actual p0B stateless path only')
        native = list(original_conv(*args))
        with torch.enable_grad():
            x, weight, bias = [x.detach().requires_grad_(True) if x is not None else None for x in args[:3]]
            output = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1]-1,
                              groups=x.shape[1])[..., :x.shape[-1]]
            if args[9]:
                output = F.silu(output)
            present = [(i,x) for i,x in enumerate([x,weight,bias]) if x is not None]
            gradients = torch.autograd.grad(output, [x for _,x in present], args[3])
        for (i,_), gradient in zip(present, gradients):
            stats['max_native_reference_conv_gradient_error'] = max(
                stats.get('max_native_reference_conv_gradient_error',0.),
                float((native[i].float()-gradient.float()).abs().max()))
            native[i].copy_(gradient)
        stats['conv_calls'] = stats.get('conv_calls',0)+1
        return tuple(native)
    if include_conv:
        interface.causal_conv1d_bwd_function = conv_backward
    try:
        yield
    finally:
        extension.bwd = original
        interface.causal_conv1d_bwd_function = original_conv
