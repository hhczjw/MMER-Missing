"""CPU clipping equivalence and GPU reference-gradient regression evidence."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=ROOT/'experiments/ablation_p0b_0919_gradient_repair_20261007'
sys.path.insert(0,str(EXPERIMENT/'runner'))
from gradient_safety import safe_clip,InvalidGradientError
from plan import atomic_json


def relative_rms(a,b):
    a,b=a.detach().float(),b.detach().float()
    return float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))


def main():
    torch.manual_seed(100)
    for scale in [.01,1.,100.]:
        a=torch.nn.Parameter(torch.zeros(128));b=torch.nn.Parameter(torch.zeros(128))
        gradient=torch.randn(128)*scale;a.grad=gradient.clone();b.grad=gradient.clone()
        before=torch.nn.utils.clip_grad_norm_([a],.1,error_if_nonfinite=True)
        after,diagnostic=safe_clip([('p',b)],.1)
        assert diagnostic is None and torch.equal(before,after) and torch.equal(a.grad,b.grad)
    p=torch.nn.Parameter(torch.zeros(2));p.grad=torch.tensor([1e20,-1e20])
    _,overflow=safe_clip([('p',p)],.1)
    assert overflow['kind']=='finite_gradient_norm_overflow' and torch.isfinite(p.grad).all()
    assert torch.allclose(p.grad.norm(),torch.tensor(.1),atol=1e-7)
    p.grad=torch.tensor([float('nan'),1.])
    try:safe_clip([('p',p)],.1)
    except InvalidGradientError:pass
    else:raise AssertionError('Invalid gradient was accepted')
    sys.path.insert(0,str(EXPERIMENT/'snapshot_r1'));import _path_setup
    from reference_backward import recurrent
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    torch.cuda.set_device(0);torch.manual_seed(100)
    B,L,H,N,P=2,50,8,64,32
    inputs=dict(Q=torch.randn(B,L,1,N,device='cuda')*.1,K=torch.randn(B,L,1,N,device='cuda')*.1,
        V=torch.randn(B,L,H,P,device='cuda')*.1,ADT=-torch.rand(B,H,L,device='cuda')*.01,
        DT=torch.rand(B,H,L,device='cuda')*.01,Trap=torch.randn(B,H,L,device='cuda'),
        Q_bias=torch.randn(H,N,device='cuda')*.1,K_bias=torch.randn(H,N,device='cuda')*.1,
        Angles=torch.randn(B,L,H,16,device='cuda')*.1,D=torch.randn(H,device='cuda')*.1,
        Z=torch.randn(B,L,H,P,device='cuda')*.1)
    for x in inputs.values():x.requires_grad_()
    native=mamba3_siso_combined(**inputs,chunk_size=64,Input_States=None,
                              return_final_states=False,cu_seqlens=None)
    ref=recurrent(**inputs)
    forward_error=relative_rms(native,ref);assert forward_error<.02
    upstream=torch.randn_like(native)
    native_grads=torch.autograd.grad(native,tuple(inputs.values()),upstream)
    reference_grads=torch.autograd.grad(ref,tuple(inputs.values()),upstream)
    errors={name:relative_rms(a,b) for name,a,b in zip(inputs,native_grads,reference_grads)}
    assert all(torch.isfinite(g).all() for g in reference_grads)
    assert all(error<.05 for error in errors.values()),errors
    report=dict(status='passed',normal_clipping_bit_identical=True,
        finite_norm_overflow_recovered=True,actual_nan_rejected=True,
        kernel_inputs_follow_native_BF16_casts=True,reference_state_precision='FP64',
        kernel_forward_relative_rms=forward_error,kernel_backward_relative_rms=errors,
        forward_tolerance=.02,backward_tolerance=.05,
        note='Reference mathematical recurrence is not bit-identical to fused backward; native forward is retained during retry.')
    output=ROOT/'results/mosi_ablations_p0b_0919_gradient_repair_20261007/repair_checks'
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'numerical_regression.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
