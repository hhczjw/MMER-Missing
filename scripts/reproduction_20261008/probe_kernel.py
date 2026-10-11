"""Small repeatability probe for the shared SISO kernel; no model/data mutation."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import _path_setup
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from queue_guard import acquire_idle_gpus


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu0-concurrent-r2',action='store_true')
    args=p.parse_args()
    if args.gpu0_concurrent_r2:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
            raise RuntimeError('Concurrent probe requires physical GPU0')
        sys.path.insert(0,str(ROOT/'scripts/reproduction_20261009'))
        from verify_r1_checkpoint import acquire_gpu0_validation
        gpu_locks=acquire_gpu0_validation()
    else:
        gpu_locks=acquire_idle_gpus()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    torch.cuda.set_device(0);torch.manual_seed(2024)
    B,L,H,N,P=8,50,8,64,32
    inputs=dict(Q=torch.randn(B,L,1,N,device='cuda')*.1,K=torch.randn(B,L,1,N,device='cuda')*.1,
        V=torch.randn(B,L,H,P,device='cuda')*.1,ADT=-torch.rand(B,H,L,device='cuda')*.01,
        DT=torch.rand(B,H,L,device='cuda')*.01,Trap=torch.randn(B,H,L,device='cuda'),
        Q_bias=torch.randn(H,N,device='cuda')*.1,K_bias=torch.randn(H,N,device='cuda')*.1,
        Angles=torch.randn(B,L,H,16,device='cuda')*.1,D=torch.randn(H,device='cuda')*.1,
        Z=torch.randn(B,L,H,P,device='cuda')*.1)
    for x in inputs.values():x.requires_grad_()
    upstream=torch.randn(B,L,H,P,device='cuda',dtype=torch.bfloat16)
    rows=[];reference=None
    for trial in range(12):
        output=mamba3_siso_combined(**inputs,chunk_size=64,Input_States=None,
                                  return_final_states=False,cu_seqlens=None)
        gradients=torch.autograd.grad(output,tuple(inputs.values()),upstream)
        values=(output.detach().cpu(),{n:g.detach().cpu() for n,g in zip(inputs,gradients)})
        if reference is None:reference=values
        row=dict(trial=trial,forward_max_abs=float((reference[0].float()-values[0].float()).abs().max()),
                 gradient_max_abs={n:float((reference[1][n].float()-g.float()).abs().max())
                                   for n,g in values[1].items()})
        rows.append(row);print(row,flush=True)
    result=dict(status='complete',torch=torch.__version__,device=torch.cuda.get_device_name(),
                shape=[B,L,H,N,P],input_states=False,return_final_states=False,rows=rows,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
