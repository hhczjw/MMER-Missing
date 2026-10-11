"""Read-only gradient diagnosis using the failed run's validation-selected weights."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/ablations_p0b_0919'))
from worker import configure,load_backend,make_loader
from plan import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);cli=ap.parse_args()
    cli.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'results/mosi_ablations_p0b_0919_gpu0_20261006_v2/plan.json').read_text())
    args=configure('R1',2024,120,{'gcmn':False},manifest['reference_args'])
    builder,forward,Dataset,collate=load_backend('R1')
    from utils.seed import set_seed
    set_seed(2024);torch.set_num_threads(4);torch.cuda.set_device(0)
    loader,stats=make_loader(args,'train',Dataset,collate,'R1',4)
    model=builder(args,{'gcmn':False}).cuda()
    cp=Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints/mosi_ablations_p0b_0919_gpu0_20261006_v2/train/R1-04_seed2024/best_validation.pth')
    state=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(state['model'],strict=True);del state
    model.train();records=[];start=time.time()
    for i,batch in enumerate(loader):
        model.zero_grad(set_to_none=True)
        out=forward(model,batch,torch.device('cuda:0'))
        label=batch['labels']['M'].cuda().reshape(-1)
        loss=torch.nn.functional.smooth_l1_loss(out['logits'].reshape(-1),label,beta=.5)
        loss.backward()
        grads=[(n,p.grad) for n,p in model.named_parameters() if p.grad is not None]
        norms=torch.stack([torch.linalg.vector_norm(g.detach()) for _,g in grads])
        norm=torch.linalg.vector_norm(norms)
        row=dict(batch=i,loss=float(loss.detach()),norm=float(norm),seconds=time.time()-start)
        if not torch.isfinite(norm):
            bad=[];extreme=[]
            for n,g in grads:
                valid=torch.isfinite(g)
                if not valid.all():bad.append(dict(name=n,nan=int(torch.isnan(g).sum()),inf=int(torch.isinf(g).sum())))
                extreme.append((float(g.detach().abs().max()),n))
            row.update(nonfinite_parameters=bad,largest_gradients=sorted(extreme,reverse=True)[:12],
                       classification='nonfinite_gradient_elements' if bad else 'finite_gradient_norm_overflow')
            atomic_json(cli.output/'failure.json',row)
            torch.save({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in batch.items()},cli.output/'batch.pt')
            print('BAD_GRADIENT',json.dumps(row),flush=True);break
        if i%25==0:print('PROBE',i,'norm',float(norm),flush=True)
        records.append(row)
    atomic_json(cli.output/'summary.json',dict(completed_batches=len(records),bad_batch_found=(cli.output/'failure.json').exists(),
         checkpoint=str(cp),weights_updated=False,seconds=time.time()-start,records=records))


if __name__=='__main__':main()
