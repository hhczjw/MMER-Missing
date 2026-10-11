"""One real MOSI training step to verify valid-only masks and gradients."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import _path_setup
from reeval_token_protocols import configuration
from dataset.data_loader import MMDataLoader
from models import UMRMv3
from scripts.train import train_epoch, TaskLoss, evaluate_token_missing
from utils.token_protocol import validity, align_validity


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--output',required=True)
    cli=ap.parse_args()
    args=configuration()
    args.batch_size=2
    args.use_curriculum=False
    args.model_name='umrmv3'
    args.token_eval_protocol='feature_valid'
    device=torch.device(cli.device)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    model=UMRMv3(args).to(device)
    model.token_missing_rate=1.0
    model.modality_missing_rate=0.0
    batch=next(iter(MMDataLoader(args,'train')))
    captured=[]
    hook=model.register_forward_hook(lambda m,i,o:captured.append(o['token_mask'].detach().cpu()))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    probe=model.proj_audio.weight
    before=probe.detach().clone()
    criterion=TaskLoss(task_type=args.task_type,num_classes=args.num_classes)
    loss=train_epoch(model,[batch],criterion,optimizer,args,device,1,60)
    hook.remove()
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before,probe.detach()), 'No parameter update'
    valid=align_validity(validity(batch['text'],batch['audio'],batch['vision'],
                                 batch['audio_valid_lengths'],batch['vision_valid_lengths']),batch['text'].shape[-1])
    student=captured[-1]
    assert bool((student[~valid]==1).all()),'Padding was marked missing'
    indices=torch.arange(valid.shape[-1]).expand_as(valid)
    first=indices.masked_fill(~valid,valid.shape[-1]).amin(-1)
    last=indices.masked_fill(~valid,-1).amax(-1)
    ends=(indices==first[...,None])|(indices==last[...,None])
    assert bool((student[ends]==1).all()),'Real endpoints were masked'
    assert bool((student[valid & ~ends]==0).all()),'Rate-one non-special valid positions were not missing'
    metrics=evaluate_token_missing(model,[batch],args,device,rates=[0.,.5,.9])
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics['avg_over_rates'].values())
    result={'status':'passed','smoke_only':True,'one_training_step':True,'loss':loss,
            'parameter_updated':True,'padding_excluded':True,'actual_endpoints_preserved':True,
            'student_missing_mask_shape':list(student.shape),'evaluation_protocol':metrics['protocol']}
    output=Path(cli.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(result)


if __name__=='__main__':main()
