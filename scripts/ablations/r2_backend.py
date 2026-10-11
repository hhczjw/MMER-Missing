"""Loss-separated clean-view training for isolated MMER ablations."""
import torch
import torch.nn.functional as F
from models import UMRMv3
from utils.token_protocol import validity,native_corruption,aligned_availability,feature_corruption_masks

PATTERNS = [[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1]]


def build_model(args, changes):
    model = UMRMv3(args)
    model.ablation_ume_enabled=changes.get('ume',True)
    model.ablation_enhance_enabled=changes.get('enhance',True)
    model.ablation_enhanced_bypass=changes.get('enhanced_bypass',True)
    model.contrastive_reconstruction.ablation_reconstruction_weight=changes.get('feature_rec_weight',1.0)
    if not model.ablation_ume_enabled:model.missing_encoder.requires_grad_(False)
    if not model.ablation_enhance_enabled:
        for m in model.missing_enhancers:m.missing_aware_gate.requires_grad_(False)
    if not changes.get('text_injection',True):model.inj_audio=model.inj_vision=None
    if not changes.get('reverse_fusion',True):model.core_fusion_rev=None
    if changes.get('text_rec_weight',.5)==0:model.text_recon.requires_grad_(False)
    if changes.get('feature_rec_weight',1.0)==0:model.contrastive_reconstruction.reconstruct_decoder.requires_grad_(False)
    if changes.get('rcd_weight',.1)==0:model.aux_cls_head.requires_grad_(False)
    return model


def classification_relation(student,teacher,label):
    index=(label.reshape(-1).clamp(-3,3).round()+3).long().unsqueeze(1)
    ps=F.softmax(student,dim=-1);pt=F.softmax(teacher.detach(),dim=-1)
    target=F.smooth_l1_loss(ps.gather(1,index),pt.gather(1,index))
    mask=torch.ones_like(pt).scatter_(1,index,0.0)
    ps=ps*mask;pt=pt*mask
    ps=ps/ps.sum(-1,keepdim=True).clamp_min(1e-6)
    pt=pt/pt.sum(-1,keepdim=True).clamp_min(1e-6)
    return target+.5*F.kl_div(ps.clamp_min(1e-8).log(),pt,reduction='batchmean')


def loss_batch(model,batch,args,track,changes,device,epoch,generator,fixed_rate=None):
    t,a,v=[batch[k].to(device) for k in ['text','audio','vision']]
    y=batch['label'].to(device)
    lengths=dict(audio_lengths=batch['audio_valid_lengths'].to(device),vision_lengths=batch['vision_valid_lengths'].to(device))
    valid=validity(t,a,v,lengths['audio_lengths'],lengths['vision_lengths'])
    B,L=t.shape[0],t.shape[-1]
    model.eval()
    with torch.no_grad():
        teacher=model(t,a,v,token_mask=torch.ones(B,3,L,device=device),modality_mask=torch.ones(B,3,device=device),
                      return_text_target=True,return_reconstruction_targets=True,**lengths)
    assert not teacher['logits'].requires_grad
    model.train()
    if track=='R2-T':
        cap=min(.9,.3+.2*(epoch-1))
        rate=fixed_rate if fixed_rate is not None else int(torch.randint(int(round(cap*10))+1,(),device=device,generator=generator))/10
        (t,a,v),keeps,_=native_corruption(t,a,v,valid,rate,generator,model.bert_encoder.tokenizer.unk_token_id)
        tm=aligned_availability(keeps,L,valid)
        mm=torch.ones(B,3,device=device)
        kd_scale=1.0
    else:
        p=(epoch-1)/args.epochs
        stage=min(int(p*4),3)
        rate=[0.,.15,.3,.6][stage] if fixed_rate is None else fixed_rate
        tm,_=feature_corruption_masks(valid,L,rate,generator)
        patterns=torch.tensor(PATTERNS,dtype=torch.float32,device=device)
        indices=torch.randint(7,(B,),device=device,generator=generator)
        mm=patterns[indices]
        kd_scale=.5+.5*(rate/.6)
    out=model(t,a,v,token_mask=tm,modality_mask=mm,return_contrastive_loss=True,
              text_reconstruction_target=teacher['text_target'],reconstruction_targets=teacher['reconstruction_targets'],**lengths)
    assert torch.equal(out['token_mask'],tm) and torch.equal(out['modality_mask'],mm)
    raw={'task':F.smooth_l1_loss(out['logits'].reshape(-1),y.reshape(-1))}
    raw.update({k:torch.as_tensor(val,device=device) for k,val in model.contrastive_reconstruction.last_components.items()})
    raw['text_reconstruction']=out['text_recon_loss']
    raw['prediction_kd']=F.smooth_l1_loss(out['logits'],teacher['logits'])
    raw['rcd']=classification_relation(out['aux_logits'],teacher['aux_logits'],y)
    per=[F.smooth_l1_loss(out['uni_logits'][k].reshape(-1),y.reshape(-1),reduction='none') for k in ['T','A','V']]
    raw['unimodal']=sum((z*mm[:,i]).sum()/mm[:,i].sum().clamp_min(1) for i,z in enumerate(per))
    full=mm.min(-1).values>.5
    raw['gap']=sum(((per[i][full]-per[j][full])**2).mean() for i,j in [(0,1),(0,2),(1,2)]) if full.sum()>1 else y.sum()*0
    weights=dict(task=1.,token_contrastive=.1,modality_contrastive=.1,correlation=.1,
                 feature_reconstruction=.1*changes.get('feature_rec_weight',1.),
                 text_reconstruction=changes.get('text_rec_weight',.5),
                 prediction_kd=changes.get('kd_weight',.3)*kd_scale,rcd=changes.get('rcd_weight',.1),
                 unimodal=.1,gap=changes.get('gap_weight',.05))
    loss=sum(weights[k]*value for k,value in raw.items())
    model.ablation_last_diagnostics=dict(raw={k:float(val.detach()) for k,val in raw.items()},
        weighted={k:float((weights[k]*val).detach()) for k,val in raw.items()},weights=weights,
        sampled_rate=rate,combo_ids=mm.tolist(),teacher_detached=True,explicit_masks_preserved=True,
        clean_targets_detached=all(not x.requires_grad for x in teacher['reconstruction_targets']))
    return loss
