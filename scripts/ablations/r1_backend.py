"""Isolated R1 adapter: effective-length ISM and masked pooling, no live repo edits."""
import torch
import torch.nn.functional as F
from models.classifier import MSAClassifier


def valid_frames(value, lengths=None):
    if lengths is not None:
        return torch.arange(value.shape[1], device=value.device)[None] < lengths.to(value.device)[:, None]
    # External files provide no lengths. Use rightmost nonzero frame, not sum(nonzero).
    nonzero = value.abs().sum(-1) > 0
    index = torch.arange(value.shape[1], device=value.device)[None] + 1
    inferred = (index * nonzero).amax(-1)
    return index <= inferred[:, None]


def masked_align(value, mask, length):
    if value.shape[1] == length:
        return value * mask[..., None], mask
    num = F.adaptive_avg_pool1d((value * mask[..., None]).transpose(1,2), length).transpose(1,2)
    den = F.adaptive_avg_pool1d(mask.float()[:,None], length)[:,0]
    return num / den.clamp_min(1e-8)[...,None], den > 0


class R1Model(MSAClassifier):
    """Same modules as p0B; all ablations share this audited length-aware adapter."""
    def encode_features(self, text, audio, video, audio_lengths=None, vision_lengths=None):
        text, text_mask = self.text_encoder(text)
        masks = [text_mask.bool(), valid_frames(audio,audio_lengths), valid_frames(video,vision_lengths)]
        values = [self.proj_text(text), self.proj_audio(audio), self.proj_video(video)]
        cls = []
        for i, encoder in enumerate([self.ism_text,self.ism_audio,self.ism_video]):
            values[i] = values[i] * masks[i][...,None]
            values[i], token = encoder(values[i], mask=masks[i], return_cls=True)
            cls.append(token)
        Lt = values[0].shape[1]
        for i in [1,2]:
            values[i], masks[i] = masked_align(values[i],masks[i],Lt)
        xt,xa,xv = values
        for layer in self.layers:
            xa,xv,xt = layer(xa,xv,xt)
            xt,xa,xv = [x*m[...,None] for x,m in zip([xt,xa,xv],masks)]
        return [xt,xa,xv], masks, cls

    def forward(self, text, audio, video, audio_lengths=None, vision_lengths=None,
                context_text=None, context_audio=None, context_video=None):
        values,masks,cls = self.encode_features(text,audio,video,audio_lengths,vision_lengths)
        pools = [self._pool(x,m) for x,m in zip(values,masks)]
        if context_text is not None:
            ctx,ctx_masks,_ = self.encode_features(context_text,context_audio,context_video)
            ctx_pools = [self._pool(x,m) for x,m in zip(ctx,ctx_masks)]
            tl,ta,tv = [torch.stack([c,m],dim=1) for c,m in zip(ctx_pools,pools)]
            for layer in self.layers:
                ta,tv,tl = layer(ta,tv,tl)
            pools = [tl[:,1],ta[:,1],tv[:,1]]
        fused = self.fusion_norm(torch.cat(pools,dim=-1))
        return dict(logits=self.head(fused),
                    sub_T=self.sub_fc_T(cls[0]), sub_A=self.sub_fc_A(cls[1]), sub_V=self.sub_fc_V(cls[2]))


def build_model(args, changes):
    # Construct identical reference modules first; apply interventions afterwards
    # so a module removal cannot shift downstream parameter initialization.
    model = R1Model(text_input_dim=768, audio_input_dim=args.feature_dims[1],
        video_input_dim=args.feature_dims[2], d_model=128, num_layers=2, dropout=.6,
        use_bert=True, bert_pretrained=args.bert_pretrained, bert_finetune=True,
        ism_depth=1, ism_seq_len=50, ism_d_state=32, ism_mixer_type='bimamba',
        d_state=64, headdim=32, multi_task=True, sub_loss_lambda=.3,
        use_bssm_gate=True, use_gcmn_gate=True, bidirectional=False)
    model.text_encoder.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for layer in model.layers:
        layer.ablation_uniform_routing = changes.get('uniform_routing',False)
        if layer.ablation_uniform_routing:
            layer.weight_nets.requires_grad_(False)
        for cell in layer.cells.values():
            cell.ablation_source_mode = changes.get('cell_source_mode','cross')
            if cell.ablation_source_mode in ('self','value_self'):
                cell.v_projs.requires_grad_(False)
        if not changes.get('gcmn',True):
            layer.gcmn = None
    for encoder in [model.ism_text,model.ism_audio,model.ism_video]:
        for block in encoder.layers:
            block.ablation_global_enabled = changes.get('global_glce',True)
            block.ablation_local_enabled = changes.get('local_glce',True)
            if not block.ablation_global_enabled:block.global_extractor.requires_grad_(False)
            if not block.ablation_local_enabled:block.local_extractor.requires_grad_(False)
            if not changes.get('bssm_gate',True):
                block.use_bssm_gate = False
                block.bssm_gate.requires_grad_(False)
    if changes.get('sub_loss_weight',.3) == 0:
        for head in [model.sub_fc_T,model.sub_fc_A,model.sub_fc_V]:head.requires_grad_(False)
    return model


def forward_batch(model,batch,device):
    names = ['text','audio','vision','context_text','context_audio','context_video']
    inputs = {('video' if k=='vision' else k):batch[k].to(device) for k in names if k in batch}
    for key in ['audio_lengths','vision_lengths']:
        if key in batch:inputs[key]=batch[key].to(device)
    return model(**inputs)
