"""Native p0B forward. No new masks, pooling or auxiliary-loss optimization."""
from models.classifier import MSAClassifier


def build_model(args, changes):
    model = MSAClassifier(text_input_dim=768, audio_input_dim=args.feature_dims[1],
        video_input_dim=args.feature_dims[2], d_model=args.d_model,
        num_layers=args.num_layers, dropout=args.dropout, use_bert=True,
        bert_pretrained=args.bert_pretrained, bert_finetune=True,
        ism_depth=args.ism_depth, ism_seq_len=50, ism_d_state=args.ism_d_state,
        ism_mixer_type=args.ism_mixer_type, d_state=args.d_state,
        headdim=args.headdim, multi_task=True, sub_loss_lambda=.3,
        use_bssm_gate=True, use_gcmn_gate=True, bidirectional=False)
    model._ism_full_frame = args.ism_full_frame
    # Activation recomputation preserves dropout RNG and FP32 arithmetic.
    model.text_encoder.transformer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False})
    for layer in model.layers:
        layer.ablation_uniform_routing = changes.get('uniform_routing',False)
        if layer.ablation_uniform_routing:layer.weight_nets.requires_grad_(False)
        for cell in layer.cells.values():
            cell.ablation_source_mode = changes.get('cell_source_mode','cross')
            if cell.ablation_source_mode in ('self','value_self'):
                cell.v_projs.requires_grad_(False)
        if not changes.get('gcmn',True):layer.gcmn = None
    for encoder in [model.ism_text,model.ism_audio,model.ism_video]:
        for block in encoder.layers:
            block.ablation_global_enabled = changes.get('global_glce',True)
            block.ablation_local_enabled = changes.get('local_glce',True)
            if not block.ablation_global_enabled:block.global_extractor.requires_grad_(False)
            if not block.ablation_local_enabled:block.local_extractor.requires_grad_(False)
            if not changes.get('bssm_gate',True):
                block.use_bssm_gate=False
                block.bssm_gate.requires_grad_(False)
    return model


def forward_batch(model,batch,device):
    names=['text','audio','vision','context_text','context_audio','context_video']
    inputs={('video' if k=='vision' else k):batch[k].to(device) for k in names if k in batch}
    return model(**inputs)
