"""CPU-only R2 preparation. Catalogue is NOT a training/launch authorization."""
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
METRICS=['Acc2_has0','Acc2_nonzero','F1_has0','F1_nonzero','Acc7','MAE','Corr']
SEEDS={'R2-M':[43,45,48],'R2-T':[45,44,46]}
RATES=[i/10 for i in range(10)]
COMMON=[
 ('00','完整对照',{},'core'),
 ('01','关闭 UME 条件编码',{'ume':False},'core'),
 ('02','关闭 Enhance 门控残差',{'enhance':False},'core'),
 ('03','关闭文本与基础特征重建',{'text_recon_weight':0.,'feature_rec_weight':0.},'core'),
 ('04','关闭预测 KD，保留 RCD',{'kd_weight':0.},'core'),
 ('05','关闭 RCD，保留预测 KD',{'rcd_weight':0.},'core'),
 ('06','关闭增强特征旁路',{'enhanced_bypass':False},'core'),
 ('07','仅关闭文本重建',{'text_recon_weight':0.},'mechanism'),
 ('08','仅关闭基础特征重建',{'feature_rec_weight':0.},'mechanism'),
 ('09','同时关闭预测 KD 与 RCD',{'kd_weight':0.,'rcd_weight':0.},'core'),
 ('10','关闭文本相似度注入',{'text_injection':False},'mechanism'),
 ('11','关闭反向融合',{'reverse_fusion':False},'mechanism'),
 ('12','仅关闭 gap 均衡损失',{'gap_weight':0.},'mechanism'),
 ('13','仅关闭 uni 监督，保留 gap',{'uni_weight':0.},'mechanism'),
]
EXTRA={
 'R2-M':[
  ('14','组合独立头改为共享头',{'combo_heads':False},'policy'),
  ('15','均衡组合采样改为原随机缺失采样',{'balanced_missing':False},'policy'),
  ('16','固定特征缺失率 .6，保留原 KD 权重课程',{'fixed_feature_rate':.6},'policy'),
  ('17','关闭课程耦合 KD，保留缺失课程',{'cd_kd':False},'policy')],
 'R2-T':[
  ('14','取消缺失率上限递增，全程采样 0–.9',{'fixed_token_rate_cap':.9},'policy')],
}


def variants(track=None):
    rows=[]
    for t in [track] if track else ['R2-M','R2-T']:
        for index,title,changes,group in COMMON+EXTRA[t]:
            rows.append(dict(id=f'{t}-{index}',track=t,title=title,
                             changes=deepcopy(changes),group=group))
    return rows


def jobs(core_only=False):
    return [dict(row,seed=seed,job_id=f'{row["id"]}_seed{seed}')
            for track in ['R2-M','R2-T'] for seed in SEEDS[track]
            for group in ['core','mechanism','policy']
            for row in variants(track) if row['group']==group and (not core_only or group=='core')]


def reference_contract(track):
    common=dict(task_loss='SmoothL1 beta=1',batch_size=8,train_batches=160,
        drop_last=True,shuffle_rng='native global Torch RNG',precision='FP32',
        model='UMRMv3',d_model=128,core_layers=2,ism_depth=3,
        bert='RoBERTa-large LoRA',lora_r=8,lora_alpha=16,
        dropout=.5,learning_rate=3e-4,weight_decay=1e-4,grad_clip=.1,
        epochs=60,patience=15,eval_batch_size=16,
        loss_weights=dict(kd=.3,text_reconstruction=.5,contrastive_reconstruction=.1,
                          uni=.1,gap=.05,rcd=.1),auxiliary_ce7=False,
        metrics=METRICS,initialization='pretrained text plus fresh LoRA and model; never resume archived best weights')
    if track=='R2-M':
        common.update(seed_order=SEEDS[track],teacher='train mode plus no_grad, shared weights',
            reconstruction='historical student text target / enhanced student feature target; do not substitute detached clean teacher',
            corruption='native implicit post-ISM feature masks, legacy fixed boundary positions, balanced 7 modality patterns',
            validation='full TAV MAE; strict improvement; not six-combo MAE',
            primary_checkpoint='best validation TAV MAE',
            schedule='LambdaLR .28/.46/.64/.82/1.0; then min-mode plateau, patience 7, native minimum LR',
            token_rates=[0.,.15,.3,.6],curriculum_epoch_ranges=[[1,15],[16,30],[31,45],[46,60]],
            kd_weights=[.15,.1875,.225,.3],test='TAV and six missing combinations, same checkpoint',
            historical_source_confidence='checkpoint args verified; complete historical training-source snapshot unavailable; first-epoch loss/mask parity remains a gate')
    elif track=='R2-T':
        common.update(seed_order=SEEDS[track],teacher='eval mode plus no_grad, detached clean targets, shared weights',
            reconstruction='clean teacher text encoding and enhanced feature targets, detached',
            corruption='input_valid_aware; UNK/zero before encoding; real special tokens retained; padding excluded; all modalities available',
            validation='Token10 mean Acc7; lower MAE exact-tie break; masks seed20261006',
            primary_checkpoint='best validation Token10 Acc7',
            schedule='manual epoch warmup .2/.4/.6/.8/1.0; max-mode plateau on Acc7, patience 5, min_lr 1e-6',
            token_rate_caps=[.3,.5,.7,.9],min_epochs=20,
            test='rates .0–.9; seed20261005; eval batch16 and dataset order frozen',
            historical_source_confidence='original Token run.json hashes verified; native train_epoch is the baseline')
    else:raise ValueError(track)
    return common


def apply_loss_changes(reference,changes):
    """Do not remove heads/teacher or change initialization for a loss-only ablation."""
    args=deepcopy(reference)
    for key in ['kd_weight','rcd_weight','text_recon_weight','uni_weight','gap_weight']:
        if key in changes:args[key]=changes[key]
    # use_kd remains true with kd_weight=0: RCD and clean targets still need teacher.
    return args


def warmup_factors(track):
    return [.1+.9*(i+1)/5 for i in range(5)] if track=='R2-M' else [(i+1)/5 for i in range(5)]
