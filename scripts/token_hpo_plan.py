"""Frozen, bounded seed42 Token hyperparameter plan and validation-only ranking."""
from copy import deepcopy

BASE = dict(learning_rate=3e-4, dropout=.5, batch_size=8,
            text_recon_weight=.5, contrastive_loss_weight=.1,
            kd_weight=.3, uni_weight=.1, gap_weight=.05, rcd_weight=.1,
            use_aux_cls=False, aux_cls_weight=.1, lora_r=8, lora_alpha=16)
LIGHT = dict(text_recon_weight=.1, contrastive_loss_weight=.03,
             kd_weight=.1, uni_weight=.05, gap_weight=.01, rcd_weight=.05)


def candidates():
    variations = [
        ('00_reference', {}),
        ('01_dropout30', dict(dropout=.3)),
        ('02_dropout10', dict(dropout=.1)),
        ('03_lr10_dropout30', dict(learning_rate=1e-4, dropout=.3)),
        ('04_lr20_dropout30', dict(learning_rate=2e-4, dropout=.3)),
        ('05_light_aux_losses', dict(LIGHT,learning_rate=2e-4,dropout=.3)),
        ('06_light_dropout10', dict(LIGHT,learning_rate=3e-4,dropout=.1)),
        ('07_light_ce7', dict(LIGHT,learning_rate=2e-4,dropout=.3,use_aux_cls=True)),
        ('08_light_batch16', dict(LIGHT,learning_rate=3e-4,dropout=.3,batch_size=16)),
        ('09_light_lora16', dict(LIGHT,learning_rate=2e-4,dropout=.3,lora_r=16,lora_alpha=32)),
    ]
    return [dict(id=name,params=dict(deepcopy(BASE),**delta)) for name,delta in variations]


def joint(avg):
    return .5*(avg['Acc7']+avg['Acc2_nonzero'])


def order(avg, objective='joint'):
    primary=joint(avg) if objective=='joint' else avg['Acc7']
    return (primary,avg['Acc7'],avg['Acc2_nonzero'],-avg['MAE'])


def champions(board):
    """Only validation data may enter this selector, never test results."""
    reference=next(row for row in board if row['id']=='00_reference')
    floor=reference['best']['joint']['avg']['Acc2_nonzero']-.01
    points=[dict(trial=row['id'],selector=key,selection=point)
            for row in board for key,point in row['best'].items()]
    winner=max(points,key=lambda p:order(p['selection']['avg'],'joint'))
    guarded=[p for p in points if p['selection']['avg']['Acc2_nonzero']>=floor]
    acc7=max(guarded,key=lambda p:order(p['selection']['avg'],'Acc7'))
    return dict(joint=winner,Acc7_guarded=acc7,acc2_nonzero_floor=floor)


def finalists(board):
    chosen=champions(board)
    ids=list(dict.fromkeys([chosen['joint']['trial'],chosen['Acc7_guarded']['trial']]))
    if len(ids)==1 and len(board)>1:
        rest=sorted((row for row in board if row['id'] not in ids),key=lambda r:order(r['best']['joint']['avg']),reverse=True)
        ids.append(rest[0]['id'])
    return ids[:2]
