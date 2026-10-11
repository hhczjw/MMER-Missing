"""User-authorized nonfinite batch skips before any optimizer/scheduler update."""
import ast
import inspect
import math
from pathlib import Path
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/r1_repair_20261010'))
import numerical_guard as safety
import torch


class SkipBatch(Exception):pass


class SkipGuard(safety.NumericalGuard):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.skips=[]
        self.epoch_start_updates=0
        self.epoch_start_skips=0

    def start_epoch(self,epoch):
        super().start_epoch(epoch)
        self.epoch_start_updates=self.updates_checked
        self.epoch_start_skips=len(self.skips)

    def abort(self,stage,error=None):
        # Skipping a batch cannot roll back an already bad optimizer update.
        # All pre-update nonfinite events are skipped; post-update corruption
        # is still captured/stopped instead of silently training poisoned weights.
        if stage=='optimizer_parameter_update':
            return super().abort(stage,error)
        if stage not in ('forward_loss','forward_logits','backward_or_norm'):
            raise RuntimeError('Unsupported skip stage: '+stage)
        names=[]
        if stage=='backward_or_norm':
            names=[n for n,p in self.trainer.model.named_parameters()
                   if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        record=dict(stage=stage,epoch=self.epoch,batch_index=self.batch_index,
                    global_step=self.trainer.global_step,error=error,
                    loss=self.loss if self.loss is not None and math.isfinite(self.loss) else None,
                    nonfinite_gradient_tensors=len(names),first_bad_names=names[:8],
                    ids=list(map(str,(self.batch or {}).get('ids',[]))),
                    optimizer_updated=False,scheduler_updated=False)
        self.skips.append(record)
        self.loss=record['loss']
        self.write(self.output/'skipped_batches.json',dict(policy='skip batch before update',count=len(self.skips),events=self.skips))
        raise SkipBatch(f'{stage}; epoch={self.epoch}, batch={self.batch_index}')

    def progress(self):
        row=super().progress()
        row.update(policy='human-authorized nonfinite batch skip before optimizer/scheduler; no nan_to_num',
                   skipped_batches=len(self.skips),epoch_skipped_batches=len(self.skips)-self.epoch_start_skips,
                   epoch_successful_updates=self.updates_checked-self.epoch_start_updates,
                   all_batches_skipped=self.batch_index>=0 and self.updates_checked==self.epoch_start_updates)
        return row


def patched_epoch(native):
    source=textwrap.dedent(inspect.getsource(native))
    tree=ast.parse(source)
    loops=[node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple)
           and [getattr(x,'id',None) for x in node.target.elts]==['step_i','batch']]
    if len(loops)!=1:raise RuntimeError('Native batch loop changed')
    loop=loops[0]
    handler=ast.parse('''try:
    pass
except SkipBatch as error:
    nan_skipped += 1
    self.optimizer.zero_grad(set_to_none=True)
    self.logger.warning(f"[NaN skip] Epoch {epoch} step {step_i}: {error}; no optimizer/scheduler update")
    continue
''').body[0]
    handler.body=loop.body
    loop.body=[handler]
    ast.fix_missing_locations(tree)
    scope=dict(native.__globals__,SkipBatch=SkipBatch)
    exec(compile(tree,'<native-epoch-with-authorized-skip>','exec'),scope)
    return scope[native.__name__]


def install(module,output,write):
    module.Trainer.train_one_epoch=patched_epoch(module.Trainer.train_one_epoch)
    safety.NumericalGuard=SkipGuard
    result=safety.install(module,output,write)
    write(Path(output)/'skip_policy.json',dict(
        authority='2026-10-10 human: 遇到NAN后直接跳过。',
        skip='nonfinite logits/loss/gradient norm before optimizer update; clear gradients, no LR/global_step update',
        loss_denominator='native counts only successful finite batches',
        finite_path='same native math/optimizer/loss/scheduler; AST wraps only batch loop with skip exception',
        protocol_change=True,completed_original_jobs_not_rewritten=True,
        post_update_corruption='not safely skippable; capture and stop',
        nonfinite_values_not_replaced=True))
    return result
