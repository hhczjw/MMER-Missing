"""Fail before poisoned updates; finite native training is left unchanged.

This is a safety repair, not a claim to have found the originating numerical
fault. Never skip/retry a batch, change precision, or substitute a backward.
"""
import hashlib
import math
from pathlib import Path
import random

import numpy as np
import torch


def digest(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def finite_tensor_rows(named):
    rows = []
    for name, value in named:
        if value is None or not value.dtype.is_floating_point:
            continue
        good = torch.isfinite(value)
        if not bool(good.all()):
            rows.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                             nonfinite=int((~good).sum()), nan=int(torch.isnan(value).sum()),
                             inf=int(torch.isinf(value).sum())))
    return rows


class NumericalGuard:
    def __init__(self, trainer, output, write):
        self.trainer = trainer
        self.output = Path(output)
        self.write = write
        self.epoch = 0
        self.batch_index = -1
        self.batch = None
        self.rng_before = None
        self.loss = None
        self.logits = None
        self.last_norm = None
        self.updates_checked = 0
        self.failed = False

    def start_epoch(self, epoch):
        self.epoch = epoch
        self.batch_index = -1

    def start_batch(self, batch):
        if not self.trainer.model.training:
            return
        self.batch_index += 1
        self.batch = batch
        self.loss = None
        self.logits = None
        # Copies only generator states: no draws, seeds or RNG resets.
        self.rng_before = dict(torch_cpu=torch.get_rng_state(),
                              torch_cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
                              numpy=np.random.get_state(), python=random.getstate())

    def loss_hook(self, module, inputs, output):
        if not self.trainer.model.training:
            return
        self.loss = float(output.detach())
        if not math.isfinite(self.loss):
            self.abort('forward_loss')

    def model_hook(self, module, inputs, output):
        if not self.trainer.model.training:
            return
        logits = output.get('logits') if isinstance(output, dict) else output
        if torch.is_tensor(logits):
            self.logits = logits.detach()
            if not bool(torch.isfinite(logits).all()):
                self.abort('forward_logits')

    def checked_clip(self, native_clip, parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        # Native error_if_nonfinite checks the computed total norm BEFORE
        # scaling any gradient. The finite branch uses identical operations.
        kwargs['error_if_nonfinite'] = True
        try:
            norm = native_clip(parameters, max_norm, *args, **kwargs)
        except RuntimeError as error:
            if 'non-finite' not in str(error):
                raise
            self.abort('backward_or_norm', str(error))
        self.last_norm = float(norm)
        return norm

    def after_step(self, optimizer, args, kwargs):
        # Batched reduction detects nonfinite parameters without altering them.
        floating = [(n, p) for n, p in self.trainer.model.named_parameters()
                    if p.requires_grad and p.dtype.is_floating_point]
        norms = torch._foreach_norm([p.detach() for _, p in floating])
        if not bool(torch.isfinite(torch.stack(norms)).all()):
            # Distinguish finite FP32 norm overflow from nonfinite elements.
            bad = finite_tensor_rows(floating)
            if bad:
                self.abort('optimizer_parameter_update')
        self.updates_checked += 1
        if self.updates_checked % 100 == 0:
            self.write(self.output/'numerical_progress.json', self.progress())

    def progress(self):
        return dict(phase='finite_so_far', epoch=self.epoch, batch_index=self.batch_index,
                    global_step=self.trainer.global_step, updates_checked=self.updates_checked,
                    last_loss=self.loss, last_preclip_norm=self.last_norm,
                    policy='native finite path; fail-before-nonfinite-update; no skips/retries/fallback')

    def abort(self, stage, error=None):
        if self.failed:
            raise RuntimeError('Numerical guard already failed')
        self.failed = True
        model = self.trainer.model
        parameters = list(model.named_parameters())
        bad_grad = finite_tensor_rows((n, p.grad) for n, p in parameters)
        bad_param = finite_tensor_rows(parameters)
        state_tensors = []
        names = {id(p): n for n, p in parameters}
        for p, state in self.trainer.optimizer.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state_tensors.append((names.get(id(p), '?')+'.'+key, value))
        bad_state = finite_tensor_rows(state_tensors)
        finite_grad_max = max((float(p.grad.detach().abs().max()) for _, p in parameters
                              if p.grad is not None and bool(torch.isfinite(p.grad).all())), default=0.)
        report = dict(phase='numerical_failure_captured', stage=stage, error=error,
                      epoch=self.epoch, batch_index=self.batch_index,
                      global_step=self.trainer.global_step, loss=self.loss,
                      bad_gradients=bad_grad, bad_parameters=bad_param,
                      bad_optimizer_state=bad_state, finite_gradient_max=finite_grad_max,
                      learning_rates=[g['lr'] for g in self.trainer.optimizer.param_groups],
                      batch_hashes={k: digest(v) for k, v in (self.batch or {}).items()
                                    if torch.is_tensor(v)},
                      ids=list(map(str, (self.batch or {}).get('ids', []))),
                      preceding_updates_checked=self.updates_checked,
                      propagation_prevented=stage in ('backward_or_norm','forward_logits','forward_loss'),
                      originating_fault='not yet identified; replay captured batch/backward next')
        self.write(self.output/'numerical_failure.json', report)
        # Diagnostic state only; production training must still start from scratch.
        # Stream tensors to disk; no full GPU clone and no change to parameter values.
        snapshot = dict(model=model.state_dict(), optimizer=self.trainer.optimizer.state_dict(),
                        scheduler=self.trainer.scheduler.state_dict(),
                        gradients={n:p.grad for n,p in parameters if p.grad is not None},
                        batch=self.batch, rng_before=self.rng_before,
                        rng_after=dict(torch_cpu=torch.get_rng_state(),
                                       torch_cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None),
                        args=vars(self.trainer.args), report=report)
        tmp = self.output/'failure_snapshot.pt.tmp'
        torch.save(snapshot, tmp)
        tmp.replace(self.output/'failure_snapshot.pt')
        raise RuntimeError(f'Numerical failure captured before continuing: {stage}; '
                           f'epoch={self.epoch} batch={self.batch_index}. No batch skipped or retried.')


def install(trainer_module, output, write):
    Native = trainer_module.Trainer
    native_clip = torch.nn.utils.clip_grad_norm_
    active = {}

    class Guarded(Native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.numerical_guard = NumericalGuard(self, output, write)
            self.criterion.register_forward_hook(self.numerical_guard.loss_hook)
            self.model.register_forward_hook(self.numerical_guard.model_hook)
            self.optimizer.register_step_post_hook(self.numerical_guard.after_step)

        def _to_device(self, batch):
            result = super()._to_device(batch)
            self.numerical_guard.start_batch(result)
            return result

        def train_one_epoch(self, loader, epoch):
            guard = self.numerical_guard
            guard.start_epoch(epoch)
            active['guard'] = guard
            try:
                result = super().train_one_epoch(loader, epoch)
                write(Path(output)/'numerical_progress.json', guard.progress())
                return result
            finally:
                active.clear()

    def checked_clip(parameters, max_norm, *args, **kwargs):
        guard = active.get('guard')
        if guard is None:
            return native_clip(parameters, max_norm, *args, **kwargs)
        return guard.checked_clip(native_clip, parameters, max_norm, *args, **kwargs)

    trainer_module.Trainer = Guarded
    torch.nn.utils.clip_grad_norm_ = checked_clip
    return Guarded
