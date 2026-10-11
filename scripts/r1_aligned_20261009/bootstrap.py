"""Invoke unchanged native train.py; only output, intervention and passive receipts."""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).parent))
from plan import BASE, config_for, check_config, read, write
from interventions import apply, verify_gradients


def model_digest(model):
    h = hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args();job=read(cli.job_file);out=cli.output
    sys.path.insert(0,str(BASE))
    from data_cache import install_cache
    if not install_cache(BASE/'input_cache_index.json'):
        raise RuntimeError('Verified input cache unavailable')
    sys.path.insert(0,str(BASE/'source'))
    import configs
    native_load=configs.load_config
    reference=vars(native_load('MOSI')).copy()
    def load(dataset):
        args=config_for(native_load(dataset),out,job['changes'])
        diff=check_config(vars(args),reference,job['changes'])
        write(out/'config_at_load.json',dict(args=vars(args),differences=diff,job=job))
        return args
    configs.load_config=load
    import models
    Original=models.MSAClassifier
    def build(*args,**kwargs):
        model=Original(*args,**kwargs)
        initial=model_digest(model)  # before any intervention; no random calls
        touched=apply(model,job['changes'])
        write(out/'intervention.json',dict(job=job,initial_state_sha256=initial,
            applied=touched,baseline_noop=not job['changes'],
            original_model_constructor=True,activation_checkpointing=False))
        return model
    models.MSAClassifier=build
    import trainer
    Native=trainer.Trainer
    class Observed(Native):
        def __init__(self,args,model,logger):
            super().__init__(args,model,logger)
            if len(self.optimizer.param_groups)!=2 or self.warmup_steps!=5778 or self.total_steps!=38520:
                raise RuntimeError('Native optimizer/scheduler mismatch')
            if [g['initial_lr'] for g in self.optimizer.param_groups]!=[1e-5,4e-4]:
                raise RuntimeError('Native learning rates changed')
            if any(g['weight_decay']!=1e-4 for g in self.optimizer.param_groups):
                raise RuntimeError('Native weight decay changed')
            if type(self.criterion).__name__!='MultiTaskLoss' or self.ema is not None:
                raise RuntimeError('Native loss/EMA mismatch')
            if self.criterion.task_weights != {'M':1.0}:
                raise RuntimeError('Expected native M-only task weights')
            write(out/'effective_config.json',dict(args=vars(args),job=job,
                config_differences=check_config(vars(args),dict(reference,feature_dims=[768,768,18]),job['changes']),
                criterion=type(self.criterion).__name__,loss_scope='M only',
                optimizer=[{k:g[k] for k in ('initial_lr','weight_decay','foreach','fused')} for g in self.optimizer.param_groups],
                warmup_steps=self.warmup_steps,total_steps=self.total_steps,
                grad_clip=self.grad_clip,grad_accum_steps=self.grad_accum_steps,
                native_train_entry=str(BASE/'source/train.py'),native_trainer=str(BASE/'source/trainer.py')))
            seen=False
            def observe_step(optimizer,step_args,step_kwargs):
                nonlocal seen
                if not seen:
                    write(out/'first_step_gradients.json',verify_gradients(model,job['changes']))
                    seen=True
            self.optimizer.register_step_pre_hook(observe_step)
        def train_one_epoch(self,loader,epoch):
            result=super().train_one_epoch(loader,epoch)
            write(out/'status.json',dict(phase='training',job=job['job_id'],epoch=epoch,
                max_epochs=self.args.epochs,global_step=self.global_step,loss=result))
            return result
    trainer.Trainer=Observed
    sys.argv=[str(BASE/'source/train.py'),'--dataset','MOSI','--seed',str(job['seed']),'--exp_tag','p0B']
    runpy.run_path(sys.argv[0],run_name='__main__')
    write(out/'status.json',dict(phase='native_finished',job=job['job_id']))


if __name__=='__main__':main()
