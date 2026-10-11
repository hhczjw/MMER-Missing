"""Run an unmodified frozen worker with checkpoint IO staged on local NVMe."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import traceback
from staging import CheckpointStaging, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resume',action='store_true',help='Resume an existing R1 full-state checkpoint')
    parser.add_argument('--batch-size', type=int, help='Explicit user-authorized R1 batch override')
    parser.add_argument('worker', type=Path)
    parser.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        os.environ.pop(name,None)
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    if args.resume and not (args.local/'latest_resume.pth').exists():
        raise FileNotFoundError('Full training resume state is required')
    if args.local.exists() and not args.resume:
        raise FileExistsError(f'Staging path already exists: {args.local}')
    args.local.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, dict(phase='training', pid=os.getpid(),
                                 destination=str(args.destination), staging=str(args.local)))
    if args.batch_size is not None:os.environ['R1_CHECKPOINT_STAGING']=str(args.local.resolve())
    from data_cache import install_cache
    install_cache(Path(__file__).resolve().parents[2]/'results/checkpoint_nvme_staging_20261007/input_cache/index.json')
    import torch
    staging = CheckpointStaging(args.destination, args.local)
    staging.install(torch)
    if args.resume:
        if args.batch_size is None or args.worker.name!='r1_worker_recovery.py':
            raise ValueError('Resume is restricted to the validated R1 recovery worker')
        output=Path(args.worker_args[args.worker_args.index('--output')+1]).absolute()
        allowed={args.destination.absolute(),args.local.absolute(),output}
        redirected_mkdir=Path.mkdir
        def resume_mkdir(path,*positional,**keyword):
            if path.absolute() in allowed:keyword['exist_ok']=True
            return redirected_mkdir(path,*positional,**keyword)
        Path.mkdir=resume_mkdir
    sys.path.insert(0, str(args.worker.resolve().parent))
    sys.argv = [str(args.worker), *args.worker_args]
    try:
        try:
            if args.batch_size is None:
                runpy.run_path(str(args.worker), run_name='__main__')
            else:
                namespace = runpy.run_path(str(args.worker), run_name='__nvme_batch_override__')
                original_configure = namespace['configure']
                def configure(track, *positional, **keyword):
                    if track != 'R1':
                        raise ValueError('Batch override is restricted to R1')
                    configured = original_configure(track, *positional, **keyword)
                    configured.original_batch_size = configured.batch_size
                    configured.batch_size = args.batch_size
                    configured.batch_override_reason = 'user-authorized CUDA OOM recovery'
                    print('BATCH_OVERRIDE', configured.original_batch_size, '->', configured.batch_size, flush=True)
                    return configured
                namespace['main'].__globals__['configure'] = configure
                namespace['main']()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        staging.restore()
        staging.stage(args.receipt)
        print('CHECKPOINT_READY_FOR_ASYNC_ARCHIVE', args.destination, flush=True)
    except BaseException:
        staging.restore()
        atomic_json(args.receipt, dict(phase='failed', staging=str(args.local),
                                     destination=str(args.destination), traceback=traceback.format_exc(),
                                     local_checkpoint_retained=True))
        raise
    finally:
        staging.restore()


if __name__ == '__main__':
    main()
