"""Restart failed R2 initialization offline; retain completed jobs and NVMe cache."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from staging import atomic_json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args();output=cli.output.resolve()
    lock=(ROOT/'results/research2_gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=read(output/'active_worker.json')
    if (Path('/proc')/str(active['pid'])).exists():raise RuntimeError('Previous GPU1 worker is still alive')
    previous=read(output/'nvme_transition.json')
    if previous['phase']!='failed':raise RuntimeError('Expected failed controller')
    plan=read(output/'plan.json');queue=read(output/'status.json');failed_job=queue['current_job']
    if failed_job in queue['completed']:raise RuntimeError('Current job is already complete')
    worker_state=read(output/'train'/failed_job/'status.json')
    if worker_state.get('epoch',0)>0:raise RuntimeError('This recovery is restricted to initialization failure')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(key,None)
    queue_file=ROOT/'scripts/research2_ablations/run_queue.py'
    sys.path.insert(0,str(queue_file.parent))
    spec=importlib.util.spec_from_file_location('original_r2_queue',queue_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from data_cache import install_cache
    if not install_cache(ROOT/'results/checkpoint_nvme_staging_20261007/input_cache/index.json'):
        raise RuntimeError('Verified NVMe input cache is required')
    module.assert_frozen(plan)
    local_root=ROOT/'results/checkpoint_nvme_staging_20261007'/output.name
    archive=output/'failed_attempts'/(time.strftime('%Y%m%d_%H%M%S')+'_'+failed_job+'_initialization')
    archive.mkdir(parents=True,exist_ok=False)
    for label,source in {'train':output/'train'/failed_job,'checkpoint_nvme':local_root/failed_job,
                         'train.log':output/'logs'/f'{failed_job}.log','archive_receipt.json':output/f'nvme_archive_{failed_job}.json'}.items():
        if source.exists():source.rename(archive/label)
    atomic_json(archive/'controller.json',previous)
    atomic_json(archive/'active_worker.json',active)
    atomic_json(output/'offline_recovery.json',dict(preserved_failure=str(archive),restart_job=failed_job,
                completed_jobs_retained=queue['completed'],fix='forced local model cache; no proxy initialization',
                training_protocol_unchanged=True,started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    state=dict(phase='active',pid=os.getpid(),gpu=1,current_job=failed_job,staging_root=str(local_root),
               destination_root=plan['checkpoint_root'],queue_lock_acquired=True,offline=True,input_cache=True,
               controller='resume_gpu1',started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    queue.update(pid=os.getpid(),phase='train',controller=str(Path(__file__).resolve()),offline=True,input_cache=True)
    atomic_json(output/'status.json',queue);atomic_json(output/'nvme_transition.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    for job in plan['jobs']:
        name=job['job_id']
        if name in queue['completed']:continue
        module.assert_frozen(plan)
        while shutil.disk_usage(local_root.parent).free<12*1024**3:
            state.update(phase='waiting_nvme_space',current_job=name);atomic_json(output/'nvme_transition.json',state);time.sleep(15)
        destination=Path(plan['checkpoint_root'])/job['track']/name
        receipt=output/f'nvme_archive_{name}.json';log=output/'logs'/f'{name}.log'
        command=[sys.executable,str(HERE/'worker.py'),'--destination',str(destination),'--local',str(local_root/name),
                 '--receipt',str(receipt),str(queue_file.with_name('worker.py')),'--output',str(output),'--job',name]
        queue.update(phase='train',current_job=name);state.update(phase='active',current_job=name)
        atomic_json(output/'status.json',queue);atomic_json(output/'nvme_transition.json',state)
        print('START_OFFLINE_NVME',name,flush=True)
        with log.open('x') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            queue['active_worker_pid']=process.pid;atomic_json(output/'status.json',queue)
            atomic_json(output/'active_worker.json',dict(pid=process.pid,job=name,stage='train',log=str(log),command=command,
                                                       checkpoint_storage='NVMe',offline=True,input_cache=True))
            code=process.wait()
        if code or read(receipt)['phase'] not in ('staged','archiving','archived','archive_failed'):
            raise RuntimeError(f'Worker failed: {name} exit={code}; NVMe files retained')
        if read(output/'train'/name/'status.json')['phase']!='complete':raise RuntimeError('Missing completed experiment')
        queue['completed'].append(name);module.aggregate(output,plan);atomic_json(output/'status.json',queue)
        print('DONE_ASYNC_ARCHIVE_QUEUED',name,flush=True)
    queue.update(phase='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    state.update(phase='complete',finished_at=queue['finished_at'])
    atomic_json(output/'status.json',queue);atomic_json(output/'nvme_transition.json',state)


if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc();print(error,file=sys.stderr,flush=True)
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            for name in ['status.json','nvme_transition.json']:
                path=output/name;value=read(path) if path.exists() else {}
                value.update(phase='failed',traceback=error);atomic_json(path,value)
        raise
