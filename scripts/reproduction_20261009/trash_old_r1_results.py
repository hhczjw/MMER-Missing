"""One-shot recoverable removal of verified inactive OLD R1 results only.

Retain shared archiver administrative receipts/status at its polled R1 root.
Do not touch Research2, new R1 native runs, frozen code, data or best results.
"""
import argparse
import json
import os
from pathlib import Path

ROOT=Path('/home/zjw/桌面/MMER-Missing')
ARCHIVES=Path('/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints')
LOCAL_TRASH=ROOT/'.trash/old_r1_ablations_20261009'
HDD_TRASH=ARCHIVES/'.trash/old_r1_ablations_20261009'
NAMES=['mosi_ablations_p0b_0919_gpu0_20261006',
       'mosi_ablations_p0b_0919_gpu0_20261006_v2',
       'mosi_ablations_p0b_0919_gradient_repair_20261007']
MIXED='mosi_ablations_gpu0_20261006'


def read(p):return json.loads(p.read_text())


def alive(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ',1)[1][0]!='Z'
    except (OSError,ValueError,TypeError):return False


def pairs():
    results=ROOT/'results';rows=[]
    # Failed early pure-R1 runs: whole result directories, all descendants R1.
    for name in NAMES[:2]:
        p=results/name
        rows.append((p,LOCAL_TRASH/'results'/name))
    # Shared archiver still polls this output root. Never rename the root or receipts.
    main=results/NAMES[2]
    for name in ('train','logs','reference','smoke','failed_attempts',
                 'repair_checks','repair_check_weights','summary.json'):
        p=main/name
        if p.exists() or p.is_symlink():rows.append((p,LOCAL_TRASH/'results'/NAMES[2]/name))
    # The earliest result directory also contains R2 smoke results: retain them.
    mixed=results/MIXED
    for folder in ('train','smoke','reference'):
        for p in sorted((mixed/folder).glob('R1-*')):
            rows.append((p,LOCAL_TRASH/'results'/MIXED/folder/p.name))
    for p in sorted((mixed/'logs').glob('*R1-*')):
        rows.append((p,LOCAL_TRASH/'results'/MIXED/'logs'/p.name))
    staging=results/'checkpoint_nvme_staging_20261007'/NAMES[2]
    if staging.exists():rows.append((staging,LOCAL_TRASH/'nvme_staging'/NAMES[2]))
    for name in NAMES:
        p=ARCHIVES/name
        if p.exists():rows.append((p,HDD_TRASH/name))
    old_mixed_cp=ARCHIVES/MIXED
    for folder in ('train','smoke','reference'):
        for p in sorted((old_mixed_cp/folder).glob('R1-*')):
            rows.append((p,HDD_TRASH/MIXED/folder/p.name))
    return rows


def verify(rows):
    for name in NAMES:
        p=ROOT/'results'/name
        state=read(p/'status.json')
        if state['phase'] not in ('failed','paused'):
            raise RuntimeError('Old queue not inactive: '+name)
        active=read(p/'active_worker.json')
        for pid in {state.get('pid'),state.get('active_worker_pid'),active.get('pid')}-{None}:
            if alive(pid):raise RuntimeError('Old queue PID still alive: '+str(pid))
    for p in (ROOT/'results'/NAMES[2]).glob('nvme_archive_*.json'):
        if read(p)['phase']!='archived':raise RuntimeError('Old R1 archive incomplete: '+str(p))
    destinations=[]
    for src,dst in rows:
        if not (src.exists() or src.is_symlink()):raise RuntimeError('Missing source '+str(src))
        if dst.exists() or dst.is_symlink():raise RuntimeError('Trash destination exists '+str(dst))
        if src==ROOT or src==ARCHIVES or src.name.startswith('R2'):
            raise RuntimeError('Unsafe broad/R2 target')
        if not str(src).startswith((str(ROOT/'results')+'/',str(ARCHIVES)+'/')):
            raise RuntimeError('Unexpected source root')
        destinations.append(str(dst))
    if len(destinations)!=len(set(destinations)):raise RuntimeError('Duplicate trash target')
    # Open descriptors are stronger evidence than a historical pathname in archiver argv.
    prefixes=[str(s)+'/' for s,_ in rows if not s.is_symlink()]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            for fd in (proc/'fd').iterdir():
                target=os.readlink(fd)
                if any(target.startswith(p) for p in prefixes):
                    raise RuntimeError('Target files still open by PID'+proc.name)
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    cli=parser.parse_args();rows=pairs();verify(rows)
    plan=[dict(source=str(src),trash=str(dst),symlink=src.is_symlink(),inode=src.lstat().st_ino) for src,dst in rows]
    if not cli.apply:
        print(json.dumps(dict(operation='recoverable old-R1 removal',count=len(rows),targets=plan),ensure_ascii=False,indent=2));return
    if LOCAL_TRASH.exists() or HDD_TRASH.exists():raise RuntimeError('Cleanup already started; do not repeat')
    LOCAL_TRASH.mkdir(parents=True,exist_ok=False)
    receipt=LOCAL_TRASH/'receipt.json'
    document=dict(authority='Human: 将之前旧的消融结果删除。 Context: Research1 old ablations replaced by new GPU1 native-aligned cohort',
        phase='removing',targets=plan,moved=[],
        retained='Research2 results/all current queues/historical best/native replay outputs/frozen sources; shared archiver R1 metadata and receipts retained',
        recovery='Rename each trash path back to source; this is not permanent erasure and does not free disk blocks')
    receipt.write_text(json.dumps(document,ensure_ascii=False,indent=2))
    try:
        for row,(src,dst) in zip(plan,rows):
            if src.lstat().st_ino!=row['inode']:raise RuntimeError('Source identity changed: '+str(src))
            dst.parent.mkdir(parents=True,exist_ok=True)
            src.rename(dst)  # Same-filesystem rename; does not follow symlinks.
            document['moved'].append(str(src))
            receipt.write_text(json.dumps(document,ensure_ascii=False,indent=2))
        if any(src.exists() or src.is_symlink() for src,_ in rows):raise RuntimeError('Some source paths remain')
        document['phase']='complete'
        receipt.write_text(json.dumps(document,ensure_ascii=False,indent=2))
    except BaseException as error:
        document.update(phase='partial_failure',error=str(error))
        receipt.write_text(json.dumps(document,ensure_ascii=False,indent=2));raise
    print(json.dumps(dict(phase='complete',removed_paths=len(rows),receipt=str(receipt)),ensure_ascii=False))


if __name__=='__main__':main()
