"""Redirect only one worker's checkpoint directory to NVMe; archive on success."""
import hashlib
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


class CheckpointStaging:
    def __init__(self, destination, local):
        self.destination = Path(destination).absolute()
        self.local = Path(local).absolute()
        if self.local == self.destination or self.local.is_relative_to(self.destination):
            raise ValueError('Staging must be on a separate path')
        self.originals = None

    def redirect(self, path):
        if not isinstance(path, (str, os.PathLike)):
            return path
        path = Path(path).absolute()
        if path.is_relative_to(self.destination):
            return self.local / path.relative_to(self.destination)
        return path

    def install(self, torch):
        original_save, original_load = torch.save, torch.load
        original_mkdir, original_replace = Path.mkdir, Path.replace
        staging = self
        def save(obj, f, *args, **kwargs):
            return original_save(obj, staging.redirect(f), *args, **kwargs)
        def load(f, *args, **kwargs):
            return original_load(staging.redirect(f), *args, **kwargs)
        def mkdir(path, *args, **kwargs):
            return original_mkdir(staging.redirect(path), *args, **kwargs)
        def replace(path, target):
            return original_replace(staging.redirect(path), staging.redirect(target))
        self.originals = torch, original_save, original_load, original_mkdir, original_replace
        torch.save, torch.load, Path.mkdir, Path.replace = save, load, mkdir, replace

    def restore(self):
        if self.originals:
            torch, save, load, mkdir, replace = self.originals
            torch.save, torch.load, Path.mkdir, Path.replace = save, load, mkdir, replace
            self.originals = None

    def stage(self, receipt):
        self.restore()
        files = sorted(p for p in self.local.rglob('*') if p.is_file())
        if not files or any(p.name.endswith('.tmp') for p in files):
            raise RuntimeError('Expected completed checkpoint files in staging')
        for source in files:
            with source.open('rb') as stream:
                os.fsync(stream.fileno())
        atomic_json(receipt, dict(phase='staged', destination=str(self.destination),
                                 staging=str(self.local), ready_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                 bytes=sum(p.stat().st_size for p in files),
                                 files=[str(p) for p in files]))

    def archive(self, receipt):
        self.restore()
        files = sorted(p for p in self.local.rglob('*') if p.is_file())
        if not files or any(p.name.endswith('.tmp') for p in files):
            raise RuntimeError('Expected completed checkpoint files in staging')
        records = []
        atomic_json(receipt, dict(phase='archiving', destination=str(self.destination),
                                 staging=str(self.local), started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        for source in files:
            target = self.destination / source.relative_to(self.local)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.nvme-archive.tmp')
            digest = hashlib.sha256()
            with source.open('rb') as incoming, temporary.open('wb') as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024**2), b''):
                    outgoing.write(block)
                    digest.update(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if temporary.stat().st_size != source.stat().st_size:
                raise RuntimeError('Archive size mismatch; staging retained')
            temporary.replace(target)
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            records.append(dict(source=str(source), destination=str(target),
                                bytes=source.stat().st_size, sha256=digest.hexdigest()))
        atomic_json(receipt, dict(phase='archived', files=records,
                                 finished_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        # Only remove the files created by this worker after durable archival.
        for source in files:
            source.unlink()
        return records
