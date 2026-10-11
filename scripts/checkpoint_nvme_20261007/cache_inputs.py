"""Copy only the queues' frozen HDD datasets to NVMe and verify frozen SHA256."""
import hashlib
import json
import os
from pathlib import Path
from staging import atomic_json

ROOT = Path(__file__).resolve().parents[2]
cache = ROOT / 'results/checkpoint_nvme_staging_20261007/input_cache'
cache.mkdir(parents=True, exist_ok=True)
index = cache / 'index.json'
inputs = {}
for name in ['mosi_ablations_p0b_0919_gradient_repair_20261007', 'research2_ablations_gpu1_20261006_v3']:
    plan = json.loads((ROOT / 'results' / name / 'plan.json').read_text())
    for source, expected in plan['data_sha256'].items():
        if source.startswith('/media/zjw/951FB31A9E1EB7E0/'):
            if source in inputs and inputs[source] != expected:
                raise ValueError('Conflicting input hashes')
            inputs[source] = expected
mapping = {}
atomic_json(index, dict(phase='copying', mapping=mapping, pid=os.getpid()))
try:
    for source, expected in inputs.items():
        target = cache / (expected[:12] + '_' + Path(source).name)
        temporary = target.with_suffix(target.suffix + '.tmp')
        digest = hashlib.sha256()
        with open(source, 'rb') as incoming, temporary.open('xb') as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024**2), b''):
                digest.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if digest.hexdigest() != expected:
            raise ValueError('Frozen dataset hash mismatch; cache will not be used')
        temporary.replace(target)
        mapping[source] = dict(local=str(target), sha256=expected, bytes=target.stat().st_size)
        atomic_json(index, dict(phase='copying', mapping=mapping, pid=os.getpid()))
        print('INPUT_CACHED_VERIFIED', source, target.stat().st_size, flush=True)
    atomic_json(index, dict(phase='ready', mapping=mapping, pid=os.getpid()))
except BaseException as exc:
    atomic_json(index, dict(phase='failed', error=str(exc), mapping=mapping))
    raise
