"""Read-only redirection to byte-verified local input copies."""
import builtins
import json
import os
from pathlib import Path


def install_cache(index):
    index = Path(index)
    if not index.exists():
        return False
    document = json.loads(index.read_text())
    if document['phase'] not in ('ready','copying'):
        return False
    mapping = document['mapping']
    original_open, original_path_open = builtins.open, Path.open
    original_exists, original_isfile = os.path.exists, os.path.isfile
    def redirect(path, mode):
        if isinstance(path, (str, os.PathLike)) and not any(flag in mode for flag in 'wa+'):
            name = str(Path(path).absolute())
            if name not in mapping and name.startswith('/media/zjw/951FB31A9E1EB7E0/dateSet/'):
                with original_path_open(index,'r') as stream:
                    updated=json.load(stream)
                if updated['phase'] in ('ready','copying'):mapping.update(updated['mapping'])
            if name in mapping:
                return Path(mapping[name]['local'])
        return path
    def open_file(file, mode='r', *args, **kwargs):
        return original_open(redirect(file, mode), mode, *args, **kwargs)
    def path_open(path, mode='r', *args, **kwargs):
        return original_path_open(redirect(path, mode), mode, *args, **kwargs)
    def exists(path):return original_exists(redirect(path,'r'))
    def isfile(path):return original_isfile(redirect(path,'r'))
    builtins.open, Path.open = open_file, path_open
    os.path.exists, os.path.isfile = exists, isfile
    return True
