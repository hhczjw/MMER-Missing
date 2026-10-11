"""Durable epoch-boundary snapshots including all training and random state."""
import os
from pathlib import Path
import random
import numpy as np
import torch


def save_training_state(path, model, optimizer, scheduler, device, **progress):
    path = Path(path)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None, **progress)
    temporary = path.with_suffix('.pth.tmp')
    torch.save(state, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore_training_state(state, model, optimizer, scheduler, device):
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'], device)
