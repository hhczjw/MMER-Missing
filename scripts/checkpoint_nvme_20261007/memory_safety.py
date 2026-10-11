"""Temporarily offload Adam moments during the rare reference-backward retry."""
from contextlib import contextmanager
import gc
import torch


@contextmanager
def offload_optimizer_state(optimizer):
    moved = []
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                target = value.device
                state[name] = value.to('cpu')
                moved.append((state, name, target))
    # Loop locals must not retain the last GPU moment tensor.
    value = None
    gc.collect()
    torch.cuda.empty_cache()
    try:
        yield len(moved)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        for state, name, target in moved:
            state[name] = state[name].to(target)
