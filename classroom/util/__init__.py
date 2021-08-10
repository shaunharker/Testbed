import torch
from math import log
from .delaykeyboardinterrupt import DelayKeyboardInterrupt
from .ignorekeyboardinterrupt import IgnoreKeyboardInterrupt
from .stopwatch import Stopwatch
from .livelist import LiveList
from .filters import Filter, KalmanFilter1D, MedianFilter, TwoWindowFilter
from pathlib import Path
from functools import lru_cache
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
nvmlInit()

lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8

# PyTorch default device
def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def default_data_path():
    return f"/home/{os.environ.get('USERNAME')}/data/"

# Convenience
def numel(model):
    return sum(p.numel() for p in model.parameters())

# PyTorch/CUDA Memory
def memory_allocated():
    return torch.cuda.memory_allocated(0)

def memory_free():
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f1 = r-a  # free inside reserved
    f2 = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free
    return f1 + f2

@lru_cache
def memory_usage(f, shape):
    # TODO: Test and see if this works. Also: try--except.
    x = torch.zeros(shape, device='cuda')
    a0 = memory_allocated()
    y = f(x)
    usage = memory_allocated() - a0
    del y
    del x
    return usage

def filesize_in_bytes(filename):
    return Path(filename).stat().st_size
