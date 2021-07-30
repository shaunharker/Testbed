import torch
from math import log
from .delaykeyboardinterrupt import DelayKeyboardInterrupt
from .ignorekeyboardinterrupt import IgnoreKeyboardInterrupt
from .stopwatch import Stopwatch
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

# UTF-8
def random_utf8_sequence(n_bytes=128, path=None):
    if path is None:
        path = default_utf8_path()
    N = filesize_in_bytes(filename)
    offset = randrange(N - n_bytes)
    result = np.fromfile(
        filename,
        dtype=np.ubyte,
        count=n_bytes,
        sep='',
        offset=offset)
    return result

def decode_broken_utf8(s):
    def charsize(b):
        if b&128 == 0:
            return 1
        elif b&(128+64+32) == (128+64):
            return 2
        elif b&(128+64+32+16) == (128+64+32):
            return 3
        elif b&(128+64+32+16+8) == (128+64+32+16):
            return 4
        return 0

    def is_payload(b):
        return b&(128+64) == 128

    repaired = []
    i=0
    while i < len(s):
        j = charsize(s[i])
        if j == 0 or i+j > len(s) or not all(is_payload(b) for b in s[i+1:i+j]):
            repaired += [239, 191, 189] #�
            i = i + 1
            continue
        else:
            repaired += [b for b in s[i:i+j]]
            i = i + j

    return bytes(repaired).decode('utf-8')

def construct_if_required(x):
    """
    Used for handling inputs of Maybe-Constructed things.
    A Maybe-Constructed thing is either:
    1. A dictionary of the form

        {
         "type": T,
         "args": args, # optional
         "kwargs": kwargs # optional
        }

    from which we may construct the object

        T(*args, **kwargs),

    which is returned, or
    2. Anything else, which is returned unchanged, as
       it is deemed already constructed.
    """
    if type(x) == dict:
        try:
            T = x["type"]
            args = x["args"] if "args" in x else []
            kwargs = x["kwargs"] if "kwargs" in x else {}
            return T(*args, **kwargs)
        except:
            pass
    return x
