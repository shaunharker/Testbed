import torch
from .delaykeyboardinterrupt import DelayKeyboardInterrupt
from .ignorekeyboardinterrupt import IgnoreKeyboardInterrupt
from .reporter import Reporter
from .stopwatch import Stopwatch
from .plot import Plot
from .parameterinspector import ParameterInspector

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

def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def numel(model):
    return sum(p.numel() for p in model.parameters())
