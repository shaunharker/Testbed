from math import log
from .filters import factory
from .filters import KalmanFilter1D
from .filters import MedianFilter
from .filters import TwoWindowFilter
from .filters import CountFilter
from .filters import SumFilter
import .live

lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8

def numel(model):
    return sum(p.numel() for p in model.parameters())
