import torch
from sortedcontainers import SortedList
import numpy as np

# Accumulation Tracking Structure

class Accumulator:
    def __init__(self):
        self.history = []
    def step(self, item):
        item = np.array(item)
        if len(self.history) > 0:
            last_tracked_item = self.history[-1]
            self.history.append(last_tracked_item+item)
        else:
            self.history.append(item)

# Exponential Moving Average
class EMA:
    def __init__(self, param, x=None):
        self.param = param
        self.x = x

    def step(self, x):
        if self.x is None:
            self.x = x
        else:
            self.x.mul_(self.param).add_(x, alpha=1-self.param) # self.x = self.param*self.x + (1-self.param)*x
        return self.x

    def __call__(self):
        return self.x


# Median Tracking Structure

class MedianTracker:
    def __init__(self, memory_limit=1024):
        self.memory_limit = memory_limit
        self.reps = SortedList()
    def step(self, item):
        self.bins.add(item)
        if len(self.bins) == self.memory_limit:
            # Ran out of spots. Kick out the outliers, this is median club!
            self.bins.pop(0)
            self.bins.pop(-1)
    def median(self):
        L = self.bins
        n = len(L)
        if len(L) % 2 == 0:
            return (L[n//2-1] + L[n//2])/2
        else:
            return L[n//2]
