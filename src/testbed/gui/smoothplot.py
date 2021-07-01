import numpy as np
from .lineplot import LinePlot

def smoother(data, lag):
    cs = np.cumsum(data)
    return (cs[lag:] - cs[:-lag])/lag

class SmoothPlot(LinePlot):
    def __init__(self, X=None, Y=None, lag=100):
        if X is not None:
            if Y is None:
                Y = np.array(X)
                X = np.array(range(len(X)))
            else:
                X = np.array(X)
                Y = np.array(Y)
            X = X[lag:]
            Y = smoother(Y, lag)
        super().__init__(X, Y)
