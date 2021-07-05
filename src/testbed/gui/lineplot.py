import time, math
import numpy as np
from .plot import Plot

class LinePlot(Plot):
    def __init__(self, X=None, Y=None):
        super().__init__()
        self.add_line(X, Y)

    def add_line(self, X=None, Y=None):
        if X is None and Y is None:
            return
        if Y is None:
            if len(X) == 0:
                raise RuntimeError("Calling Plot([]) is an error.")
            elif type(X[0]) is not tuple:
                Y = np.array(X)
                X = np.array(range(len(Y)))
            elif len(X[0]) == 2:
                X = np.array(X)
                x = X[:,0]
                y = X[:,1]
                X = x
                Y = y
            else:
                raise RuntimeError(f"Cannot construct LinePlot(X,Y) with arguments X={X} and Y={Y}.")

        self.count += 1
        self.bokeh["line_"+str(self.count)] = self.bokeh["figure"].line(X,Y)
