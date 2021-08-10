import time, math
import numpy as np
from .plot import Plot
from bokeh.palettes import Spectral4

class LinePlot(Plot):
    def __init__(self, lines):
        super().__init__()
        self.count = 0
        for k in lines:
            self.add_line(k, lines[k]["x"], lines[k]["y"])

    def add_line(self, k, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        self.count += 1
        self.bokeh[k] = self.bokeh["figure"].line(X, Y, line_width=2, color=Spectral4[(self.count-1)%4], alpha=.8, legend_label=k)
