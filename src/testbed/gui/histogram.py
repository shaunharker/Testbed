import time, math
import numpy as np
from .plot import Plot

class Histogram(Plot):
    def __init__(self, data=None, bins=100, range=None, title='Histogram'):
        super().__init__()
        self.bokeh["figure"].y_range.start = 0
        self.bokeh["figure"].grid.grid_line_color="white"
        self.add_histogram(data,bins=bins,range=range)

    def add_histogram(self, data, bins=100, range=None):
        if data is None:
            return
        hist, edges = np.histogram(data, density=True, bins=bins, range=range)
        self.count += 1
        self.bokeh["histogram_"+str(self.count)] = (self.bokeh["figure"].quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color=["red","navy","green","orange"][self.count%4], line_color="white", alpha=0.5))
