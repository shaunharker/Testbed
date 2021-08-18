from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
from bokeh.models import HoverTool
from bokeh.palettes import Spectral4
import scipy.ndimage
import numpy as np
import asyncio
from collections import defaultdict

class Histogram:
    def __init__(self, data, bins=100, range=None):
        if data is None:
            return
        hist, edges = np.histogram(data, density=True, bins=bins, range=range)
        self.bokeh = {}
        self.bokeh["figure"] = figure(tools="pan,wheel_zoom,box_zoom,reset")
        self.bokeh["figure"].y_range.start = 0
        self.bokeh["figure"].grid.grid_line_color="white"
        self.bokeh["histogram"] = self.bokeh["figure"].quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color=["red","navy","green","orange"][1], line_color="white")

    def __repr__(self):
        self.bokeh_handle = self.show()
        return ""

    def show(self):
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        return self.bokeh_handle
