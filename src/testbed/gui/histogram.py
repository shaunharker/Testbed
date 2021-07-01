from bokeh.io import push_notebook, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
import time, math
import numpy as np

class Histogram:
    def __init__(self, data, bins=100, range=None, title='Histogram'):
        hist, edges = np.histogram(data, density=True, bins=bins, range=range)
        p = figure(title=title, tools='', background_fill_color="#fafafa")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=0.5)
        p.y_range.start = 0
        p.grid.grid_line_color="white"
        show(p)
