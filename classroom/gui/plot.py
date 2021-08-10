from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
from bokeh.models import HoverTool
from bokeh.palettes import Spectral4
import scipy.ndimage
import numpy as np
import asyncio
from collections import defaultdict

class Plot:
    def __init__(self, **plots):
        self.bokeh = {}
        self.count = 0
        if "x" in plots:
            self.x = plots["x"]
            del plots["x"]
        else:
            self.x = "time"
        if "y" in plots:
            self.y = plots["y"]
            del plots["y"]
        else:
            self.y = "mean_loss"
        self.task = None
        self.plots = plots

    def __repr__(self):
        self.bokeh["figure"] = figure(x_axis_label=self.x.replace('_',' '), y_axis_label=self.y.replace('_',' '), tools="pan,wheel_zoom,box_zoom,reset")
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        self.hover = HoverTool(show_arrow=True, mode='vline', line_policy='next', tooltips=[('X_value', '$data_x'), ('Y_value', '$data_y')])
        self.bokeh["figure"].add_tools(self.hover)
        for name in self.plots:
            self.count += 1
            self.bokeh[name] = self.bokeh["figure"].line([], [], line_width=2, color=Spectral4[(self.count-1)%4], alpha=.8, legend_label=name)
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        if self.task is None:
            self.task = asyncio.create_task(Plot.loop(self.plots, self.bokeh, self.bokeh_handle))
        return ""

    def add_histogram(self, data, bins=100, range=None):
        if data is None:
            return
        hist, edges = np.histogram(data, density=True, bins=bins, range=range)
        self.count += 1
        self.bokeh["figure"].y_range.start = 0
        self.bokeh["figure"].grid.grid_line_color="white"
        self.bokeh["histogram_"+str(self.count)] = (self.bokeh["figure"].quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color=["red","navy","green","orange"][self.count%4], line_color="white", alpha=0.5))

    def show(self):
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        return self.bokeh_handle

    def __del__(self):
        try:
            self.task.cancel()
        except:
            pass

    @staticmethod
    async def loop(plots, bokeh, bokeh_handle):

        tick = defaultdict(lambda: 0)
        while True:
            try:
                await asyncio.sleep(.1)
                try:
                    for name in plots:
                        t = tick[name]
                        X,Y = list(plots[name])
                        xdata = X[t:]
                        ydata = Y[t:]
                        n = min(len(xdata), len(ydata))
                        xdata = xdata[:n]
                        ydata = ydata[:n]
                        if n > 0:
                            bokeh[name].data_source.stream({'x': xdata, 'y': ydata})
                        tick[name] += n
                    push_notebook(handle=bokeh_handle)
                except Exception as e:
                    print(1, e)
            except Exception as e:
                print(2, e)
                break
