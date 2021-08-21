import asyncio
from collections import defaultdict
import numpy as np
from bokeh.io import push_notebook, show
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import Spectral4


class Plot:
    def __init__(self, legend=True, **plots):
        self.legend = legend
        self.bokeh = {}
        if "x" in plots:
            self.x = plots["x"]
            del plots["x"]
        else:
            self.x = "x"
        if "y" in plots:
            self.y = plots["y"]
            del plots["y"]
        else:
            self.y = "y"
        self.task = None
        self.plots = plots

    def __repr__(self):
        self.bokeh["figure"] = figure(x_axis_label=self.x.replace('_',' '), y_axis_label=self.y.replace('_',' '), tools="pan,wheel_zoom,box_zoom,reset")
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        self.hover = HoverTool(show_arrow=True, mode='vline', line_policy='next', tooltips=[('X_value', '$data_x'), ('Y_value', '$data_y')])
        self.bokeh["figure"].add_tools(self.hover)
        for (idx, name) in enumerate(self.plots):
            if self.legend:
                legend_label = name
            else:
                legend_label = None
            self.bokeh[name] = self.bokeh["figure"].line([], [], line_width=2, color=Spectral4[idx%4], alpha=.8, legend_label=legend_label)

        if self.legend:
            self.bokeh["figure"].legend.location = "bottom_right"
        #self.bokeh["figure"].add_layout(self.bokeh["figure"].legend[0], 'right')
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        if self.task is None:
            self.task = asyncio.create_task(Plot.loop(self.plots, self.bokeh, self.bokeh_handle))
        return ""

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
                await asyncio.sleep(1.0)
                try:
                    for name in plots:
                        t = tick[name]
                        (X,Y) = plots[name]
                        try:
                            xdata = X[t:]
                            ydata = Y[t:]
                        except:
                            xdata = X.output[t:]
                            ydata = Y.output[t:]  # TODO: make this unnecessary
                        n = min(len(xdata), len(ydata))
                        xdata = xdata[:n]
                        ydata = ydata[:n]
                        # if t == 0:
                        #     # only see tail end, reduce need to zoom
                        #     xdata = xdata[n//2:]
                        #     ydata = ydata[n//2:]
                        if n > 0:
                            bokeh[name].data_source.stream({'x': xdata, 'y': ydata})
                        tick[name] += n
                    push_notebook(handle=bokeh_handle)
                except Exception as e:
                    print(1, e)
            except Exception as e:
                print(2, e)
                break
