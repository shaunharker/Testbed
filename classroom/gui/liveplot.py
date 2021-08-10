from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
import time, math
import asyncio
from collections import defaultdict

class LivePlot:
    def __init__(self, plots, x='time', y='mean_loss'):
        self.plots = plots
        self.x = x
        self.y = y
        self.bokeh = {}
        self.task = None

    def __repr__(self):
        self.bokeh["figure"] = figure(x_axis_label=self.x.replace('_',' '), y_axis_label=self.y.replace('_',' '), tools="pan,wheel_zoom,box_zoom,reset")
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        for name in self.plots:
            self.bokeh[name] = self.bokeh["figure"].line([], [], line_width=2, color=self.plots[name]["color"], alpha=.8, legend_label=name)
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        if self.task is None:
            self.task = asyncio.create_task(LivePlot.loop(self.plots, self.x, self.y, self.bokeh, self.bokeh_handle))
        return ""

    def __del__(self):
        try:
            self.task.cancel()
        except:
            pass

    @staticmethod
    async def loop(plots, x, y, bokeh, bokeh_handle):
        tick = defaultdict(lambda: 0)
        while True:
            try:
                await asyncio.sleep(1)
                for name in plots:
                    t = tick[name]
                    X = plots[name]["x"]
                    Y = plots[name]["y"]
                    xdata = X[t:]
                    ydata = Y[t:]
                    n = len(xdata)
                    if n > 0:
                        bokeh[name].data_source.stream({'x': xdata, 'y': ydata})
                    tick[name] += n
                push_notebook(handle=bokeh_handle)
            except Exception as e:
                break
