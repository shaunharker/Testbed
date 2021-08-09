from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
import time, math
import asyncio

class StatsTicker:
    def __init__(self, metrics, x='time', y='mean_loss'):
        self.metrics = metrics
        self.x = x
        self.y = y
        self.bokeh = {}
        self.task = None

    def __repr__(self):
        self.bokeh["figure"] = figure(x_axis_label=self.x.replace('_',' '), y_axis_label=self.y.replace('_',' '), tools="pan,wheel_zoom,box_zoom,reset")
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        self.bokeh["data"] = self.bokeh["figure"].line([],[])
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        if self.task is None:
            self.task = asyncio.create_task(StatsTicker.loop(self.metrics, self.x, self.y, self.bokeh, self.bokeh_handle))
        return ""

    def __del__(self):
        try:
            self.task.cancel()
        except:
            pass

    @staticmethod
    async def loop(metrics, x, y, bokeh, bokeh_handle):
        tick = 0
        while True:
            try:
                await asyncio.sleep(1)
                data = {x: [item[x] for item in metrics[tick:]], y: [item[y] for item in metrics[tick:]]}
                if len(data) > 0:
                    bokeh["data"].data_source.stream({'x': data[x], 'y': data[y]})
                    tick = len(metrics)
                push_notebook(handle=bokeh_handle)
            except Exception as e:
                break
