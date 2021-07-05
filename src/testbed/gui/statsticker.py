from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
output_notebook()
import time, math
import numpy as np
from threading import Thread, Lock

class StatsTicker:
    def __init__(self, trainer, x='compute_time', y='mean_loss', kind='line'):
        self.trainer = trainer
        self.tick = 0
        self.losses = []
        self.bokeh = {}
        self.bokeh_handle = None
        self.updating = False
        self.kind = kind
        self.x = x
        self.y = y

    def __repr__(self):
        self.display()
        return ""

    def display(self, updates=True):
        TOOLS="pan,wheel_zoom,box_zoom,reset"
        self.bokeh["figure"] = figure(tools=TOOLS)
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        self.trainer.update()
        self.losses = self.trainer.losses
        self.tick = 0
        data = {self.x : [], self.y : []}
        if self.kind == 'line':
            self.bokeh["data"] = self.bokeh["figure"].line(data[self.x],data[self.y])
        else:
            self.bokeh["data"] = self.bokeh["figure"].circle(data[self.x],data[self.y])
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        if updates:
            self.start()

    def start(self):
        if not self.updating:
            self.updating = True
            self.updater = Thread(target=StatsTicker._update_loop, args=(self,), daemon=True)
            self.updater.start()

    def stop(self):
        if self.updating:
            self.updating = False
            self.updater.join()

    def data_tail(self, tick=0):
        data = {self.x : [ item[self.x] for item in self.losses[tick:]],
                self.y : [ item[self.y] for item in self.losses[tick:]]}
        return data

    def _update_loop(self):
        while self.updating:
            time.sleep(1)
            self.trainer.update()
            self.losses = self.trainer.losses
            data = self.data_tail(self.tick)
            if len(data) > 0:
                self.bokeh["data"].data_source.stream({'x':data[self.x],
                                                       'y':data[self.y]})
                self.tick = len(self.losses)
            push_notebook(handle=self.bokeh_handle)
