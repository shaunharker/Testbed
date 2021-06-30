from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
output_notebook()
import time, math
import numpy as np
from threading import Thread, Lock

class StatsTicker:
    def __init__(self, trainer):
        self.trainer = trainer
        self.tick = 0
        self.losses = self.trainer.update_losses()
        self.bokeh = {}
        self.bokeh_handle = None
        self.updating = False
        self.display()
        
    def recent_stats(self):
        self.losses = self.trainer.update_losses()
        data = {'time' : [ x[1] for x in self.losses[self.tick:]],
                'mean_loss' : [8*x[2]/math.log(256) for x in self.losses[self.tick:]]}
        #print(data)
        return data

    def display(self, updates=True):
        TOOLS="pan,wheel_zoom,box_zoom,reset"
        self.bokeh["figure"] = figure(tools=TOOLS)
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        hover = HoverTool(show_arrow=False,
                          mode='vline',
                          line_policy='next',
                          tooltips=[('X_value', '$data_x'),
                                    ('Y_value', '$data_y')])
        self.bokeh["figure"].add_tools(hover)
        data = self.recent_stats()
        self.bokeh["mean_loss"] = self.bokeh["figure"].line(data['time'],data['mean_loss'])
        self.tick = len(self.losses)
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

    def _update_loop(self):
        while self.updating:
            time.sleep(1)
            data = self.recent_stats()
            if len(self.losses) > self.tick:
                self.bokeh["mean_loss"].data_source.stream({'x':data['time'],
                                                            'y':data['mean_loss']})
                self.tick = len(self.losses)
            push_notebook(handle=self.bokeh_handle)
