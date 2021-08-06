from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
import time, math
from threading import Thread, Event, Lock

class StatsTicker:
    def __init__(self, trainer, x='time', y='mean_loss'):
        self.trainer = trainer
        self.x = x
        self.y = y
        self.stop_event = Event()
        self.bokeh = {}

    def __repr__(self):
        self.stop_event.set()
        self.stop_event = Event()
        with self.trainer.metrics_lock:
            self.bokeh["figure"] = figure(x_axis_label=self.x.replace('_',' '), y_axis_label=self.y.replace('_',' '), tools="pan,wheel_zoom,box_zoom,reset")
            self.bokeh["figure"].axis.major_label_text_font_size = "24px"
            self.bokeh["data"] = self.bokeh["figure"].line([],[])
            self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
            self.updater = Thread(target=StatsTicker._update_loop, args=(self.stop_event, self.x, self.y, self.bokeh, self.bokeh_handle, self.trainer), daemon=True)
            self.updater.start()
        return ""

    def __del__(self):
        self.stop_event.set()

    @staticmethod
    def _update_loop(stop_event, x, y, bokeh, bokeh_handle, trainer):
        tick = 0
        while not stop_event.is_set():
            try:
                time.sleep(1)
                with trainer.metrics_lock:
                    data = {x: [item[x] for item in trainer.metrics[tick:]], y: [item[y] for item in trainer.metrics[tick:]]}
                    if len(data) > 0:
                        if stop_event.is_set():
                            break
                        bokeh["data"].data_source.stream({'x': data[x], 'y': data[y]})
                        tick = len(trainer.metrics)
                push_notebook(handle=bokeh_handle)
            except Exception as e:
                print(e)
                break
