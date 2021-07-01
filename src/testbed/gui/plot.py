from bokeh.io import push_notebook, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
import time, math
import numpy as np

class Plot:
    def __init__(self, X=None, Y=None):
        self.bokeh = {}
        self.count = 0
        TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"
        self.bokeh["figure"] = figure(tools=TOOLS)
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        #hover = HoverTool(tooltips=None, mode="vline")
        hover = HoverTool(show_arrow=True,
                          mode='vline',
                          line_policy='next',
                          tooltips=[('X_value', '$data_x'),
                                    ('Y_value', '$data_y')])
        self.bokeh["figure"].add_tools(hover)
        self.add_line(X, Y)

    def add_line(self, X=None, Y=None):
        if X is None and Y is None:
            return
        if Y is None:
            if len(X) == 0:
                raise RuntimeError("Calling Plot([]) is an error.")
            elif type(X[0]) is not tuple:
                Y = np.array(X)
                X = np.array(range(len(Y)))
            elif len(X[0]) == 2:
                X = np.array(X)
                x = X[:,0]
                y = X[:,1]
                X = x
                Y = y
            else:
                raise RuntimeError(f"Cannot construct Plot(X,Y) with X={X} and Y={Y} is invalid.")

        self.count += 1
        self.bokeh["line"+str(self.count)] = self.bokeh["figure"].line(X,Y)

    def show(self):
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        return self.bokeh_handle

    def __repr__(self):
        self.show()
        return "Line Plot"
