import numpy as np
from bokeh.io import show
from bokeh.models import HoverTool
from bokeh.plotting import figure

def smoother(data, lag):
    cs = np.cumsum(data)
    return (cs[lag:] - cs[:-lag])/lag

class SmoothPlot:
    def __init__(self, x, y=None, lag=100):
        self.lag = lag
        self.logfig = figure(tools="pan,wheel_zoom,box_zoom,reset")
        self.logfig.axis.major_label_text_font_size = "24px"
        self.hover = HoverTool(show_arrow=False,
                               mode='vline',
                               line_policy='next',
                               tooltips=[('X_value', '$data_x'),
                                         ('Y_value', '$data_y')])
        self.logfig.add_tools(self.hover)
        if y is None:
            self.X = np.array(range(len(x)))
            self.Y = np.array(x)
        else:
            self.X = np.array(x)
            self.Y = np.array(y)
        self.X = smoother(self.X, self.lag)
        self.Y = self.Y[self.lag:]
        self.logline = self.logfig.line(self.X,self.Y)
        self.bokeh_handle = show(self.logfig)
