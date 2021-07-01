from bokeh.io import show
from bokeh.models import HoverTool
from bokeh.plotting import figure


class Plot:
    def __init__(self, X=None, Y=None):
        self.bokeh = {}
        self.count = 0
        self.bokeh["figure"] = figure(tools="pan,wheel_zoom,box_zoom,reset")
        self.bokeh["figure"].axis.major_label_text_font_size = "24px"
        self.hover = HoverTool(show_arrow=True,
                               mode='vline',
                               line_policy='next',
                               tooltips=[('X_value', '$data_x'),
                                         ('Y_value', '$data_y')])
        self.bokeh["figure"].add_tools(self.hover)

    def show(self):
        self.bokeh_handle = show(self.bokeh["figure"], notebook_handle=True)
        return self.bokeh_handle

    def __repr__(self):
        self.show()
        return ""
