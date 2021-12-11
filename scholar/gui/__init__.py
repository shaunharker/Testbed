from bokeh.io import output_notebook
from .plot import Plot
from .fun import Fun
from .filters import Count
from .filters import Diff
from .filters import Sum
from .filters import Log2Sum
from .filters import KalmanFilter1D
from .filters import MedianFilter
from .filters import TwoWindowFilter

def turn_on_notebook_plotting():
    """
    Call this in a jupyter notebook to get the notebook plots to work.
    """
    output_notebook()
