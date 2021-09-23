# classroom/__init__.py
# Shaun Harker
# 2021-06-20
import classroom.dataset
import classroom.gui
import classroom.model
import classroom.student
import classroom.util

from classroom.dataset import BytesDataset
from classroom.dataset import GutenbergSnippetsDataset
from classroom.dataset import BitSnippetsDataset
from classroom.dataset import RandomTokensDataset
from classroom.dataset import utf8encode
from classroom.dataset import utf8decode

from classroom.model import MLPLM, MyLM
from classroom.model import TransformerLM
from classroom.model import ngrams

from classroom.optimizer import AdamW
from classroom.optimizer import Sonny
from classroom.optimizer import Floyd

from classroom.student import Student
from classroom.student import BaselineComparison # for testing
from classroom.classroom import Classroom
from classroom.gui import Plot

from classroom.util import Fun
from classroom.util import Count
from classroom.util import Sum
from classroom.util import Diff
from classroom.util import Log2Sum
from classroom.util import KalmanFilter1D
from classroom.util import MedianFilter
from classroom.util import TwoWindowFilter

def numel(model):
    return sum(p.numel() for p in model.parameters())

from math import log
lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8
