# classroom/__init__.py
# Shaun Harker
# 2021-06-20
import classroom.dataset
import classroom.gui
import classroom.model
import classroom.student
import classroom.util

from classroom.dataset import BytesDataset
from classroom.dataset import RandomTokensDataset
from classroom.dataset import utf8encode
from classroom.dataset import utf8decode

from classroom.model import MLPLM
from classroom.model import MLPLM2
from classroom.model import TransformerLM

from classroom.optimizer import AdamW
from classroom.optimizer import Sonny
from classroom.optimizer import Floyd

from classroom.student import Student
from classroom.classroom import Classroom

from classroom.util import factory
from classroom.util import KalmanFilter1D
from classroom.util import MedianFilter
from classroom.util import TwoWindowFilter
from classroom.util import CountFilter
from classroom.util import SumFilter
from classroom.util import numel

from classroom.gui import Plot
from classroom.gui import Histogram

import classroom.util.live

def numel(model):
    return sum(p.numel() for p in model.parameters())

lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8
