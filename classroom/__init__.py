# classroom/__init__.py
# Shaun Harker
# 2021-06-20
import torch

import classroom.dataset
import classroom.gui
import classroom.model
import classroom.trainer

from classroom.dataset import GutenbergBitsDataset
from classroom.dataset import GutenbergBytesDataset
from classroom.dataset import GutenbergGPT2Dataset

from classroom.model import MLPLM
from classroom.model import MyLM
from classroom.model import ABPCNLM
from classroom.model import TransformerLM
from classroom.model import MinervaNLM

from classroom.optimizer import AdamW
from classroom.trainer import Trainer
from classroom.gui import Plot

def numel(model):
    return sum(p.numel() for p in model.parameters())

from math import log
lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8 vs gutenberg.gpt2, encoding the same text
