# scholar/__init__.py
# Shaun Harker
# 2021-06-20
import torch

import scholar.dataset
import scholar.gui
import scholar.model
import scholar.trainer

from scholar.dataset import GutenbergBitsDataset
from scholar.dataset import GutenbergBytesDataset
from scholar.dataset import GutenbergGPT2Dataset

from scholar.model import MLPLM
from scholar.model import MyLM
from scholar.model import ABPCNLM
from scholar.model import TransformerLM
from scholar.model import MinervaNLM

from scholar.optimizer import AdamW
from scholar.trainer import Trainer
from scholar.gui import Plot

def numel(model):
    return sum(p.numel() for p in model.parameters())

from math import log
lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8 vs gutenberg.gpt2, encoding the same text
