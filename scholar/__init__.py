# scholar/__init__.py
# Shaun Harker
# 2021-06-20
import torch

# import .model
# import .dataset
# import .optimizer
# import .trainer
# import .gui

# import scholar.model
# import scholar.dataset
# import scholar.optimizer
# import scholar.trainer
# import scholar.gui

# from .model import model
# from .dataset import dataset
# from .optimizer import optimizer
# from .trainer import trainer
# from .gui import gui

from . import model
from . import dataset
from . import optimizer
from . import trainer
from . import gui

def numel(model):
    return sum(p.numel() for p in model.parameters())

from math import log
lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8 vs gutenberg.gpt2, encoding the same text
