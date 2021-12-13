# scholar/__init__.py
# Shaun Harker
# 2021-06-20
import torch

import scholar.model
import scholar.dataset
import scholar.optimizer
import scholar.trainer
import scholar.gui


def numel(model):
    return sum(p.numel() for p in model.parameters())

from math import log
lyles_constant = 9115131782/14818489608 * log(50257)/log(65536) # compression achieved via gpt2-token encoding compared to utf8-byte encoding on gutenberg.utf8 vs gutenberg.gpt2, encoding the same text
