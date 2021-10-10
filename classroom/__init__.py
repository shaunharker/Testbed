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
from classroom.dataset import GutenbergBitsDataset
from classroom.dataset import GutenbergBytesDataset
from classroom.dataset import GutenbergGPT2Dataset
from classroom.dataset import RandomTokensDataset
from classroom.dataset import utf8encode
from classroom.dataset import utf8decode
from classroom.dataset import utf8bitsencode
from classroom.dataset import utf8bitsdecode
from classroom.dataset import gpt2decode
from classroom.dataset import gpt2encode

from classroom.model import MLPLM
from classroom.model import MyLM
from classroom.model import ABPCNLM
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

def bit_autocomplete(model, prompt=None, n_generate=8192,
                     n_ctx=None, temp=1.0,
                     encode=None, decode=None, output=None):
    Categorical = torch.distributions.Categorical
    if n_ctx is None:
        n_ctx = model.n_ctx
    if encode is None:
        encode = utf8bitsencode
    if decode is None:
        decode = utf8bitsdecode
    if prompt is None:
        prompt = decode(student.dataset.batch(1, 2*n_ctx, offset=None).tolist()[0])  # kludge
    x = encode(prompt)
    x = x[-n_ctx:]
    prompt = decode(x)
    print(f"=== Prompt ===\n{prompt}\n=== Autocompletion ===\n")

    def sampler(x):
        x = list(x)
        for _ in range(n_generate):
            probs = model.inference(torch.tensor(x, dtype=torch.long, device="cuda").unsqueeze(0)).view(-1)[-model.n_vocab_out:]
            if temp > 0:
                y = Categorical(probs=probs**(1.0/temp)).sample().item()
            else:
                y = torch.argmax(probs).item()
            x = (x + [y])[-n_ctx:]
            if output is not None:
                output.append(y)
            yield y
    result = decode(list(sampler(x)))
    print(result)
