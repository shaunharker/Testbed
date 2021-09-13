import torch
from pydivsufsort import divsufsort, kasai, sa_search
from random import choices, randrange
from math import log
from time import time
import numpy as np
from collections import defaultdict
from classroom import GutenbergSnippetsDataset as Dataset
import numba
from numba import njit, guvectorize, int64
import numpy as np

## datasets
def gutenberg(idx):
    with open('/home/sharker/data/gutenberg.utf8', 'rb') as infile:
        infile.seek(idx*2**25)
        return infile.read(2**25)

def pile(idx):
    with open('/home/sharker/data/thepile/00.jsonl', 'rb') as infile:
        infile.seek(idx*2**25)
        return infile.read(2**25)

## suffix array tools
def sa_count(a, sa, bs):
    return sa_search(a, sa, bs)[0]

def sa_dist(a, sa, bs):
    return [sa_search(a, sa, bs + bytes([i]))[0] for i in range(256)]

def sa_autocomplete(a, sa, n_generate, k=16):
    bs = b""
    bt = b""
    for k in range(n_generate):
        while sa_count(a, sa, bs) < k:
            bs = bs[1:]
        choice = choices(range(256), sa_dist(a, sa, bs))
        bs += bytes(choice)
        bt += bytes(choice)
    return bt

def sa_test(a, sa, examples, k=16):
    bits = 0
    characters = len(examples)
    for example in examples:
        while sa_count(a, sa, example[:-1]) < k:
            example = example[1:]
        dist = sa_dist(a, sa, example[:-1])
        p = max(1/256, dist[example[-1]]/sum(dist))
        bits += -log(p)/log(2)
    return bits/characters

def sa_shard_test(a, sa, examples):
    return (
        np.stack([
            np.stack([
                    vec(sa_count(a, sa, bytes(example[idx:].tolist())),
                        sa_count(a, sa, bytes(example[idx:-1].tolist())))
                    for idx in range(len(example))])
            for example in examples]))

## entropy
def shannon_entropy(dist):
    entropy = 0
    N = sum(dist[k] for k in dist)
    for (k,v) in dist.items():
        p = v/N
        entropy += -p*log(p)/log(2)
    return entropy


## number theory
def bezout(a, b):
    if a == 0:
        return b,0,1
    d, u, v = bezout(b%a, a)
    x = v - (b//a) * u
    y = u
    return d, x, y

def modinv(x, p):
    d, a, b = bezout(p, x)
    assert d == 1
    # a*p + b*x = 1, so...
    return b

p = 2**32 - 5
seed = torch.initial_seed()
torch.manual_seed(42)
A = torch.randint(low=1, high=p,size=[256], dtype=torch.long, device='cuda')
B = torch.randint(low=0, high=p,size=[256], dtype=torch.long, device='cuda')
C = torch.randint(low=0, high=p,size=[256], dtype=torch.long, device='cuda')
D = torch.tensor([(((B[idx].item()*C[idx].item())%p+1)*modinv(A[idx].item(), p))%p
                  for idx in range(256)], dtype=torch.long, device='cuda')
torch.maual_seed(seed)  # todo: use a context manager for this sort of thing

def ngram_count_generator(n=8, n_shards=4):
    for shard_idx in range(n_shards):
        chunk = gutenberg(shard_idx)
        print(f"{time()-start}. Got chunk. length = {len(chunk)}")
        index = torch.tensor(np.frombuffer(chunk, dtype=np.uint8),device='cuda').long()
        a = torch.gather(input=A, dim=0, index=index)
        b = torch.gather(input=B, dim=0, index=index)
        c = torch.gather(input=C, dim=0, index=index)
        d = torch.gather(input=D, dim=0, index=index)
        print(f"{time()-start}. Translated chunk to character matrices.")
        k = 1
        while k < n:
            (a,b,
             c,d) = (
                a[:-k]*a[k:]+b[:-k]*c[k:], a[:-k]*b[k:]+b[:-k]*d[k:],
                c[:-k]*a[k:]+d[:-k]*c[k:], c[:-k]*b[k:]+d[:-k]*d[k:]
            )
            a = torch.fmod(a, 2**32-5)
            b = torch.fmod(b, 2**32-5)
            c = torch.fmod(c, 2**32-5)
            d = torch.fmod(d, 2**32-5)
            k = k*2
        Y = 2**31 * b + c  # todo: get that extra bit despite signed integer type
        print(f"{time()-start}. Computed substring hashes.")
        yield torch.stack(torch.unique(Y,return_counts=True))

@guvectorize([(int64[:], int64[:], int64[:], int64[:],
               int64[:], int64[:], int64[:], int64[:])],
             '(n),(n),(n),(n),(m),(m)->(m),(m)')
def mergecounts(x1, y1, x2, y2, x, y, X, Y):
    i1 = 0
    i2 = 0
    j = 0
    X[0] = min(x1[0], x2[0])
    Y[0] = 0
    while i1 < x1.shape[0] or i2 < x2.shape[0]:
        if i1 < x1.shape[0] and x1[i1] == X[j]:
            Y[j] += y1[i1]
            i1 += 1
        elif i2 < x2.shape[0] and x2[i2] == X[j]:
            Y[j] += y2[i2]
            i2 += 1
        else:
            if i1 < x1.shape[0] and i2 < x2.shape[0]:
                j += 1
                X[j] = min(x1[i1], x2[i2])
                Y[j] = 0
            elif i1 < x1.shape[0]:
                j += 1
                X[j] = x1[i1]
                Y[j] = 0
            elif i2 < x2.shape[0]:
                j += 1
                X[j] = x2[i2]
                Y[j] = 0
    X[-1] = j + 1

def merge(item1, item2):
    x1 = item1[0]
    y1 = item1[1]
    x2 = item2[0]
    y2 = item2[1]
    L = x1.shape[0] + x2.shape[0]
    X = np.zeros(L, np.int64)
    Y = np.zeros(L, np.int64)
    mergecounts(x1, y1, x2, y2, X, Y, X, Y)
    N = X[-1]
    X = X[:N]
    Y = Y[:N]
    return np.stack((X, Y))

def ngrams(n, n_shards):
    items = {}
    max_sz = 0
    for item in ngram_count_generator(n, n_shards):
        sz = 1
        while sz in items:
            item = merge(items[sz], item)
            del items[sz]
            sz += 1
        items[sz] = item
        max_sz = max(sz, max_sz)
    item = None
    # print(f"items = {items}")
    for sz in range(max_sz+1):
        if sz in items:
            if item is None:
                item = items[sz]
            else:
                item = merge(items[sz], item)
            del items[sz]
    return item

## assorted unused utilities
vec = lambda x,y: np.array([x,y],dtype=np.int64)  # todo check if used

def pad_to_power_of_two(Z,dim=-1):
    n = Z.shape[dim]
    k = 0
    while n > 2**k:
        k = k + 1
    return torch.nn.functional.pad(Z,(0, 2**k - n)), k

def sort_by_first_row(x):
    return x[:, torch.sort(x[0]).indices]

def sort_by_second_row(x):
    return x[:, torch.sort(x[1]).indices]
