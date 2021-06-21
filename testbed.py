# testbed.py
# Shaun Harker
# 2021-06-20

import copy, time, datetime, os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue

from random import randint, randrange
from statistics import median

def decode_broken_utf8(s):
    def charsize(b):
        if b&128 == 0:
            return 1
        elif b&(128+64+32) == (128+64):
            return 2
        elif b&(128+64+32+16) == (128+64+32):
            return 3
        elif b&(128+64+32+16+8) == (128+64+32+16):
            return 4
        return 0

    def is_payload(b):
        return b&(128+64) == 128

    repaired = []
    i=0
    while i < len(s):
        j = charsize(s[i])
        if j == 0 or i+j > len(s) or not all(is_payload(b) for b in s[i+1:i+j]):
            repaired += [239, 191, 189] #�
            i = i + 1
            continue
        else:
            repaired += [b for b in s[i:i+j]]
            i = i + j

    return bytes(repaired).decode('utf-8')

def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

class TextDataset:
    def __init__(self,
                 filename='minicorpus.txt',
                 B=1,
                 N=64,
                 device=None):
        with open('minicorpus.txt', 'r') as infile:
            self.text = infile.read()
        self.tokens = list(bytes(self.text, 'utf-8'))
        self.B = B
        self.N = N
        self.D = len(self.tokens) // (B*N)
        self.batches = [None]*self.D
        if device is None:
            device = default_device()
        self.device = device

    def __getitem__(self, idx):
        k = self.B*self.N
        if self.batches[idx] is None:
            self.batches[idx] = torch.tensor(self.tokens[k*idx:k*(idx+1)]).reshape(self.B, self.N)
        return self.batches[idx].to(self.device)

    def __len__(self):
        return self.D

    def random_text_snippet(self, N):
        idx = randrange(len(self.text) - N)
        return self.text[idx:idx+N]

    def inspect(self, idx):
        batch = self[idx].tolist()
        return [decode_broken_utf8(example) for example in batch]

class TextIterableDataset(IterableDataset):
    def __init__(self,
                 filename='minicorpus.txt',
                 B=1,
                 N=64,
                 shuffle=True,
                 device=None):
        super(TextIterableDataset).__init__()
        with open('minicorpus.txt', 'r') as infile:
            self.text = infile.read()
        self.B = B
        self.N = N
        self.shuffle = shuffle
        if device is None:
            device = default_device()
        self.device = device
        self.tokens = torch.tensor([ ord(c) for c in self.text if ord(c) < 256])  # 10 seconds
        self.pos = 0
        self.epoch = 0
        self.D = len(self.tokens)
        self.offset = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # in a worker process
            raise BaseException("Multiple workers not supported by TextIterableDataset yet")
        return self

    def random_text_snippet(self, N):
        idx = randrange(self.D - N)
        return text[idx:idx+N]

    def __next__(self):
        B = self.B
        N = self.N
        D = self.D
        device = self.device
        if self.shuffle:
            self.pos = randrange(D - B*N)
        if self.pos + B*N > D:
            self.offset = self.offset + 1
            if self.offset >= B*N:
                self.offset = 0
                self.epoch = self.epoch + 1
                print(f"Epoch {self.epoch}")
            self.pos = self.offset
        batch_start = self.pos
        batch_end = batch_start + B*N
        self.pos = batch_end
        X = self.tokens[batch_start:batch_end].reshape((B, N)).to(device)
        return X

class Net0(nn.Module):
    def __init__(self, H=256, L=64, K=8, C=256):
        super(Net0, self).__init__()
        self.C = C # number of classes
        self.K = K # dimension of token embedding
        self.L = L # context window length
        self.H = H # number of hidden neurons
        self.embedding = nn.Embedding(C, K)
        self.fc1 = nn.Linear(L*K, H)
        self.fc2 = nn.Linear(H, C)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)

    def name(self):
        return f"net0_H{self.H}_L{self.L}_K{self.K}_C{self.C}"

    def forward(self, X):
        L = self.L
        K = self.K
        x = self.embedding(X[...,-L-1:-1])
        y = X[...,-1].reshape((-1,))
        x = x.reshape(-1,L*K)
        x = self.fc2(torch.sigmoid(self.fc1(x)))
        loss = self.criterion(x,y)
        return loss

    def probs(self, x):
        L = self.L
        K = self.K
        x = self.embedding(x[...,-L:])
        x = x.reshape(-1,L*K)
        x = self.fc2(torch.sigmoid(self.fc1(x)))
        P = self.softmax(x)
        return P

class Net1(nn.Module):
    def __init__(self, H, L, K=8, C=256):
        super(Net1, self).__init__()
        self.C = C # number of classes
        self.K = K # dimension of token embedding
        self.L = L # context window length
        self.H = H # number of hidden neurons
        self.embedding = nn.Embedding(C, K)
        self.layer0 = nn.Conv1d(K, H, L)
        self.layer1 = nn.Linear(H, C)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def name(self):
        return f"net1_H{self.H}_L{self.L}_K{self.K}_C{self.C}"

    def forward(self, X):
        """
        Input:
        X is a Tensor with shape [B, N+1] holding integers in {0...K-1}
        We understand this as rows of text, where each continguous chunk
        of L letters is used to guess the next one.
        """
        x = self.embedding(X[...,:-1]) # x.shape == [B, N, K]
        y = X[...,self.L:] # x.shape == [B, N-L+1]

        x = torch.transpose(x, -1, -2)
        x = self.layer0(x) # x.shape == [B, H, N-L+1]
        x = torch.transpose(x, -1, -2) # x.shape == [B, N-L+1, H]
        x = torch.sigmoid(x) # x.shape == [B, N-L+1, H]
        x = self.layer1(x) # x.shape == [B, N-L+1, C]

        x = x.reshape((-1, self.C)) # x.shape = [B*(N-L+1), C]
        y = y.reshape((-1,))

        return self.criterion(x, y)

    def probs(self, X): # X.shape == [B, N]
        x = self.embedding(X) # x.shape == [B, N, K]
        x = torch.transpose(x, -1, -2) # x.shape == [B, K, N]
        x = self.layer0(x) # x.shape == [B, H, N-L+1]
        x = torch.transpose(x, -1, -2) # x.shape == [B, N-L+1, H]
        x = torch.sigmoid(x) # x.shape == [B, N-L+1, H]
        x = self.layer1(x) # x.shape == [B, N-L+1, C]
        P = self.softmax(x) # P.shape == [B, N-L+1, C]
        return P

def numel(model):
    return sum(p.numel() for p in model.parameters())

class Reporter:
    def __init__(self,
                 logfile = None,
                 time_between_report=1.0,
                 time_between_autocomplete=60.0,
                 time_between_saves=3600.0) :
        self.n = 0
        if logfile is None:
            logfile = str(time.time()) + '.log'
        self.logfile = logfile
        self.log = open(logfile, 'a')
        self.log.write('date,step,time,loss\n')

    def step(self, loss):
        self.n = self.n + 1
        self.log.write(','.join([
            datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S"),
            str(self.n),
            str(time.time()),
            str(loss)])+'\n')

def trainer_worker(inbox, outbox):
    outbox.put("ready")
    model = inbox.get()
    dataset = inbox.get()
    UserOptimizer = inbox.get()
    optimizer = UserOptimizer(model.parameters())
    reporter = Reporter()
    data = []
    while True:
        print(f"Waiting for instruction. {reporter.n} steps so far.")
        instruction = inbox.get()
        print(f"Received instruction '{instruction}'.")
        if instruction == "start":
            for X in DataLoader(dataset):
                X = X.reshape(X.shape[1:])
                try:
                    instruction = inbox.get(False)
                    if instruction != "start":
                        print(f"Interrupted by instruction '{instruction}'.")
                        break
                except:
                    pass
                optimizer.zero_grad()
                loss = model(X)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())
                data.append((reporter.n, time.time(), loss.item()))
            print("Exiting compute loop.")
        if instruction == "loss":
            print("Sending loss data.")
            outbox.put(data)
            continue
        if instruction == "pause":
            continue
        if instruction == "stop":
            break


class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer_class=None):
        if optimizer_class is None:
           optimizer_class = torch.optim.AdamW
        self.model = model
        self.dataset = dataset
        self.optimizer_class = optimizer_class
        self.inbox = Queue()
        self.outbox = Queue()
        self.running = False
        self.paused = None

    def status(self):
        return f"Running: {self.running}\nPaused: {self.paused}"

    def start(self):
        if self.running == False:
            self.process = Process(target=trainer_worker, args=(self.outbox, self.inbox))
            self.process.start()
            ready = self.inbox.get() # Wait for ready.
            self.outbox.put(self.model)
            self.outbox.put(self.dataset)
            self.outbox.put(self.optimizer_class)
            self.running = True
        self.outbox.put("start")
        self.running = True
        self.paused = False

    def loss(self):
        if self.running == True:
            self.outbox.put("loss")
            data = self.inbox.get()
            if not self.paused:
                self.outbox.put("start")
            return data
        else:
            return []

    def pause(self):
        if self.running == True:
            self.outbox.put("pause")
            self.paused = True

    def stop(self):
        if self.running == True:
            self.outbox.put("stop")
            self.running = False
            self.paused = None
            self.process.join()

    def load(self, path=None):
        # todo: default load of last step, load opt
        if path is None:
            path = self.model.name() + "_" + str(self.n) +".pt"
        torch.save(self.model, path)

    def save(self, path=None):
        # todo: save the optimizer too
        if path is None:
            path = self.model.name() + "_" + str(self.n) +".pt"
        torch.save(self.model, path)

    def autocomplete(self, prompt="", N=1024):
        was_paused = self.running and self.paused
        self.pause()
        L = model.L
        prompt = [b for b in bytes(self.dataset.random_text_snippet(L) + prompt, 'utf-8')][-L:]
        completion = []
        tail = prompt
        for _ in range(N):
            x = (torch.tensor(tail)
                     .reshape((1,L))
                     .to(default_device()))
            P = model.probs(x)
            prob_dist = torch.distributions.Categorical(P)
            c_ord = prob_dist.sample()[0]
            tail = tail[1:] + [c_ord]
            completion += [c_ord]
        print(decode_broken_utf8(bytes(prompt)+bytes("\n~AUTOCOMPLETE~\n",'utf-8') + bytes(completion)))

        if not was_paused:
            self.start()

        return decode_broken_utf8(bytes(completion))
