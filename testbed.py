# testbed.py
# Shaun Harker
# 2021-06-20

import copy, time, datetime, os, signal
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from stopwatch import Stopwatch

import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue

from random import randint, randrange
from statistics import median
from transformer import Transformer

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
                 N=64,
                 B=64,
                 batch_first=True,
                 device=None):
        if device is None:
            device = default_device()
        self.device = device
        self.N = N
        self.B = B
        self.batch_first = batch_first
        with open('minicorpus.txt', 'r') as infile:
            self.text = infile.read()
        try:
            self.tokens = torch.load('minicorpus.pt').to(device)
        except:
            self.tokens = torch.tensor(list(bytes(self.text, 'utf-8'))).byte()
            torch.save(self.tokens, 'minicorpus.pt')
        D = len(self.tokens) // (N*B)
        self.D = D
        self.perm = list(range(self.D))
        self.ready = False

    def set_batch_size(self, B):
        self.B = B
        N = self.N
        D = len(self.tokens) // (N*B)
        self.D = D
        device = self.device
        if self.batch_first:
            self.batches = self.tokens[:D*B*N].view(B,D,N).transpose(0,1).contiguous().to(device)
        else:
            self.batches = self.tokens[:D*B*N].view(B,D,N).transpose(0,1).transpose(1,2).contiguous().to(device)

    def __getitem__(self, idx):
        if not self.ready:
            self.set_batch_size(self.B)
            self.ready = True
        idx = self.perm[idx]
        return self.batches[idx].long()

    def __len__(self):
        return self.D

    def set_permutation(self, perm):
        self.perm = perm

    def random_text_snippet(self, N):
        idx = randrange(len(self.text) - N)
        return self.text[idx:idx+N]

    def inspect(self, idx):
        batch = self[idx].tolist()
        return [decode_broken_utf8(example) for example in batch]

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
        self.batch_first = True

    def name(self):
        return f"net0_H{self.H}_L{self.L}_K{self.K}_C{self.C}"

    def forward(self, X):
        """
        Requires N = L + 1, where X.shape == [N, B]
        """
        L = self.L
        K = self.K
        x = self.embedding(X[:,-L-1:-1,:]) # x.shape == [B, L, K]
        y = X[:,-1].view(-1) # y.shape == [B]
        x = x.view(-1,L*K)  # s.shape == [B, L*K]
        x = self.fc2(torch.sigmoid(self.fc1(x))) # x.shape == [B, C]
        loss = self.criterion(x,y)
        return loss

    def probs(self, X):
        L = self.L
        K = self.K
        x = self.embedding(X[:,-L-1:-1,:]) # x.shape == [B, L, K]
        x = x.view(-1,L*K)  # s.shape == [B, L*K]
        x = self.fc2(torch.sigmoid(self.fc1(x))) # x.shape == [B, C]
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
        self.batch_first = True

    def name(self):
        return f"net1_H{self.H}_L{self.L}_K{self.K}_C{self.C}"

    def forward(self, X):
        """
        Input:
        X is a Tensor with shape [N, B] holding integers in {0...K-1}
        We understand this as columns of contiguous text where {0...K-1} is
        the alphabet. N > L required.

        N is batch example length.
        B is number of examples in a batch (i.e. batch size)
        K is embedding dimension
        C is number of classifications ()
        """
        x = self.embedding(X[:,:-1]) # x.shape == [B, N-1, K]
        y = X[:,self.L:] # y.shape == [B, N-L]
        x = x.transpose(1,2) # x.shape == [B, K, N-1]
        x = self.layer0(x) # x.shape == [B, H, N-L]
        x = x.transpose(1,2) # x.shape == [B, N-L, H]
        x = torch.sigmoid(x) # x.shape == [B, N-L, H]
        x = self.layer1(x) # x.shape == [B, N-L, C]
        x = x.view(-1, self.C) # x.shape = [B*(N-L), C]
        y = y.reshape(-1) # y.shape == [B*(N-L)]
        return self.criterion(x, y)

    def probs(self, X): # X.shape == [B, N]
        x = self.embedding(X) # x.shape == [B, N, K]
        x = x.transpose(1,2) # x.shape == [B, K, N]
        x = self.layer0(x) # x.shape == [B, H, N-L+1]
        x = x.transpose(1,2) # x.shape == [B, N-L+1, H]
        x = torch.sigmoid(x) # x.shape == [B, N-L+1, H]
        x = self.layer1(x) # x.shape == [B, N-L+1, C]
        P = self.softmax(x)[:,-1] # P.shape == [1, C]
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

class IgnoreKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Ignoring KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        #if self.signal_received:
        #    self.old_handler(*self.signal_received)

def trainer_worker(inbox, outbox, loss_outbox):
    with IgnoreKeyboardInterrupt():
        outbox.put("ready")
        model = inbox.get()
        dataset = inbox.get()
        UserOptimizer = inbox.get()
        batch_size = inbox.get()
        shuffle = inbox.get()
        optimizer = UserOptimizer(model.parameters())
        reporter = Reporter()
        parent = torch.multiprocessing.parent_process()
        compute_time = 0.0
        waiting = False
        dataset.set_batch_size(batch_size)
        model.train()
        while True:
            if not waiting:
                print(f"Waiting for instruction. {reporter.n} steps so far.")
                waiting = True
            try:
                instruction = inbox.get(True,1.0)
                waiting = False
            except:
                if not parent.is_alive():
                    print("Orphaned, exiting.")
                    instruction = "stop"
                    situation = "break"
                    break
                continue
            print(f"Received instruction '{instruction}'.")
            if instruction == "start":
                with Stopwatch() as stopwatch:
                    situation = "normal"
                    while situation == "normal":
                        print(f"Beginning epoch. batch_size={batch_size}, shuffle={shuffle}")
                        for X in DataLoader(dataset, batch_size=None, shuffle=shuffle):
                            optimizer.zero_grad()
                            loss = model(X)
                            loss.backward()
                            optimizer.step()
                            reporter.step(loss.item())
                            loss_outbox.put((reporter.n, compute_time + stopwatch.time_elapsed, loss.item()))
                            try:
                                instruction = inbox.get(False)
                                if instruction != "start":
                                    print(f"Interrupted by instruction '{instruction}'.")
                                    situation = "break"
                                    break
                            except:
                                pass
                compute_time += stopwatch.total_run_time
                print("Exiting compute loop.")
            if instruction == "pause":
                outbox.put("paused")
                continue
            if instruction == "stop":
                break
            if instruction == "set_batch_size":
                print("Setting new batch size.")
                batch_size = inbox.get()
                dataset.set_batch_size(batch_size)
                continue
            if instruction == "set_batch_permutation":
                print("Receiving permutation.")
                perm = inbox.get()
                print("Setting new permutation.")
                dataset.set_permutation(perm)
                continue
    print("Exiting process.")

class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer_class=None,
                 batch_size=None,
                 shuffle=True):
        if optimizer_class is None:
           optimizer_class = torch.optim.AdamW
        if batch_size is None:
            batch_size = 1
        self.model = model
        self.dataset = dataset
        self.optimizer_class = optimizer_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inbox = Queue()
        self.outbox = Queue()
        self.running = False
        self.paused = None
        self.loss_inbox = Queue()
        self.loss_stream_pos = 0
        self.data = []

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        if self.running == True:
            self.outbox.put("set_batch_size")
            self.outbox.put(self.batch_size)
            if self.paused == False:
                self.outbox.put("start")

    def set_batch_permutation(self, perm):
        self.dataset.set_permutation(perm)
        if self.running == True:
            self.outbox.put("set_batch_permutation")
            self.outbox.put(perm)
            if self.paused == False:
                self.outbox.put("start")

    def status(self):
        return f"Running: {self.running}\nPaused: {self.paused}"

    def start(self):
        if self.running == False:
            self.process = Process(target=trainer_worker, args=(self.outbox, self.inbox, self.loss_inbox))
            self.process.start()
            ready = self.inbox.get() # Wait for ready.
            self.outbox.put(self.model)
            self.outbox.put(self.dataset)
            self.outbox.put(self.optimizer_class)
            self.outbox.put(self.batch_size)
            self.outbox.put(self.shuffle)
            self.running = True
        self.outbox.put("start")
        self.running = True
        self.paused = False

    def loss(self):
        start = len(self.data)
        if self.running == True:
            while True:
                try:
                    item = self.loss_inbox.get(False)
                except:
                    break
                self.data.append(item)
            return self.data[start:]
        else:
            return []

    def pause(self):
        """
        Note: blocks until the remote is paused
        """
        if self.running == True:
            self.outbox.put("pause")
            self.paused = True
            self.inbox.get()

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
        self.model.eval()
        was_paused = self.running and self.paused
        self.pause()
        L = self.dataset.N - 1
        prompt = [b for b in bytes(self.dataset.random_text_snippet(L) + prompt, 'utf-8')][-L:]
        completion = []
        tail = prompt
        for _ in range(N):
            x = (torch.tensor(tail)
                      .unsqueeze(0)
                      .to(default_device())) # shape [1,L]
            P = self.model.probs(x).view(-1)
            prob_dist = torch.distributions.Categorical(P)
            c_ord = prob_dist.sample().item()
            tail = tail[1:] + [c_ord]
            completion += [c_ord]
        print(decode_broken_utf8(bytes(prompt)+bytes("\n~AUTOCOMPLETE~\n",'utf-8') + bytes(completion)))
        if not was_paused:
            self.start()
        return decode_broken_utf8(bytes(completion))
