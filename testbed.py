import numpy as np
import math, copy, time, datetime, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.multiprocessing import Process, Queue
from random import randint, randrange

from statistics import median


def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

class TextIterableDataset(IterableDataset):
    def __init__(self,
                 filename='minicorpus.txt',
                 B=1,
                 N=64,
                 shuffle=False,
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
        batch_start = self.pos
        batch_end = batch_start + B * N
        if self.pos + B*N > D:
            self.offset = self.offset + 1
            if self.offset >= B*N:
                self.offset = 0
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
        x = self.embedding(X[...,-L-1:-1])
        y = X[...,-1].reshape((-1,))
        x = x.reshape(-1,L*K)
        x = self.fc2(torch.sigmoid(self.fc1(x)))
        loss = self.criterion(x,y)
        return loss

    def probs(self, x):
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
                 model,
                 logfile = None,
                 time_between_report=1.0,
                 time_between_autocomplete=60.0,
                 time_between_saves=3600.0) :
        self.model = model
        self.writer = None
        self.n = 0
        if logfile is None:
            logfile = str(time.time()) + '.log'
        self.logfile = logfile
        self.log = open(logfile, 'a')
        self.initial_time = time.time() #seconds since 1970 jan 1
        self.loss_series = []
        self.avg_loss_series = []
        self.time_between_report = time_between_report
        self.time_between_autocomplete = time_between_autocomplete
        self.time_between_saves = time_between_saves
        self.last_report =  self.initial_time
        self.last_autocomplete = self.initial_time
        self.last_save = self.initial_time

    def statistics(self):
        if len(self.loss_series) < 2:
            return {}

        trailing_steps = min(len(self.loss_series), 1000)
        trailing_loss = self.loss_series[-trailing_steps:]

        avg_loss = sum(trailing_loss) / len(trailing_loss)
        median_loss = median(trailing_loss)
        if trailing_steps > 1:
            var_loss = sum( (x-avg_loss)*(x-avg_loss) for x in trailing_loss ) / (trailing_steps - 1)
        else:
            var_loss = 0

        return {"date" : datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S:"),
                "step" : self.n,
                "time" : time.time(),
                "loss" : self.loss_series[-1],
                "mean_loss" : avg_loss,
                "median_loss" : median_loss,
                "var_loss" : var_loss }

    def step(self, loss):

        # Get loss sequences straightened out
        self.loss_series.append(loss)
        self.n = self.n + 1

        # Interrupts
        self.current_time = time.time()
        if self.current_time - self.last_report > self.time_between_report:
            time.time(), loss,
            #print(),
            #  f"\n    Step {self.n}\n    Elapsed {self.current_time-self.initial_time}.")
            self.last_report = self.current_time
            stats = self.statistics()
            for k in ["date", "step", "time", "loss"]:
                self.log.write(stats[k])

def trainer_worker(q):
    (model, dataloader, optimizer, reporter) = q.get()
    while True:
        instruction = q.get()
        if instruction == "start":
            for (n, X) in enumerate(dataloader):
                try:
                    instruction = q.get(False)
                    if instruction != "start":
                        break
                except:
                    pass
                optimizer.zero_grad()
                loss = model(X)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())
        if instruction == "pause":
            continue
        if instruction == "stop":
            break
        if instruction == "sync":
            q.put((model, dataloader, optimizer, reporter))

class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 dataloader,
                 optimizer=None,
                 reporter=None):
        if reporter is None:
            reporter = Reporter(model)
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters())
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.reporter = reporter
        self.queue = Queue()
        self.running = False
        self.paused = False

    def status(self):
        return "Running: {self.running}\nPaused: {self.paused}"

    def sync(self):
        if self.running == True:
            self.queue.put("sync")
            (self.model, self.dataloader, self.optimizer, self.reporter) = self.queue.get()

    def start(self):
        if self.running == True and self.paused == False:
            # No effect
            return
        if self.running == False:
            self.process = Process(target=trainer_worker, args=(self.queue,))
            self.process.start()
            self.queue.put((self.model, self.dataloader, self.optimizer, self.reporter))
            self.running = True
            self.paused = True
        if self.paused == True:
            self.queue.put("start")
            self.running = True
            self.paused = False

    def pause(self):
        if self.running == True and self.paused == False:
            self.queue.put("pause")
            self.paused = True

    def stop(self, sync_first=True):
        if sync_first:
            self.sync()
        if self.running == True:
            self.queue.put("stop")
            self.running = False

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
        self.sync()
        was_paused = self.running and self.paused
        self.pause()
        L = model.L
        prompt = (self.dataset.random_text_snippet(2*L) + prompt)
        completion = ""
        tail = prompt
        for _ in range(N):
            x = (torch.tensor(([0]*L + [ord(c) for c in tail if ord(c) < 256])[-L:])
                     .reshape((1,L))
                     .to(default_device()))
            P = model.probs(x)
            prob_dist = torch.distributions.Categorical(P)
            c_ord = prob_dist.sample()[0]
            c = chr(c_ord)
            tail = tail[1:] + c
            completion += c
        print(prompt+"\n~AUTOCOMPLETE~\n"+completion)
        if not was_paused:
            self.start()

        return completion
