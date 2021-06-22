import torch
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from ..util import decode_broken_utf8, default_device
from .worker import Worker

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
            self.process = Worker(self.outbox, self.inbox, self.loss_inbox)
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
