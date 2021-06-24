import torch
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from ..util import decode_broken_utf8, default_device
from .worker import Worker
from ..data.textdataset import TextDataset

class Trainer:
    def __init__(self,
                 model=None,
                 dataset=None,
                 optimizer=None):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.inbox = Queue()
        self.outbox = Queue()
        self.loss_inbox = Queue()
        self.running = False
        self.paused = None
        self.step = 0
        self.compute_time = 0.0
        self.data = []

    def set_batch_size(self, batch_size):
        self.dataset.set_batch_size(batch_size)
        if self.running == True:
            self.outbox.put("set_batch_size")
            self.outbox.put(batch_size)
            if self.paused == False:
                self.outbox.put("start")

    def set_example_length(self, N):
        self.dataset.set_example_length(N)
        if self.running == True:
            self.outbox.put("set_example_length")
            self.outbox.put(N)
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
            self.outbox.put(self.optimizer)
            self.outbox.put(self.compute_time)
            self.outbox.put(self.step)
            self.running = True
        self.outbox.put("start")
        self.running = True
        self.paused = False

    def loss(self):
        start = len(self.data)
        while True:
            try:
                item = self.loss_inbox.get(False)
                self.step = item[0]
                self.compute_time = item[1]
                # print(f'recv {self.step}')
            except:
                break
            self.data.append(item)
            # print(f"{self.step}. {self.data}")
        return self.data[start:]

    def pause(self):
        """
        Note: blocks until the remote is paused
        """
        self.loss()
        if self.running == True:
            self.outbox.put("pause")
            self.paused = True
            self.loss()
            self.inbox.get()

    def stop(self):
        self.pause() # to get self.step and self.compute_time
        if self.running == True:
            self.outbox.put("stop")
            self.running = False
            self.paused = None
            self.process.join()

    def load(self, path=None):
        self.stop()
        if path is None:
            path = "checkpoint.pt"
        checkpoint = torch.load(path)
        self.compute_time = checkpoint["time"]
        self.step = checkpoint["step"]
        self.data = checkpoint["loss"]
        self.model = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]
        self.dataset = TextDataset(**checkpoint["dataset"])

    def save(self, path=None):
        was_paused = self.paused
        self.pause()
        if path is None:
            path = "checkpoint.pt"
        checkpoint = {
            'time': self.compute_time,
            'step': self.step,
            'loss': self.data,
            'model': self.model,
            'optimizer': self.optimizer,
            'dataset': self.dataset.state_dict()} # TODO: introduce schedulers
        torch.save(checkpoint, 'checkpoint.pt')
        if self.running and not was_paused:
            self.start()

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
