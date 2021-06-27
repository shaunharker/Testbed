import torch
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from ..util import decode_broken_utf8, default_device
from .worker import Worker
from ..data import TextDataset

class Trainer:
    def __init__(self,
                 model=None,
                 N=64, # text example length
                 B=32, # initial batch size
                 optimizer=None,
                 dataset=None):
        self.model = model.to(device='cuda')
        self.N = N
        self.B = B
        self.optimizer = optimizer
        if dataset is None:
            self.dataset = TextDataset(N=N)
        else:
            self.dataset = dataset
        self.datafile = dataset.filename
        self.inbox = Queue()
        self.outbox = Queue()
        self.loss_inbox = Queue()
        self.running = False
        self.paused = None
        self.step = 0
        self.compute_time = 0.0
        self.losses = []

    def set_batch_size(self, batch_size):
        if self.running == True:
            self.outbox.put("set_batch_size")
            self.outbox.put(batch_size)
            if self.paused == False:
                self.outbox.put("start")

    def set_example_length(self, N):
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
            self.outbox.put(self.N) # example length
            self.outbox.put(self.B) # batch size
            self.outbox.put(self.optimizer)
            self.outbox.put(self.dataset)
            self.outbox.put(self.compute_time)
            self.outbox.put(self.step)
            self.outbox.put(self.losses)
            self.running = True
        self.outbox.put("start")
        self.running = True
        self.paused = False

    def update_losses(self):
        while True:
            try:
                item = self.loss_inbox.get(False)
                self.step = item[0]
                self.compute_time = item[1]
                # print(f'recv {self.step}')
            except:
                break
            self.losses.append(item)
            # print(f"{self.step}. {self.losses}")
        return self.losses

    def pause(self):
        """
        Note: blocks until the remote is paused
        """
        if self.running == True:
            self.outbox.put("pause")
            self.paused = True
            self.inbox.get()
        self.loss()

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
        self.losses = checkpoint["loss"]
        self.model = checkpoint["model"].to(device='cuda')
        self.optimizer = checkpoint["optimizer"]
        self.N = checkpoint["sequence_length"]
        self.B = checkpoint["batch_size"]
        datafile = checkpoint["datafile"]
        if datafile == self.datafile:
            self.dataset.set_example_length(self.N)
        else:
            self.datafile = datafile
            self.dataset = TextDataset(filename=self.datafile, N=self.N)

    def save(self, path=None):
        was_paused = self.paused
        self.pause()
        if path is None:
            path = "checkpoint.pt"
        self.outbox.put("save")
        self.outbox.put(path)
        self.inbox.get()
        if self.running and not was_paused:
            self.start()

    def autocomplete(self, prompt="", N=1024):
        was_paused = self.paused
        self.pause()
        L = self.N - 1
        self.model.eval()
        prompt = [b for b in bytes(self.dataset.random_text_snippet() + prompt, 'utf-8')][-L:]
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
        if self.running and not was_paused:
            self.start()
        return decode_broken_utf8(bytes(completion))
