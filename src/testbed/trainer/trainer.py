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
                 example_length=64, # text example length
                 batch_size=32, # initial batch size
                 DatasetType=None,
                 dataset_kwargs={},
                 OptimizerType=None,
                 optimizer_args=[],
                 optimizer_kwargs={}):

        self.example_length = example_length
        self.batch_size = batch_size
        if DatasetType is None:
            DatasetType = TextDataset
        self.DatasetType = DatasetType
        self.dataset_kwargs = dataset_kwargs
        if OptimizerType is None:
            OptimizerType = torch.optim.AdamW
        self.OptimizerType = OptimizerType
        self.optimizer_args = optimizer_args
        self.optimizer_kwargs = optimizer_kwargs
        self.inbox = Queue()
        self.outbox = Queue()
        self.loss_inbox = Queue()
        self.running = False
        self.paused = None
        self.step = 0
        self.compute_time = 0.0
        self.compute_energy = 0.0
        self.losses = []

        self.dataset = self.DatasetType(**self.dataset_kwargs)
        self.dataset.set_example_length(self.example_length)
        
        if model is not None:
            if type(model) is str:
                self.load(model).to(device='cuda')
            else:
                self.model = model.to(device='cuda')

    def set_batch_size(self, batch_size):
        if self.running == True:
            self.outbox.put("set_batch_size")
            self.outbox.put(batch_size)
            if self.paused == False:
                self.outbox.put("start")
        self.batch_size = batch_size

    def set_example_length(self, example_length):
        if self.running == True:
            self.outbox.put("set_example_length")
            self.outbox.put(example_length)
            if self.paused == False:
                self.outbox.put("start")
        self.example_length = example_length
        self.dataset.set_example_length(example_length)

    def set_optimizer_settings(self, **settings):
        if self.running == True:
            self.outbox.put("set_optimizer_settings")
            self.outbox.put(settings)
            if self.paused == False:
                self.outbox.put("start")

    def get_optimizer_stats(self):
        stats = None
        if self.running == True:
            self.outbox.put("get_optimizer_stats")
            stats = self.inbox.get()
            if self.paused == False:
                self.outbox.put("start")
        return stats

    def status(self):
        return f"Running: {self.running}\nPaused: {self.paused}"

    def start(self):
        if self.running == False:
            self.process = Worker(self.outbox, self.inbox, self.loss_inbox)
            self.process.start()
            ready = self.inbox.get() # Wait for ready.
            # the problem with the following is that one needs to memorize the order
            # uselessly. Instead, use a dictionary. TODO
            self.outbox.put(self.model)
            self.outbox.put(self.example_length)
            self.outbox.put(self.batch_size)
            self.outbox.put(self.OptimizerType)
            self.outbox.put(self.optimizer_args)
            self.outbox.put(self.optimizer_kwargs)
            self.outbox.put(self.DatasetType)
            self.outbox.put(self.dataset_kwargs)
            self.outbox.put(self.compute_time)
            self.outbox.put(self.compute_energy)
            self.outbox.put(self.step)
            self.outbox.put(self.losses)
            self.running = True
        self.outbox.put("start")
        self.running = True
        self.paused = False

    def update(self):
        while True:
            try:
                item = self.loss_inbox.get(False)
                # these three things need to be updated to checkpoint properly
                self.step = item['step']
                self.compute_time = item['compute_time']
                self.compute_energy = item['compute_energy']
            except:
                break
            self.losses.append(item)

    def pause(self):
        """
        Note: blocks until the remote is paused
        """
        if self.running == True:
            self.outbox.put("pause")
            self.paused = True
            self.inbox.get()
        self.update()

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
        self.compute_time = checkpoint["compute_time"]
        try:
            self.compute_energy = checkpoint["compute_energy"]
        except:
            self.compute_energy = 0.0
        self.step = checkpoint["step"]
        self.losses = checkpoint["losses"]
        self.model = checkpoint["model"].to(device='cuda')
        self.example_length = checkpoint["example_length"]
        self.batch_size = checkpoint["batch_size"]
        self.OptimizerType = checkpoint["OptimizerType"]
        self.optimizer_args = checkpoint["optimizer_args"]
        self.optimizer_kwargs = checkpoint["optimizer_kwargs"]
        filename = checkpoint["dataset.filename"]
        if filename == self.dataset.filename:
            self.dataset.set_example_length(self.example_length)
        else:
            self.dataset = TextDataset(filename=filename, example_length=self.example_length)

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

    def autocomplete(self, prompt="", output_length=1024):
        was_paused = self.paused
        self.pause()
        L = self.example_length - 1  # CODE SMELL (refactor net0 to not compute own loss, refactor closure to reflect new case, etc.)
        #print(L, self.example_length)
        self.model.eval()
        prompt = [b for b in bytes(self.dataset.random_text_snippet() + prompt, 'utf-8')][-L:]
        completion = []
        tail = prompt
        for _ in range(output_length):
            x = (torch.tensor(tail)
                      .unsqueeze(0)
                      .to(default_device())) # shape [1,L]
            #print(x.shape)
            P = self.model.probs(x).view(-1)
            prob_dist = torch.distributions.Categorical(P)
            c_ord = prob_dist.sample().item()
            tail = tail[1:] + [c_ord]
            completion += [c_ord]
        print(decode_broken_utf8(bytes(prompt)+bytes("\n~AUTOCOMPLETE~\n",'utf-8') + bytes(completion)))
        if self.running and not was_paused:
            self.start()
        return decode_broken_utf8(bytes(completion))
