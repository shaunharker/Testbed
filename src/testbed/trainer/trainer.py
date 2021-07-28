import torch
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from ..util import decode_broken_utf8, default_device, construct_if_required
from .worker import Worker
from ..data import ByteDataset

class Trainer:
    """
    Trainer
    """
    def __init__(self,
                 path_or_config):
        self.running = False
        self.paused = None
        self.metrics = []
        self.step = 0
        self.info = {
            "name": "Trainer",
            "time": 0.0,
            "energy": 0.0}

        self.inbox = Queue()
        self.outbox = Queue()
        self.metrics_inbox = Queue()
        self.metrics_inbox_daemon = threading.Thread(
            target=self._metrics_inbox_daemon,
            daemon=True)
        self.path_or_config = path_or_config


    def _metrics_inbox_daemon(self):
        while True:
            try:
                item = self.metrics_inbox.get(False)
                # these three things need to be updated to checkpoint properly
                self.step = item['step']
                self.compute_time = item['compute_time']
                self.compute_energy = item['compute_energy']
            except:
                break
            self.metrics.append(item)

    def status(self):
        return {"running": self.running, "paused": self.paused}

    def start(self):
        if self.running == False:
            self.process = Worker(self.outbox, self.inbox, self.metrics_inbox)
            self.process.start()
            self.inbox.get() # blocks until process is ready
            self.outbox.put(self.path_or_config)
            self.running = True
        self.outbox.put("start")
        self.running = True
        self.paused = False

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
        self.outbox.put("load")
        self.outbox.put(path)
        self.inbox.get()
        if self.running and not was_paused:
            self.start()

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

    def update(self, entity, settings):
        if self.running == True:
            self.outbox.put("update")
            self.outbox.put(entity)
            self.outbox.put(settings)
            self.inbox.get() # blocks

    def get_optimizer_stats(self):
        stats = None
        if self.running == True:
            self.outbox.put("get_optimizer_stats")
            stats = self.inbox.get()
            if self.paused == False:
                self.outbox.put("start")
        return stats

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
