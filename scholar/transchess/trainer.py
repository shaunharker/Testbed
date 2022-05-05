from datetime import datetime
import numpy as np
import torch
import asyncio
import time
from IPython.display import HTML
from asyncio import LifoQueue as Queue
from .targets import maketargets, bytes_to_tensor

class Trainer:
    def __init__(self, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.reset()
        self.seq_length = 16
        self.batch_size = 1
        self.batch_multiplier = 1
        self.seq_coef = 0.0
        self.visual_coef = 0.0
        self.action_coef = 0.0
        self.lr = lambda n: 1e-5*(n/100) if n < 100 else 1e-5
        self.beta1 = lambda n: 0.9
        self.beta2 = lambda n: 0.999
        self.weight_decay = lambda n: 0.001
        self.warm_up = 0
        self.update = (lambda n: (n < self.warm_up)
            or (n%self.batch_multiplier == 0))
        for (pn, p) in self.model.named_parameters():
            state = self.optimizer.state[pn]
            state["lr"] = self.lr
            state["beta1"] = self.beta1
            state["beta2"] = self.beta2
            state["weight_decay"] = self.weight_decay
            state["update"] = self.update
        self.queue = Queue(maxsize=16)

    async def prepare(self):
        while True:
            targets = [maketargets(game=self.dataset.bookgame(),
                seq_length=self.seq_length)
                for _ in range(self.batch_size)]
            # collate
            seq_len = min(t[1].shape[0] for t in targets)
            seq_len = min(seq_len, self.seq_length)
            # enqueue
            job = tuple(torch.stack([t[k][:seq_len] for t in targets], dim=0) for k in range(4))
            await self.queue.put(job)

    async def step(self):
        tgt = await self.queue.get()
        (seq_loss, visual_loss,
          action_loss) = self.model(targets=tgt)
        seq_loss_mean = torch.mean(seq_loss)
        visual_loss_mean = torch.mean(visual_loss)
        action_loss_mean = torch.mean(action_loss)
        loss = (self.seq_coef * seq_loss_mean +
            self.visual_coef * visual_loss_mean +
            self.action_coef * action_loss_mean)
        loss.backward()
        self.losses.append(loss.item())
        self.seq_losses.append(seq_loss_mean.item())
        self.visual_losses.append(visual_loss_mean.item())
        self.action_losses.append(action_loss_mean.item())
        self.times.append(time.time() - self.t0)
        self.n += 1
        self.optimizer.step()
        return (seq_loss_mean.item(),
            visual_loss_mean.item(),
            action_loss_mean.item())

    def reset(self):
        self.times = []
        self.losses = []
        self.seq_losses = []
        self.visual_losses = []
        self.action_losses = []
        self.monitor = []
        self.n = 0
        self.t0 = time.time()

    def status(self, lag=2000):
        losses = self.losses
        n = self.n
        t = time.time() - self.t0
        N = min(n, lag)
        if N == 0:
            return ""
        S = np.mean(np.array(self.seq_losses[n-N:n]))
        V = np.mean(np.array(self.visual_losses[n-N:n]))
        A = np.mean(np.array(self.action_losses[n-N:n]))
        L = np.mean(np.array(self.losses[n-N:n]))
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        message = HTML(f"""<pre>self.monitor[{len(self.monitor)}] = {{
    "current_time" : "{now[:-2]}",
    "step"         : {n},
    "seq_length"   : {self.seq_length},
    "batch_size"   : {self.batch_size},
    "training_time": {int(t)},
    "learning rate": {self.lr(self.n)},
    "seq_loss"     : {S:.6},
    "visual_loss"  : {V:.6},
    "action_loss"  : {A:.6},
    "total_loss"   : {L:.6},
    "games per sec": {int(n/t*10)/10}
}}
</pre>""")
        self.monitor.append({
            "current_time" : now[:-2],
            "step"         : n,
            "seq_length"   : self.seq_length,
            "batch_size"   : self.batch_size,
            "training_time": int(t),
            "learning rate": self.lr(self.n),
            "seq_loss"     : S,
            "visual_loss"  : V,
            "action_loss"  : A,
            "total_loss"   : L,
            "games per sec": int(n/t*10)/10})
        return message
