from datetime import datetime
import numpy as np
import torch
import asyncio
import time
from IPython.display import HTML

class Trainer:
    def __init__(self, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.reset()
        self.plies = 100
        self.seq_coef = 0.0
        self.visual_coef = 0.0
        self.action_coef = 0.0
        self.lr = lambda n: 1e-5*(n/100) if n < 100 else 1e-5
        self.beta1 = lambda n: 0.9
        self.beta2 = lambda n: 0.999
        self.weight_decay = lambda n: 0.001
        self.batch_multiplier = 1
        self.warm_up = 0
        self.update = (lambda n: (n < self.warm_up)
            or (n%self.batch_multiplier == 0))
        self.games = []


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
    "plies"        : {self.plies},
    "training_time": {int(t)},
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
            "plies"        : self.plies,
            "training_time": int(t),
            "seq_loss"     : S,
            "visual_loss"  : V,
            "action_loss"  : A,
            "total_loss"   : L,
            "games per sec": int(n/t*10)/10})
        return message

    def closure(self):
        while True:
            try:
                game = self.dataset.bookgame(
                   max_plies=self.plies)
                (game, seq_input, seq_target, visual_target,
                 action_target, seq_loss, visual_loss,
                 action_loss) = self.model(game)
                break
            except:
                pass
        seq_loss_mean = torch.mean(seq_loss)
        visual_loss_mean = torch.mean(visual_loss)
        action_loss_mean = torch.mean(action_loss)
        loss = (self.seq_coef * seq_loss_mean +
            self.visual_coef * visual_loss_mean +
            self.action_coef * action_loss_mean)
        loss.backward()
        self.games.append(game)
        self.losses.append(loss.item())
        self.seq_losses.append(seq_loss_mean.item())
        self.visual_losses.append(visual_loss_mean.item())
        self.action_losses.append(action_loss_mean.item())
        self.times.append(time.time() - self.t0)
        self.n += 1
        return (game, seq_loss_mean.item(),
            visual_loss_mean.item(), action_loss_mean.item())

    def step(self):
        self.update = (lambda n: (n < self.warm_up)
            or (n%self.batch_multiplier == 0))
        for (pn, p) in self.model.named_parameters():
            state = self.optimizer.state[pn]
            state["lr"] = self.lr
            state["beta1"] = self.beta1
            state["beta2"] = self.beta2
            state["weight_decay"] = self.weight_decay
            state["update"] = self.update
        return self.optimizer.step(self.closure)

    async def train(self):
        await asyncio.sleep(1)
        while True:
            try:
                self.step()
            except:
                pass
            await asyncio.sleep(.1)

    def reset(self):
        self.times = []
        self.losses = []
        self.seq_losses = []
        self.visual_losses = []
        self.action_losses = []
        self.monitor = []
        self.n = 0
        self.t0 = time.time()
