from datetime import datetime
import numpy as np
import torch
import asyncio
import time

class Trainer:
    def __init__(self, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.reset()
        self.plies = 4

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
        L = S+V+A
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        message = (
            f"{now[:-2]}\nstep {n}\nplies {self.plies}\ntime {int(t)} s\n"
            f"seq loss {8*S} bpc\n"
            f"visual loss {V}\naction loss {A}\n"
            f"total loss {L}\ntraining on {int(n/t*10)/10} games per second")
        return message

    def closure(self):
        (game, seq_input, seq_target, visual_target,
         action_target, seq_loss, visual_loss,
         action_loss) = self.model(self.dataset.bookgame(max_plies=self.plies))
        seq_loss_mean = torch.mean(seq_loss)
        visual_loss_mean = torch.mean(visual_loss)
        action_loss_mean = torch.mean(action_loss)
        loss = seq_loss_mean + visual_loss_mean + action_loss_mean
        loss.backward()
        self.losses.append(loss.item())
        self.seq_losses.append(seq_loss_mean.item())
        self.visual_losses.append(visual_loss_mean.item())
        self.action_losses.append(action_loss_mean.item())
        self.times.append(time.time() - self.t0)
        self.n += 1
        return (game, seq_loss_mean.item(), visual_loss_mean.item(),
            action_loss_mean.item())

    def step(self):
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
        self.n = 0
        self.t0 = time.time()
