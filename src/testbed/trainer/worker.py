import numpy as np
import gc
import torch
from torch.optim import AdamW
from ..optim import Sonny
from ..data import Loader, TextDataset
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
from queue import Empty
from ..util import IgnoreKeyboardInterrupt, Stopwatch, default_device, construct_if_required
import time
import json

class InstructionInterrupt(Exception):
    def __init__(self, instruction):
        super().__init__()
        instruction = instruction

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, metrics_outbox):
        super().__init__(daemon=True)
        self.inbox = inbox
        self.outbox = outbox
        self.metrics_outbox = metrics_outbox

        self.model = None
        self.dataset = None
        self.optimizer = None
        self.scheduler = None
        self.step = 0
        self.info = {
            "name": "Worker",
            "time": 0.0,
            "energy": 0.0,
            "data": 0.0,
            "children": {
                "model": {}}}
        self.metrics = []
        self.logs = []

    def profile(self):
        return self.info

    def log(self, message):
        log_message = {
            "step": self.step,
            "time": self.info["time"],
            "message": message}
        self.logs.append(log_message)
        print(json.dumps(log_message))

    def closure(self):
        while True:
            try:
                return self._closure()
            except RuntimeError as err: # CUDA OOM
                gc.collect()
                torch.cuda.empty_cache()
                self.log(f"Closure computation experienced RuntimeError {err}.")
                self.minibatches *= 2
                self.minibatch_size = self.batch_size // self.minibatches
                if self.minibatch_size == 0:
                    raise RuntimeError("Cannot compute gradient even with minibatch_size=1.")
                self.log(f"{self.step}. Splitting batch of {self.batch_size} examples into {self.minibatches} of {self.minibatch_size} examples.")

    def _closure(self):
        num_examples = 0
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        Y = []
        for _ in range(self.minibatches):
            try:
                instruction = self.inbox.get(False)
                if instruction != "start":
                    self.log(f"{self.step}. Interrupted by instruction '{instruction}'.")
                    raise InstructionInterrupt(instruction)
            except Empty:
                pass
            X = self.loader.batch(batch_size=self.minibatch_size).to(device='cuda',dtype=torch.long, non_blocking=True)
            batch_losses = self.model(X)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                loss.backward()
            num_examples += torch.numel(batch_losses)
            Y.append(batch_losses.detach())
            if torch.any(torch.isnan(batch_losses)):
                raise RuntimeError("1nan")
        Y = torch.cat(Y)

        if torch.any(torch.isnan(Y)):
            raise RuntimeError("2nan")
        mean_loss = torch.mean(Y).item()
        mean_sqr_loss = torch.mean(Y*Y).item()
        return (mean_loss, mean_sqr_loss, num_examples)

    def train(self):
        self.log("train")
        with Stopwatch() as stopwatch:
            while True:
                if not self.parent.is_alive():
                    raise RuntimeError("Parent process is not alive.")
                try:
                    loss_statistics = self.optimizer.step(self.closure)
                except InstructionInterrupt as e:
                    instruction = e.instruction
                    break
                self.step += 1

                self.info["model"] = self.model.profile()
                self.info["energy"] = self.info["model"]["energy"]
                self.info["ingest"] += self.scheduler.batch_size * self.scheduler.example_length
                step_info = {'step': self.step,
                             'time': self.info["time"] + stopwatch.time_elapsed,
                             'energy': self.info["energy"]/1E12,
                             'ingest': self.info["ingest"],
                             'profile': self.info,
                             'loss_statistics': loss_statistics}
                self.metrics.append(step_info)
                self.metrics_outbox.put(step_info)
        self.info["time"] += stopwatch.total_run_time
        self.log(f"InstructionInterrupt({instruction})")
        return instruction

    def run(self):
        self.log(f"Worker.run")
        with IgnoreKeyboardInterrupt(): # Jupyter sends these in error sometimes when cells are interrupted.
            self.outbox.put("ready")
            job = self.inbox.get()
            self.model = construct_if_required(job["model"])
            self.optimizer = construct_if_required(job["optimizer"])
            self.dataset = construct_if_required(job["dataset"])
            self.scheduler = construct_if_required(job["scheduler"])
            self.info = job["profile"]
            self.step = job["step"]
            self.metrics = job["metrics"]
            self.parent = torch.multiprocessing.parent_process()
            instruction = "pause"
            while instruction != "stop":
                try:
                    instruction = self.inbox.get(True,1.0)
                    self.dispatch(instruction)
                except Empty:
                    if not self.parent.is_alive():
                        self.log("orphaned")
                        instruction = "stop"
        self.log("terminate")

    def dispatch(self, instruction):
        self.log(f"dispatch({instruction}")
        if instruction == "start":
            instruction = self.train()
            # ... and then case fall-through for whatever it returns.
        if instruction == "update":
            entity = self.inbox.get()
            settings = self.inbox.get()
            self.update(entity, settings)
        if instruction == "get_optimizer_stats":
            self.log("get_optimizer_stats")
            self.outbox.put(self.optimizer.state['stats'])
        if instruction == "save":
            savefile = self.inbox.get()
            self.save(savefile)
            self.outbox.put("saved")
        if instruction == "pause":
            self.outbox.put("paused")

    def update(self, entity, settings):
        self.log(f"{entity}.update({settings})")
        if entity == "model":
            self.model.update(settings)
        if entity == "optimizer":
            self.optimizer.update(settings)
        if entity == "dataset":
            self.dataset.update(settings)
        if entity == "scheduler":
            self.scheduler.update(settings)

    def load(self, path):
        self.log(f"load({path})")
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.dataset = construct_if_required(checkpoint["dataset"])
        self.optimizer = construct_if_required(checkpoint["optimizer"])
        self.scheduler = construct_if_required(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        self.profile = checkpoint["profile"]
        self.metrics = checkpoint["metrics"]
        self.logs = checkpoint["logs"]

    def save(self, path):
        self.log(f"save({path})")
        checkpoint = {
            "model": self.model,
            "dataset": {
                "type": type(self.dataset),
                "kwargs": self.dataset.kwargs},
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "step": self.step,
            "profile": self.profile(),
            "metrics": self.metrics,
            "logs": self.logs}
        torch.save(checkpoint, path)
