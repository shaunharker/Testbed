import gc
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
from queue import Empty
from ..util import IgnoreKeyboardInterrupt, Stopwatch, default_device, memory_allocated
import time
import json
import numpy as np

class JobInterrupt(Exception):
    def __init__(self, job):
        super().__init__()
        self.job = job

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, metrics_outbox):
        super().__init__(daemon=True)
        self.inbox = inbox
        self.outbox = outbox
        self.metrics_outbox = metrics_outbox

        self.model = None
        self.dataset = None
        self.optimizer = None
        self.batch_size = None
        self.example_length = None

        self.minibatches = None
        self.minibatch_size = None
        self.step = 0
        self.info = {
            "name": "Worker",
            "time": 0.0,
            "energy": 0.0,
            "ingest": 0.0,
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
            except RuntimeError as e: # CUDA OOM
                if "CUDA out of memory" in str(e):
                    self.minibatches *= 2
                    self.minibatch_size = self.batch_size // self.minibatches
                    if self.minibatch_size == 0:
                        raise RuntimeError("Cannot compute gradient even with minibatch_size=1.")
                    self.log(f"Splitting batch of {self.batch_size} examples into "
                             f"{self.minibatches} minibatches of size {self.minibatch_size} "
                             "due to memory constraints.\n"
                             "The results will not be affected.")
                else:
                    raise e

    def _closure(self):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        Y = []
        for _ in range(self.minibatches):
            try:
                job = self.inbox.get(False)
                if job["instruction"] != "start":
                    raise JobInterrupt(job)
                else:
                    self.outbox.put("already started")
            except Empty:
                pass
            X = (self.dataset.batch(
                    batch_size=self.minibatch_size,
                    example_length=self.example_length)
                .to(device='cuda',
                    dtype=torch.long,
                    non_blocking=True))
            batch_losses = self.model(X)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                loss.backward()
            Y.append(batch_losses.detach())
        Y = torch.cat(Y)
        return Y.cpu().numpy()

    def train(self):
        self.log("train")
        with Stopwatch() as stopwatch:
            while True:
                if not self.parent.is_alive():
                    raise RuntimeError("Parent process is not alive.")
                try:
                    batch_losses = self.optimizer.step(self.closure)
                except JobInterrupt as e:
                    job = e.job
                    break
                self.step += 1
                self.info["energy"] = self.model.profile()["energy"]
                # TODO: don't neglect optimizer costs
                self.info["ingest"] += self.batch_size * self.example_length
                step_info = {'step': self.step,
                             'time': self.info["time"] + stopwatch.time_elapsed,
                             'energy': self.info["energy"]/1E12,
                             'ingest': self.info["ingest"],
                             'profile': self.info,
                             'mean_loss': np.mean(batch_losses),
                             'batch_losses': batch_losses}
                self.metrics.append(step_info)
                self.metrics_outbox.put(step_info)
        self.info["time"] += stopwatch.total_run_time
        self.log(f"JobInterrupt({job})")
        return job

    def run(self):
        self.log(f"Worker.run")
        self.parent = torch.multiprocessing.parent_process()
        self.outbox.put("ready")
        job = self.inbox.get()
        instruction = job["instruction"]
        assert instruction == "boot"
        self.log(f"boot")
        if "config" in job["kwargs"]:
            self.log(f"boot from config")
            config = job["kwargs"]["config"]
            try:
                self.model = config["model"]["type"](**config["model"]["kwargs"])
            except:
                self.model = config["model"]
            self.optimizer = config["optimizer"]["type"](
                self.model.parameters(), **config["optimizer"]["kwargs"])
            self.dataset = config["dataset"]["type"](**config["dataset"]["kwargs"])
            self.batch_size = config["batch_size"]
            self.example_length = config["example_length"]
            self.minibatches = 1
            self.minibatch_size = self.batch_size

        elif "path" in job["kwargs"]:
            self.log(f"boot from path")
            path = job["kwargs"]["path"]
            self.load(path)
        else:
            raise RuntimeError("Invalid job")
        self.outbox.put("booted")

        # Jupyter sends KeyboardInterrupt in error sometimes
        # when cells are interrupted, so we ignore them in our
        # main event loop:
        self.log(f"main_event_loop")
        with IgnoreKeyboardInterrupt():
            while True:
                try:
                    job = self.inbox.get(True,1.0)
                    result = self.dispatch(job["instruction"],
                                           *job["args"],
                                           **job["kwargs"])
                    self.outbox.put(result)

                    if job["instruction"] == "stop":
                        break
                except Empty:
                    if not self.parent.is_alive():
                        self.log("orphaned")
                        break
        self.log("terminate")

    def dispatch(self, instruction, *args, **kwargs):
        self.log(f"dispatch({instruction, kwargs}")
        if instruction == "start":
            self.log("start")
            self.outbox.put("started")
            job = self.train()
            instruction = job["instruction"]
            args = job["args"]
            kwargs = job["kwargs"]
            assert instruction != "start"
            # ... and then case fall-through to handle
            #     the job that train was interrupted by
        if instruction == "stop":
            self.log("stop")
            return "stopped"
        if instruction == "update":
            return self.update(*args, **kwargs)
        if instruction == "save":
            return self.save(*args, **kwargs)
        if instruction == "pause":
            self.log("pause")
            return "paused"

    def update(self, *args, **kwargs):
        self.log(f"update({args},{kwargs})")
        entity = args[0]
        args = args[1:]
        if entity == "model":
            return self.model.update(*args, **kwargs)
        if entity == "optimizer":
            return self.optimizer.update(*args, **kwargs)
        if entity == "dataset":
            return self.dataset.update(*args, **kwargs)
        if entity == "batch_size":
            self.batch_size = kwargs["batch_size"]
            return self.batch_size
        if entity == "example_length":
            self.example_length = kwargs["example_length"]
            return self.example_length

    def load(self, path):
        self.log(f"load({path})")
        checkpoint = torch.load(path)
        make = lambda x: x["type"](**x["kwargs"])
        self.model = checkpoint["model"]
        self.dataset = make(checkpoint["dataset"])
        self.optimizer = checkpoint["optimizer"]
        self.batch_size = checkpoint["batch_size"]
        self.example_length = checkpoint["example_length"]
        self.step = checkpoint["step"]
        self.profile = checkpoint["profile"]
        self.metrics = checkpoint["metrics"]
        self.logs = checkpoint["logs"]
        return "loaded"

    def save(self, path):
        self.log(f"save({path})")
        checkpoint = {
            "model": self.model,
            "dataset": {
                "type": type(self.dataset),
                "kwargs": self.dataset.kwargs},
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "example_length": self.example_length,
            "step": self.step,
            "profile": self.profile(),
            "metrics": self.metrics,
            "logs": self.logs}
        torch.save(checkpoint, path)
        return "saved"
