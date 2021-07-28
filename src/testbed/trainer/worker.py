import gc
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
from queue import Empty
from ..util import IgnoreKeyboardInterrupt, Stopwatch, default_device, memory_allocated
import time
import json

class JobInterrrupt(Exception):
    def __init__(self, job):
        super().__init__()
        self.job = job

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, metrics_outbox):
        super().__init__(daemon=True)
        self.inbox = inbox
        self.outbox = outbox
        self.metrics_outbox = metrics_outbox
        self.parent = torch.multiprocessing.parent_process()

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
                    raise JobInterrrupt(job)
                else:
                    self.outbox.put("already started")
            except Empty:
                pass
            X = (self.dataset.batch(
                    example_length=self.example_length,
                    batch_size=self.minibatch_size)
                .to(device='cuda',
                    dtype=torch.long,
                    non_blocking=True))
            batch_losses = self.model(X)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                loss.backward()
            Y.append(batch_losses.detach())
        Y = torch.cat(Y)
        return Y.numpy()

    def train(self):
        self.log("train")
        with Stopwatch() as stopwatch:
            while True:
                if not self.parent.is_alive():
                    raise RuntimeError("Parent process is not alive.")
                try:
                    batch_losses = self.optimizer.step(self.closure)
                except JobInterrrupt as e:
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
                             'batch_losses': batch_losses}
                self.metrics.append(step_info)
                self.metrics_outbox.put(step_info)
        self.info["time"] += stopwatch.total_run_time
        self.log(f"JobInterrrupt({job})")
        return job

    def run(self):
        self.log(f"Worker.run")
        self.minibatches = 1
        self.outbox.put("ready")
        job = self.inbox.get()
        instruction = job["instruction"]
        assert instruction == "boot"
        if "config" in job:
            self.config = job["config"]
            make = lambda x: x["type"](**x["kwargs"])
            self.model = make(self.config["model"])
            self.optimizer = make(self.config["optimizer"])
            self.dataset = make(self.config["dataset"])
            self.batch_size = self.config["batch_size"]
            self.example_length = self.config["example_length"]
        elif "path" in job:
            self.load(job["path"])
        else:
            raise RuntimeError("Invalid job")

        # Jupyter sends KeyboardInterrupt in error sometimes
        # when cells are interrupted, so we ignore them in our
        # main event loop:
        with IgnoreKeyboardInterrupt():
            while True:
                try:
                    job = self.inbox.get(True,1.0)
                    result = self.dispatch(job["instruction"],
                                           job["kwargs"])
                    self.outbox.put(result)

                    if job["instruction"] == "stop":
                        break
                except Empty:
                    if not self.parent.is_alive():
                        self.log("orphaned")
                        break
        self.log("terminate")

    def dispatch(self, instruction, kwargs):
        self.log(f"dispatch({instruction, kwargs}")
        if instruction == "start":
            self.log("start")
            self.outbox.put("started")
            job = self.train()
            instruction = job["instruction"]
            kwargs = job["kwargs"]
            assert instruction != "start"
            # ... and then case fall-through to handle
            #     the job that train was interrupted by
        if instruction == "stop":
            self.log("stop")
            return "stopped"
        if instruction == "update":
            entity = kwargs["entity"]
            settings = kwargs["settings"]
            return self.update(entity, settings)
        if instruction == "fetch":
            entity = kwargs["entity"]
            return self.fetch(entity)
        if instruction == "save":
            path = kwargs["path"]
            self.save(path)
            return "saved"
        if instruction == "pause":
            self.log("pause")
            return "paused"

    def update(self, entity, settings):
        self.log(f"update({entity},{settings})")
        if entity == "model":
            self.model.update(settings)
            return self.model.settings
        if entity == "optimizer":
            self.optimizer.update(settings)
            return self.optimizer.settings
        if entity == "dataset":
            self.dataset.update(settings)
            return self.dataset.settings
        if entity == "batch_size":
            self.batch_size = settings
            return self.batch_size
        if entity == "example_length":
            self.example_length = settings
            return self.example_length

    def fetch(self, entity):
        self.log(f"fetch({entity})")
        if entity == "model":
            return 0 # TODO
        if entity == "optimizer":
            return 0 # TODO
        if entity == "dataset":
            return 0 # TODO

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
