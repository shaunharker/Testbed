import gc
import torch.multiprocessing
import torch
ctx = torch.multiprocessing.get_context("spawn")
from queue import Empty
from ..util import IgnoreKeyboardInterrupt, Stopwatch, default_device, memory_allocated, memory_free, numel
import time
import json
import numpy as np
from threading import Thread, Event

class AlreadyTrainingError(Exception):
    pass

class StopTrainThread(Exception):
    pass

class JobInterrupt(Exception):
    def __init__(self, job):
        super().__init__()
        self.job = job

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, metrics_outbox, autocomplete_outbox):
        super().__init__(daemon=True)
        # Communication queues
        self.inbox = inbox
        self.outbox = outbox
        self.metrics_outbox = metrics_outbox
        self.autocomplete_outbox = autocomplete_outbox

        # Thread control
        self.thread = None
        self.terminate_event = Event()

        # Major objects
        self.model = None
        self.dataset = None
        self.optimizer = None

        # Tuning hyperparameters and logs
        self.step = 0
        self.minibatches = None
        self.minibatch_size = None
        self.info = {
            "name": "Worker",
            "time": 0.0,
            "energy": 0.0,
            "ingest": 0.0}
        self.metrics = []
        self.logs = []
        self.last_autocomplete = None

    def log(self, *args, **kwargs):
        log_message = {
            "step": self.step,
            "time": self.info["time"]}
        if len(args) > 0:
            log_message["args"] = args
        log_message.update(kwargs)
        self.logs.append(log_message)
        print(json.dumps(log_message, indent=4))

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
                self.model = config["model"]["type"](
                    **config["model"]["kwargs"]).to('cuda')
                self.log(f"model constructed, number of parameters = {numel(self.model)}")
            except Exception as e:
                print(e)
                self.model = config["model"]
                self.log(f"model received")
            try:
                self.optimizer = config["optimizer"]["type"](
                    parameters=self.model.parameters(),
                    **config["optimizer"]["kwargs"])
                self.log(f"optimizer constructed")
            except Exception as e:
                print(e)
                self.optimizer = config["optimizer"]
                self.log(f"optimizer received")
            try:
                self.dataset = config["dataset"]["type"](
                    **config["dataset"]["kwargs"])
                self.log(f"dataset constructed")
            except Exception as e:
                print(e)
                self.dataset = config["dataset"]
                self.log(f"dataset received")
            self.minibatches = 1
            self.minibatch_size = self.dataset.batch_size
        elif "path" in job["kwargs"]:
            self.log(f"boot from path")
            path = job["kwargs"]["path"]
            self.load(path)
        else:
            raise RuntimeError("Invalid job")
        self.outbox.put("booted")
        main_event_loop()

    def main_event_loop(self):
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
                    if job["instruction"] == "terminate":
                        break
                except Empty:
                    if not self.parent.is_alive():
                        self.log("orphaned")
                        break
        self.log("terminated")

    def dispatch(self, instruction, *args, **kwargs):
        self.log(f"dispatch({instruction, args, kwargs}")
        with Stopwatch() as stopwatch:
            try:
                if instruction == "train":
                    return self.train(*args, **kwargs)
                elif instruction == "terminate":
                    return self.terminate(*args, **kwargs)
                elif instruction == "autocomplete":
                    return self.autocomplete(*args, **kwargs)
                elif instruction == "save":
                    return self.save(*args, **kwargs)
                elif instruction == "load":
                    return self.load(*args, **kwargs)
                elif instruction == "pause":
                    return self.pause(*args, **kwargs)
            except Exception as e:
                return e
        self.log(f"instruction {instruction} handled in {stopwatch.total_run_time}s")

    def train(self, *args, **kwargs):
        self.log("train")
        if "model" in kwargs:
            self.model.update(*args, **kwargs)
        if "optimizer" in kwargs:
            self.optimizer.update(*args, **kwargs)
        if "dataset" in kwargs:
            self.dataset.update(*args, **kwargs)
        self.breakpoint_step = kwargs["breakpoint_step"]
        if self.thread != None:
            raise AlreadyTrainingError()
        self.thread = threading.Thread(target=self._train)
        self.thread.start()
        return f"> Train from step {self.step} to {self.breakpoint_step}."

    def terminate(self):
        self.stop_train_thread_event.set()
        return "terminated"


    def autocomplete(self,
                     prompt=None,
                     encoder=None,
                     decoder=None,
                     n_generate=128,
                     max_ctx=32):
        Categorical = torch.distributions.Categorical
        decode = self.dataset.decode
        encode = self.dataset.encode
        batch = self.dataset.batch
        example_length = self.dataset.example_length
        if max_ctx is None:
            max_ctx = example_length - 1
        if prompt is None:
            if self.last_autocomplete is None:
                prompt = decode(batch(1, max_ctx).tolist()[0])
            else:
                prompt = self.last_autocomplete
                # sloppy, but meh:
                for _ in range(32):
                    prompt = prompt.replace('\n\n','\n').replace('\r', '').replace('__', '_').replace('""', '"')
        else:
            if prompt == "":
                prompt = " "
            prompt = decode(encode(prompt))
        x = encode(prompt)
        x = x[-max_ctx:]
        assert len(x) <= max_ctx, f"x={x} max_ctx={max_ctx}"
        def sampler(x):
            x = list(x)
            for _ in range(n_generate):
                y = Categorical(self.model.probs(torch.tensor(x, dtype=torch.long,device='cuda').unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]).sample().item()
                x = (x + [y])[-max_ctx:]
                self.autocomplete_outbox.put(y) # hook for streaming
                yield y
        try:
            self.last_autocomplete = decode(list(sampler(x)))
        except:
            # if something is wrong (i.e. too many repeated characters shortened prompt too much for some model to accept), just autocomplete a random snippet
            prompt = decode(batch(1, max_ctx).tolist()[0])
            x = encode(prompt)
            x = x[-max_ctx:]
            self.last_autocomplete = decode(list(sampler(x)))
        return self.last_autocomplete

    def save(self, path):
        self.log(f"save({path})")
        checkpoint = {
            "model": self.model,
            "dataset": self.dataset, # just the filename and settings
            "optimizer": self.optimizer,
            "step": self.step,
            "metrics": self.metrics,
            "logs": self.logs}
        torch.save(checkpoint, path)
        return "saved"

    def load(self, path):
        self.log(f"load({path})")
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.dataset = checkpoint["dataset"]
        self.optimizer = checkpoint["optimizer"]
        self.step = checkpoint["step"]
        self.metrics = checkpoint["metrics"]
        self.logs = checkpoint["logs"]
        return self.metrics

    def pause(self):
        return "paused"

    def _train(self):
        self.log("train")
        next_job = {"instruction": None, "args": None, "kwargs": None}
        with Stopwatch() as stopwatch:
            while self.breakpoint_step is None or self.step < self.breakpoint_step:
                if not self.parent.is_alive():
                    raise RuntimeError("Parent process is not alive.")
                try:
                    batch_losses = self.optimizer.step(self.closure)
                except StopTrainThread:
                    self.log("StopTrainThread")
                    self.info["time"] += stopwatch.time_elapsed
                    return
                self.step += 1
                self.info["energy"] = 0 #self.model.profile()["energy"]
                # TODO: don't neglect optimizer costs in energy estimates
                self.info["ingest"] += self.dataset.batch_size * self.dataset.example_length
                step_info = {'step': self.step,
                             'time': self.info["time"] + stopwatch.time_elapsed,
                             'energy': self.info["energy"],
                             'ingest': self.info["ingest"],
                             'mean_loss': np.mean(batch_losses),
                             'batch_losses': batch_losses,
                             'last_autocomplete': self.last_autocomplete}
                self.metrics.append(step_info)
                self.metrics_outbox.put(step_info)
        self.info["time"] += stopwatch.total_run_time
        self.log(f"train reached breakpoint_step {self.breakpoint_step}")

    def closure(self):
        self.minibatches = 1
        self.minibatch_size = self.dataset.batch_size
        while True:
            try:
                return self._closure()
            except RuntimeError as e: # CUDA OOM
                if "CUDA" in str(e): # false positives?
                    torch.cuda.empty_cache()
                    self.minibatches *= 2
                    self.minibatch_size = self.dataset.batch_size // self.minibatches
                    if self.minibatch_size == 0:
                        raise RuntimeError("Cannot compute gradient even with minibatch_size=1.")
                    f = memory_free()
                    a = memory_allocated()
                    self.log(f"Splitting batch of {self.dataset.batch_size} examples into "
                             f"{self.minibatches} minibatches of size {self.minibatch_size} "
                             f"due to memory constraints.\n",
                             batch_size=self.dataset.batch_size,
                             example_length=self.dataset.example_length,
                             cuda_memory={"free": f"{f//2**20}MiB",
                                          "allocated": f"{a//2**20}MiB",
                                          "total": f"{(f+a)//2**20}MiB"})
                else:
                    raise e

    def _closure(self):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        Y = []
        for _ in range(self.minibatches):
            if self.stop_train_thread_event.is_set():
                raise StopTrainThread()
            X = self.dataset.batch(
                    batch_size=self.minibatch_size,
                    example_length=self.dataset.example_length)
            batch_losses = self.model(X)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                loss.backward()
            Y.append(batch_losses.detach())
        Y = torch.cat(Y)
        return Y.cpu().numpy()
