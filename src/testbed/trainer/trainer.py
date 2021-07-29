import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from .worker import Worker
import threading


class Trainer:
    """
    Trainer

    Args:

    Supply exactly one of the following arguments:

        path: the path of a checkpoint

    or

        config: a configuration dictionary of the form:

    Example:

        config = {
            "model": {
                "type": ModelType,
                "kwargs": model_kwargs},
            "dataset": {
                "type": DatasetType,
                "kwargs": dataset_kwargs},
            "optimizer": {
                "type": OptimizerType,
                "kwargs": optimizer_kwargs},
            "batch_size": batch_size,
            "example_length": example_length}

        trainer = Trainer(config)

    """
    def __init__(self, path=None, config=None):
        # Check arguments and compute arguments for
        # worker.
        if path is None and config is None:
            raise ValueError("Trainer requires either a path or "
                             "a config argument.")
        if path is not None and config is not None:
            raise ValueError("Trainer accepts either a path or "
                             "a config argument, but not both.")
        bootargs = {}
        if path is not None:
            bootargs.update({"path": path})
        if config is not None:
            assert "model" in config
            assert "optimizer" in config
            assert "dataset" in config
            assert "batch_size" in config
            assert "example_length" in config
            bootargs.update({"config": config})

        # main inbox/output communication channels:
        self.inbox = Queue()
        self.outbox = Queue()

        # self.metrics
        #     is a list that stores data for the client to use.
        #
        # self.metrics_inbox
        #     is a communication queue used to
        #     receive the continually streaming data that is
        #     placed in the metrics list.
        #
        # self.metrics_inbox_daemon
        #     is a thread that moves the contents
        #     of self.metrics_inbox into self.metric.
        #
        # self.halt
        #     is an event that signals to self.metrics_inbox_daemon
        #     to break from its event loop and shut down.

        self.metrics = []
        self.metrics_inbox = Queue()
        self.halt = threading.Event()
        def _metrics_inbox_daemon(halt, metrics_inbox, metrics):
            while not halt.is_set():
                try:
                    item = metrics_inbox.get(block=True,timeout=1.0)
                    metrics.append(item)
                except:
                    pass
        self.metrics_inbox_daemon = threading.Thread(
            target=_metrics_inbox_daemon,
            args=(self.halt, self.metrics_inbox, self.metrics),
            daemon=True)
        self.metrics_inbox_daemon.start()

        # Boot up the worker. It will be in a "paused" state.
        self.process = Worker(self.outbox, self.inbox, self.metrics_inbox)
        self.process.start()
        self.inbox.get() # blocks until process is ready
        self.paused = True

        self.call("boot", **bootargs)

    def __del__(self):
        self.halt.set()

    def start(self):
        if self.paused == True:
            result = self.call("start")
            self.paused = False
            return result
        return "already started"

    def pause(self):
        if self.paused == False:
            self.paused = True
            return self.call("pause")
        return "already paused"

    def stop(self):
        self.paused = None
        result = self.call("stop")
        self.process.join()
        self.halt_metrics_inbox_daemon.set()
        return result

    def save(self, path="checkpoint.pt"):
        return self.call("save", path=path)

    def update(self, entity, settings):
        return self.call("update", entity=entity, setting=settings)

    def fetch(self, entity):
        return self.call("fetch", entity=entity)

    def autocomplete(self, prompt="", output_length=256, max_ctx=512):
        return self.call("autocomplete", prompt=prompt,
                         output_length=output_length, max_ctx=max_ctx)

    def status(self):
        return {"paused": self.paused}

    def call(self, instruction, **kwargs):
        if self.paused is None:
            raise RuntimeError("Trainer is stopped.")
        self.outbox.put({"instruction": instruction,
                         "kwargs": kwargs})
        result = self.inbox.get() # blocks

        # If the trainer was not paused to begin, then resume:
        if self.paused == False:
            self.paused = True # prevents infinite recursion
            self.call("start")
            self.paused = False
        return result
