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
                "kwargs": optimizer_kwargs}}
        assert "max_ctx" in config["model"]["kwargs"]
        assert "batch_size" in config["dataset"]["kwargs"]
        assert "example_length" in config["dataset"]["kwargs"]

        trainer = Trainer(config=config)

    """
    def __init__(self, path=None, config=None):
        # Validate arguments
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
            if type(config["model"]) == dict:
                assert "max_ctx" in config["model"]["kwargs"]
            else:
                assert hasattr(config["model"], "max_ctx")
            if type(config["dataset"]) == dict:
                assert "batch_size" in config["dataset"]["kwargs"]
                assert "example_length" in config["dataset"]["kwargs"]
            else:
                assert hasattr(config["dataset"], "batch_size")
                assert hasattr(config["dataset"], "example_length")

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

        self.autocomplete_inbox = Queue()

        # Boot up the worker. It will be in a "paused" state.
        self.process = Worker(self.outbox, self.inbox, self.metrics_inbox, self.autocomplete_inbox)
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

    def update(self, *args, **kwargs):
        return self.call("update", *args, **kwargs)

    def autocomplete(self, prompt="", n_generate=256, max_ctx=512):
        def sequence():
            for _ in range(n_generate):
                while self.process.is_alive():
                    try:
                        yield self.autocomplete_inbox.get(block=True, timeout=1.0)
                    except:
                        pass
        thread = threading.Thread(
            target=self.call,
            args=("autocomplete",),
            kwargs=dict(prompt=prompt,
                        n_generate=n_generate,
                        max_ctx=max_ctx))
        thread.start()
        return self.dataset.decoder(sequence())


    def status(self):
        return {"paused": self.paused}

    def call(self, instruction, *args, **kwargs):
        if self.paused is None:
            raise RuntimeError("Trainer is stopped.")
        self.outbox.put({"instruction": instruction,
                         "args": args,
                         "kwargs": kwargs})
        result = self.inbox.get() # blocks

        # If the trainer was not paused to begin, then resume:
        if self.paused == False:
            self.paused = True # prevents infinite recursion
            self.call("start")
            self.paused = False
        return result
