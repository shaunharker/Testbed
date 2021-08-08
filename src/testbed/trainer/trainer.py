import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from .worker import Worker
import threading
from threading import Lock

class Trainer:
    def __init__(self, path=None, config=None):
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
            if type(config["dataset"]) == dict:
                assert "batch_size" in config["dataset"]["kwargs"]
                assert "example_length" in config["dataset"]["kwargs"]
                self.decode = config["dataset"]["type"].decode
            else:
                assert hasattr(config["dataset"], "batch_size")
                assert hasattr(config["dataset"], "example_length")
                self.decode = config["dataset"].decode
            bootargs.update({"config": config})
        self.inbox = Queue()
        self.outbox = Queue()
        self.autocomplete_inbox = Queue()
        self.metrics_lock = Lock()
        self.metrics = []
        self.metrics_inbox = Queue()
        self.metrics_inbox_daemon_halt = threading.Event()
        def _metrics_inbox_daemon(metrics_inbox_daemon_halt, metrics_inbox, metrics):
            while not metrics_inbox_daemon_halt.is_set():
                try:
                    item = metrics_inbox.get(block=True,timeout=1.0)
                    with self.metrics_lock:
                        metrics.append(item)
                except:
                    pass
        self.metrics_inbox_daemon = threading.Thread(
            target=_metrics_inbox_daemon,
            args=(self.metrics_inbox_daemon_halt, self.metrics_inbox, self.metrics),
            daemon=True)
        self.metrics_inbox_daemon.start()
        self.process = Worker(self.outbox, self.inbox, self.metrics_inbox, self.autocomplete_inbox)
        self.process.start()
        self.inbox.get() # blocks until process is ready
        self.paused = True
        self.call("boot", **bootargs)

    def call(self, instruction, *args, **kwargs):
        self.outbox.put({"instruction": instruction,
                         "args": args,
                         "kwargs": kwargs})
        return self.inbox.get()

    def __del__(self):
        self.metrics_inbox_daemon_halt.set()
        self.call("terminate")

    def update(self, *args, **kwargs):
        return self.call("update", *args, **kwargs)

    def start(self, breakpoint_step=None):
        return self.call("start", breakpoint_step=breakpoint_step)

    def pause(self):
        return self.call("pause")

    def terminate(self):
        calling_thread = threading.Thread(
            target=self.call,
            args=("terminate"),
            daemon=True)
        calling_thread.start()
        return None

    def save(self, path="checkpoint.pt"):
        return self.call("save", path=path)

    def load(self, path="checkpoint.pt"):
        with self.metrics_lock:
            self.metrics = self.call("load", path=path)

    def autocomplete(self, prompt=None, n_generate=256, max_ctx=512):
        def sequence():
            for idx in range(n_generate):
                c = None
                while c is None and self.process.is_alive():
                    try:
                        c = self.autocomplete_inbox.get(block=True, timeout=1.0)
                        yield c
                    except:
                        pass
        thread = threading.Thread(
            target=self.call,
            args=("autocomplete",),
            kwargs=dict(prompt=prompt,
                        n_generate=n_generate,
                        max_ctx=max_ctx))
        thread.start()
        return self.decode(sequence())
