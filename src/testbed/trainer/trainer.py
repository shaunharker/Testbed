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

      path_or_config: the path of a checkpoint or else
                      a configuration dictionary of the form:

        {"model": {"type": ModelType, "kwargs": model_kwargs},
         "dataset": {"type": DatasetType, "kwargs": dataset_kwargs},
         "optimizer": {"type": OptimizerType, "kwargs": optimizer_kwargs},
         "scheduler": {"type": SchedulerType, "kwargs": scheduler_kwargs}}

    """
    def __init__(self, path_or_config=None):
        # communication
        self.inbox = Queue()
        self.outbox = Queue()

        # self.metrics is a list that stores data for the client to use.
        self.metrics = []

        # self.metrics_inbox is a communication queue used to
        #   receive the data that is placed in the metrics list.
        self.metrics_inbox = Queue()

        # self.metrics_inbox_daemon is a thread that moves the contents
        #   of self.metrics_inbox into self.metric.
        self.metrics_inbox_daemon = threading.Thread(
            target=self._metrics_inbox_daemon,
            daemon=True)

        # self.halt_metrics_inbox_daemon is an event that signals to the
        #   thread to break from its event loop and shut down. It is
        #   called when the trainer is garbage collected.
        self.halt_metrics_inbox_daemon = threading.Event()

        self.path_or_config = path_or_config
        if self.path_or_config is None:
            raise ValueError("Trainer requires either a path or a config argument.")
        self.process = Worker(self.outbox, self.inbox, self.metrics_inbox)
        self.process.start()
        self.inbox.get() # blocks until process is ready
        self.paused = True
        self.call("boot", path_or_config=self.path_or_config)

    def __del__(self):
        self.halt_metrics_inbox_daemon.set()

    def _metrics_inbox_daemon(self):
        while not self.halt_metrics_inbox_daemon.is_set():
            try:
                item = self.metrics_inbox.get(block=True,timeout=1.0)
                self.metrics.append(item)
            except:
                pass

    def start(self):
        if self.paused == True:
            self.call("start")
            self.paused = False

    def pause(self):
        if self.paused == False:
            self.paused = True
            self.call("pause")

    def stop(self):
        self.paused = None
        self.call("stop")
        self.process.join()

    def save(self, path="checkpoint.pt"):
        self.call("save", path=path)

    def update(self, entity, settings):
        self.call("update", entity=entity, setting=settings)

    def fetch(self, entity):
        return self.call("fetch", "entity"=entity)

    def autocomplete(self, prompt="", output_length=256, max_ctx=512):
        return self.call("autocomplete", prompt=prompt,
                         output_length=output_length, max_ctx=max_ctx)

    def status(self):
        return {"paused": self.paused}

    def call(self, instruction, **kwargs):
        if self.paused is None:
            raise RuntimeError("Trainer is defunct since stop has been called.")
        self.outbox.put({"instruction": instruction,
                         "kwargs": kwargs})
        result = self.inbox.get() # blocks
        if self.paused == False:
            self.outbox.put({"instruction": "start"})
            self.inbox.get()
        return result
