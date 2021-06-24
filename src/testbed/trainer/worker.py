import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
from ..util import IgnoreKeyboardInterrupt, Reporter, Stopwatch

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, loss_outbox):
        super().__init__(daemon=True)
        self.inbox = inbox
        self.outbox = outbox
        self.loss_outbox = loss_outbox

    def run(self):
        inbox = self.inbox
        outbox = self.outbox
        loss_outbox = self.loss_outbox
        with IgnoreKeyboardInterrupt():
            outbox.put("ready")
            model = inbox.get()
            dataset = inbox.get()
            optimizer = inbox.get()
            compute_time = inbox.get()
            step = inbox.get()
            optimizer = AdamW(model.parameters())
            print(f"{step}. This is the worker process.")
            if step > 0:
                print(f"Model has been trained for {compute_time}s so far.")
            # reporter = Reporter()
            parent = torch.multiprocessing.parent_process()
            waiting = False
            while True:
                if not waiting:
                    print(f"{step}. Waiting for instruction.")
                    waiting = True
                try:
                    instruction = inbox.get(True,1.0)
                    waiting = False
                except:
                    if not parent.is_alive():
                        print(f"{step}. Orphaned, exiting.")
                        instruction = "stop"
                        situation = "break"
                        break
                    continue
                print(f"{step}. Received instruction '{instruction}'.")
                if instruction == "start":
                    with Stopwatch() as stopwatch:
                        situation = "normal"
                        while situation == "normal":
                            print(f"{step}. Entering training loop. B={dataset.B} N={dataset.N}")
                            model.train()
                            for X in DataLoader(dataset, batch_size=None, shuffle=False):
                                if not parent.is_alive():
                                    situation = "break"
                                    break
                                ######################
                                # THE ACTUAL PROGRAM #
                                ######################
                                optimizer.zero_grad()
                                loss = model(X)
                                loss.backward()
                                optimizer.step()
                                step += 1
                                ######################
                                # reporter.step(loss.item())
                                #print(step, 'out')
                                loss_outbox.put((step, compute_time + stopwatch.time_elapsed, loss.item()))
                                try:
                                    instruction = inbox.get(False)
                                    if instruction != "start":
                                        print(f"{step}. Interrupted by instruction '{instruction}'.")
                                        situation = "break"
                                        break
                                except:
                                    pass
                    compute_time += stopwatch.total_run_time
                    print(f"{step}. Exiting compute loop.")
                if instruction == "pause":
                    outbox.put("paused")
                    continue
                if instruction == "stop":
                    break
                if instruction == "set_batch_size":
                    print(f"{step}. Setting new batch size.")
                    batch_size = inbox.get()
                    dataset.set_batch_size(batch_size)
                    print(f"{step}. There are {len(dataset)} batches.")
                    continue
                if instruction == "set_example_length":
                    print(f"{step}. Setting new example length.")
                    example_length = inbox.get()
                    dataset.set_example_length(example_length)
                    print(f"{step}. The example length is {example_length} tokens.")
                    continue

        print(f"{step}. Exiting process.")
