import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")

from ..util import IgnoreKeyboardInterrupt, Reporter, Stopwatch, default_device
import time

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
            N = inbox.get()
            B = inbox.get()
            optimizer = inbox.get()
            dataset = inbox.get()
            compute_time = inbox.get()
            step = inbox.get()
            losses = inbox.get()
            dataset.cache_data()
            outer_batch = 1 # if batches can't fit, we'll divide it up and loop before calling optimizer
            if len(optimizer.state_dict()['state']) == 0:
                optimizer = torch.optim.AdamW(model.parameters()) #can we use same class? check python docs...
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
                print(f"{step}. Received instruction '{instruction}' at time {compute_time}.")
                if instruction == "start":
                    with Stopwatch() as stopwatch:
                        situation = "normal"
                        while situation == "normal":
                            print(f"{step}. Entering training loop at time {compute_time + stopwatch.time_elapsed}. B={B} N={dataset.N}")
                            model.train()
                            context_entry_time = time.time() # We only want to time the interior of the context, not the entry and exit
                            context_exit_time = None
                            first_pass = True
                            running_loss = 0.0
                            if B > 8192:
                                outer_batch = B//8192
                                inner_batch = 8192
                            else:
                                outer_batch = 1
                                inner_batch = B
                            starting_modulus = step % outer_batch
                            outer_batch_idx = 0
                            for X in DataLoader(dataset,
                                                batch_size=inner_batch,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=True):
                                if first_pass:
                                    context_entry_time = time.time() - context_entry_time
                                    first_pass = False
                                X = X.to(device='cuda',dtype=torch.long, non_blocking=True)
                                if not parent.is_alive():
                                    situation = "break"
                                    context_exit_time = time.time() # we're crashing, but rules are rules after all
                                    break
                                ######################
                                # THE ACTUAL PROGRAM #
                                ######################
                                if outer_batch_idx % outer_batch == 0:
                                    running_loss = 0
                                    optimizer.zero_grad()
                                loss = model(X)
                                loss.backward(retain_graph = True)
                                running_loss += loss.item()
                                if (outer_batch_idx+1) % outer_batch == 0:
                                    optimizer.step()
                                    step += 1
                                ######################
                                outer_batch_idx += 1
                                context_exit_time = time.time() # just in case
                                if outer_batch_idx % outer_batch == 0:
                                    step_info = [step, compute_time + stopwatch.time_elapsed - context_entry_time, running_loss/outer_batch ]
                                    running_loss = 0.0
                                    losses.append(step_info)
                                    loss_outbox.put(step_info)
                                    #print(step, compute_time + stopwatch.time_elapsed - context_entry_time)
                                    try:
                                        instruction = inbox.get(False)
                                        if instruction != "start":
                                            print(f"{step}. Interrupted at time {step_info[1]} by instruction '{instruction}'.")
                                            situation = "break"
                                            break
                                    except:
                                        pass
                            context_exit_time = time.time() - context_exit_time
                    compute_time += stopwatch.total_run_time - context_entry_time - context_exit_time
                    print(f"{step}. Exiting compute loop at time {compute_time}.")
                if instruction == "pause":
                    outbox.put("paused")
                    continue
                if instruction == "stop":
                    break
                if instruction == "set_batch_size":
                    print(f"{step}. Setting new batch size.")
                    B = inbox.get()
                    print(f"{step}. There are {len(dataset)//B} batches.")
                    continue
                if instruction == "set_example_length":
                    print(f"{step}. Setting new example length.")
                    example_length = inbox.get()
                    dataset.set_example_length(example_length)
                    print(f"{step}. The example length is {example_length} tokens.")
                    continue
                if instruction == "save":
                    savefile = inbox.get()
                    checkpoint = {
                        'time': compute_time,
                        'step': step,
                        'loss': losses,
                        'model': model,
                        'sequence_length': N,
                        'batch_size': B,
                        'optimizer': optimizer,
                        'datafile': dataset.filename}
                    torch.save(checkpoint, savefile)
                    outbox.put("saved")
        print(f"{step}. Exiting process.")
