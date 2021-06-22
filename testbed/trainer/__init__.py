import torch
from torch.utils.data import DataLoader
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
from util import decode_broken_utf8, IgnoreKeyboardInterrupt, Reporter

def trainer_worker(inbox, outbox, loss_outbox):
    with IgnoreKeyboardInterrupt():
        outbox.put("ready")
        model = inbox.get()
        dataset = inbox.get()
        UserOptimizer = inbox.get()
        batch_size = inbox.get()
        shuffle = inbox.get()
        optimizer = UserOptimizer(model.parameters())
        reporter = Reporter()
        parent = torch.multiprocessing.parent_process()
        compute_time = 0.0
        waiting = False
        dataset.set_batch_size(batch_size)
        model.train()
        while True:
            if not waiting:
                print(f"Waiting for instruction. {reporter.n} steps so far.")
                waiting = True
            try:
                instruction = inbox.get(True,1.0)
                waiting = False
            except:
                if not parent.is_alive():
                    print("Orphaned, exiting.")
                    instruction = "stop"
                    situation = "break"
                    break
                continue
            print(f"Received instruction '{instruction}'.")
            if instruction == "start":
                with Stopwatch() as stopwatch:
                    situation = "normal"
                    while situation == "normal":
                        print(f"Beginning epoch. batch_size={batch_size}, shuffle={shuffle}")
                        for X in DataLoader(dataset, batch_size=None, shuffle=shuffle):
                            optimizer.zero_grad()
                            loss = model(X)
                            loss.backward()
                            optimizer.step()
                            reporter.step(loss.item())
                            loss_outbox.put((reporter.n, compute_time + stopwatch.time_elapsed, loss.item()))
                            try:
                                instruction = inbox.get(False)
                                if instruction != "start":
                                    print(f"Interrupted by instruction '{instruction}'.")
                                    situation = "break"
                                    break
                            except:
                                pass
                compute_time += stopwatch.total_run_time
                print("Exiting compute loop.")
            if instruction == "pause":
                outbox.put("paused")
                continue
            if instruction == "stop":
                break
            if instruction == "set_batch_size":
                print("Setting new batch size.")
                batch_size = inbox.get()
                dataset.set_batch_size(batch_size)
                continue
            if instruction == "set_batch_permutation":
                print("Receiving permutation.")
                perm = inbox.get()
                print("Setting new permutation.")
                dataset.set_permutation(perm)
                continue
    print("Exiting process.")
