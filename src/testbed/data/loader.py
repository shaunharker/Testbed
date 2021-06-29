# deal with this somehow

class Loader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def refresh_loader(self):
        sampler = RandomSampler(dataset,
                                replacement=True)
        self.loader = DataLoader(self.dataset,
                                 sampler=sampler,
                                 batch_size=self.batch_size,
                                 pin_memory=True,
                                 drop_last=True)

    def batch(self, batch_size=None):
        self.set_batch_size(batch_size)
        try:
            item = next(self.loader)
        except StopIteration:
            refresh_loader()
            item = next(self.loader)
        return item

    def set_batch_size(self, batch_size):
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.refresh_loader()
