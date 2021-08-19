from random import randrange
import torch


class RandomTokensDataset:
    def __init__(self, n_tokens, n_vocab):
        self.n_tokens = n_tokens
        self.n_vocab = n_vocab
        self.data = torch.tensor([randrange(n_vocab) for _ in range(n_tokens)], dtype=torch.long, device='cuda')

    def expand(self, n_tokens):
        self.data = torch.cat([self.data, torch.tensor([randrange(self.n_vocab) for _ in range(n_tokens)], dtype=torch.long, device='cuda')])
        self.n_tokens = len(self.data)

    def batch(self, batch_size, example_length):
        example = lambda n: self.data[n:n+example_length].view(1,-1)
        rand_pos = lambda: randrange(self.n_tokens-example_length)
        return torch.stack([example(rand_pos()) for _ in range(batch_size)])
