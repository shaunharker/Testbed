import numpy as np

class ParameterInspector:
    def __init__(self, trainer):
        self.model = trainer.model
        self.trainer = trainer
        self.data = trainer.get_optimizer_stats()
        self.param = [p.detach() for p in trainer.model.parameters()]
        self.numels = np.array([p.numel() for p in self.param])
        self.cumsum_numels = np.cumsum(self.numels)
        self.numel = self.cumsum_numels[-1]

    def __len__(self):
        return self.numel

    def _address(self, idx):
        for (i, p) in enumerate(self.param):
            if idx < p.numel():
                return (i, idx)
            else:
                idx = idx - p.numel()
        raise ValueError(f"Not a valid parameter index.")

    def _reverse_address(self, i, j):
        if j < self.numels[i]:
            raise ValueError(f"Not a valid parameter index.")
        return self.cumsum_numels[i] + j

    def stats(self, idx):
        (i, j) = self._address(idx)
        weight = self.param[i].view(-1)[j].item()
        sum_grad = self.data[i]['sum_grad'].view(-1)[j].item()
        sum_sqr_grad = self.data[i]['sum_sqr_grad'].view(-1)[j].item()
        count = self.data[i]['count']
        mean = sum_grad / count
        if count > 1:
            var = (count / (count-1))*((sum_sqr_grad / count) - mean**2)
        else:
            var = 0
        return (weight, mean, var)

    def param_group(self, gidx):
        return self.param[gidx]

    def numpy(self):
        if len(self.data) > 0:
            weight = np.concatenate([self.param[i].view(-1).cpu().numpy() for i in range(len(self.param))]).reshape(-1)
            sum_grad = np.concatenate([self.data[i]['sum_grad'].view(-1).cpu() for i in range(len(self.param))]).reshape(-1)
            sum_sqr_grad = np.concatenate([self.data[i]['sum_sqr_grad'].view(-1).cpu() for i in range(len(self.param))]).reshape(-1)
            count = self.data[0]['count']
            mean = sum_grad / count
            var = (count / (count-1))*((sum_sqr_grad / count) - mean**2)
            return np.concatenate([weight.reshape(1,-1), mean.reshape(1,-1), var.reshape(1,-1)])
        else:
            raise RuntimeError("No data.")
