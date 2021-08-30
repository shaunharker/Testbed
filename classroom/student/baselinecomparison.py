import torch
from math import sqrt


class BaselineComparison:
    """
    The purpose of this class is to attempt to predict the
    performance of a model in training (henceforth, "the target model")
    using a snapshot of its parameter weights (henceforth, "the baseline model").

    Example usage pattern:

    ```python
    model = MyModel()
    baseline_model = copy.deepcopy(model)
    baseline = BaselineComparison(baseline_model=baseline_model)
    for n in range(2**16):
        batch = get_training_batch()
        losses = model(batch)
        predicted_losses, baseline_losses = (
            baseline(
                batch=batch,
                n=n,
                target_losses=losses))
        # ... now plot, call backward, optimize on model, etc ...
    ```

    In order to perform predictions the BaselineComparison object `self`
    creates a model consisting of parameters `self.a` and `self.b` to predict
    the target model loss according to

    ```python
    baseline_losses = self.baseline_model(batch)
    predicted_losses = (
        baseline_losses +
        self.a*n*torch.ones_like(baseline_losses) +
        self.b*n*baseline_losses)
    ```

    We train the prediction model parameters `self.a` and `self.b` based on the mean square error criterion
    ```python
    mse = 0.5 * (predicted_losses - target_losses)**2
    ```
    using an Adam optimizer which stores its parameters in the attributes `self.da`, `self.da2`, `self.db`, and `self.db2`.

    For futher details see the individual method docstrings and explore the code.
    """
    def __init__(self, baseline_model):
        # baseline model parameters
        self.baseline_model = baseline_model
        # prediction model parameters
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 1.0
        # prediction model optimization parameters
        self.da = 0.0
        self.db = 0.0
        self.da2 = 0.0
        self.db2 = 0.0
        self.dc = 0.0
        self.dc2 = 0.0
        self.dd = 0.0
        self.dd2 = 0.0

    def __call__(self, batch, n, target_losses):
        return self.update(batch, n, target_losses)

    @torch.no_grad()
    def update(self, batch, n, target_losses, lr=None, beta1=None, beta2=None, eps=None):
        """
        I don't know the correct update algorithm yet, so I'm going to go with Adam and tweak
        parameters and see what happens to start with.
        kwargs = {
        "self": "the BaselineComparison object",
        "batch": "a batch of examples, a tensor of shape (batch_size, example_length)",
        "n": "the number of steps the target model has been trained since making the baseline copy",
        "target_loss": "the loss of the target model for the given batch",
        "lr": "optional optimizer parameter for learning rate in Adam optimization",
        "beta1": "optional optimizer parameter for EMA on first moment of gradient in Adam optimization",
        "beta2": "optional optimizer parameter for EMA on second moment of gradient in Adam optimization",
        "eps": "optional optimizer parameter for numerical stability in Adam optimization"
        }
        """
        # arg parsing
        lr = lr or 1e-8*(1000/n) if n > 1000 else 1e-6*n/1000
        beta1 = beta1 or 0.9
        beta2 = beta2 or 0.999
        eps = eps or 1e-8
        lr = lr if type(lr) == float else lr(n)
        beta1 = beta1 if type(beta1) == float else beta1(n)
        beta2 = beta2 if type(beta2) == float else beta2(n)
        eps = eps if type(eps) == float else eps(n)
        # model evaluation
        baseline_losses = self.baseline_model(batch)
        predicted_losses = self.c + self.d*baseline_losses + self.a*n*torch.ones_like(baseline_losses) + self.b*n*baseline_losses
        # mse = torch.mean((predicted_losses - target_losses)**2 / 2.0)
        avg_error = torch.mean(predicted_losses - target_losses).item()
        avg_baseline_loss = torch.mean(baseline_losses).item()
        # adam optimization, though perhaps it is not best suited for this use case. leaving to experiments to decide.
        self.da = beta1*self.da+(1-beta1)*(n*avg_error)
        self.da2 = beta2*self.da2+(1-beta2)*(n*avg_error)**2
        self.db = beta1*self.db+(1-beta1)*(n*avg_baseline_loss*avg_error)
        self.db2 = beta2*self.db2+(1-beta2)*(n*avg_baseline_loss*avg_error)**2
        self.dc = beta1*self.dc+(1-beta1)*(avg_error)
        self.dc2 = beta2*self.dc2+(1-beta2)*(avg_error)**2
        self.dd = beta1*self.dd+(1-beta1)*(avg_baseline_loss*avg_error)
        self.dd2 = beta2*self.dd2+(1-beta2)*(avg_baseline_loss*avg_error)**2
        self.a -= lr*self.da/(sqrt(self.da2)+eps)
        self.b -= lr*self.db/(sqrt(self.db2)+eps)
        self.c -= lr*self.dc/(sqrt(self.dc2)+eps)
        self.d -= lr*self.dd/(sqrt(self.dd2)+eps)
        return predicted_losses, baseline_losses

    @torch.no_grad()
    def query(self, batch, n):
        """
        Return the predicted target loss given a batch and the number of steps of target model
        training since taking the baseline.

        ```python
        kwargs = {
        "self": "the BaselineComparison object",
        "batch": "a batch of examples, a tensor of shape (batch_size, example_length)",
        "n": "the number of steps the target model has been trained since making the baseline copy"
        }
        ```
        """
        baseline_losses = self.baseline_model(batch)
        predicted_losses = baseline_losses + self.a*n*torch.ones_like(baseline_losses) + self.b*n*baseline_losses
        return predicted_losses
