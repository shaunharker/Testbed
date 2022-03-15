from .nn import LanguageModel, Embedding, Lambda, MLP

class MLPLM(Module):
    def __init__(self, n_ctx, n_vocab_in, d_model, d_hidden, nonlinearity, n_vocab_out, autocast_enabled=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.n_vocab_out = n_vocab_out
        self.autocast_enabled = autocast_enabled or False
        self.language_model = (
            LanguageModel(
                n_vocab_out=n_vocab_out,
                mode="last",
                module=(
                    Sequential(
                        Embedding(n_classes=n_vocab_in, d_model=d_model),
                        Lambda(lambda x: x.view(-1,n_ctx*d_model)),
                        MLP(d_in=n_ctx*d_model,
                            d_hidden=d_hidden,
                            nonlinearity=nonlinearity,
                            d_out=n_vocab_out),
                        Lambda(lambda x: x.view(-1, 1, n_vocab_out))))))

    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model(x)

    @torch.no_grad()
    def inference(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model.inference(x)

    def clone(self):
        return copy.deepcopy(self)
