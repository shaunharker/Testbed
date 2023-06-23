import torch
from torch.nn import Module, Embedding
from .nn import LanguageModel, Lambda, MLP, Sequential
from torch.cuda.amp import autocast

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
                        Embedding(n_vocab_in, d_model),
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

    # def double(self, idx):
    #     """
    #     Given a layer index, double the dimensionality of layer[idx]'s output space
    #     (and, correspondingly, layer[idx+1]'s input space. Fill in missing data with
    #     copied neural data from the same Linear module. For the layer[idx+1]'s weights
    #     you also need to scale the weights to make random proportions from the left
    #     copy's input and the right copy's input. This breaks the symmetry in further
    #     training.)
    #     """
    #     linear_layers = ([self.language_model.module.layers[2].module.layers[0]]
    #         +  list(self.language_model.module.layers[2].module.layers[1].layers)
    #         + [self.language_model.module.layers[2].module.layers[3]])
    #     firstlayer = layers[idx]
    #     secondlayer = layers[idx+1]
    #     # TODO: 
    #     # update firstlayer.weight, firstlayer.bias
    #     # and secondlayer.weight, secondlayer.bias by cloning data blockwise
    #     # and getting random weightings on the incoming cloned data to make it
    #     # operate equivalently right after doubling
    #     # update the self structure so the changes take effect in whatever
    #     # way is necessary.

    # def double(self, idx):
    #     """
    #     Given a layer index, double the dimensionality of layer[idx]'s output space
    #     (and, correspondingly, layer[idx+1]'s input space. Fill in missing data with
    #     copied neural data from the same Linear module. For the layer[idx+1]'s weights
    #     you also need to scale the weights to make random proportions from the left
    #     copy's input and the right copy's input. This breaks the symmetry in further
    #     training.)
    #     """
    #     linear_layers = (
    #         [
    #             self.language_model.module.layers[2].module.layers[0],
    #         ]
    #         + list(self.language_model.module.layers[2].module.layers[1].layers)
    #         + [self.language_model.module.layers[2].module.layers[3]]
    #     )
    #     firstlayer = linear_layers[idx]
    #     secondlayer = linear_layers[idx + 1]

    #     # update firstlayer's output dimension
    #     old_firstlayer_weight = firstlayer.weight.data
    #     old_firstlayer_bias = firstlayer.bias.data

    #     firstlayer.weight = torch.nn.Parameter(
    #         torch.cat([old_firstlayer_weight, old_firstlayer_weight], dim=0)
    #     )
    #     firstlayer.bias = torch.nn.Parameter(
    #         torch.cat([old_firstlayer_bias, old_firstlayer_bias], dim=0)
    #     )

    #     # update secondlayer's input dimension, mix original and copied input weights
    #     old_secondlayer_weight = secondlayer.weight.data

    #     left_input_weights = old_secondlayer_weight * (1.0/3.0)
    #     right_input_weights = old_secondlayer_weight * (2.0/3.0)

    #     new_secondlayer_weights = torch.cat(
    #         [left_input_weights, right_input_weights], dim=1
    #     )

    #     secondlayer.weight = torch.nn.Parameter(new_secondlayer_weights)

    def clone(self):
        return copy.deepcopy(self)
