from .nn import SplitExample, Sequential, Lambda, Nonlinearity, CrossEntropyLoss, Softmax, ResidualLayerNorm, MLP, LanguageModel
from .mlplm import MLPLM
from .transformer import TransformerLMHead
#from .mutabletransformer import MutableTransformerLM

# from transformers import GPT2LMHeadModel, GPT2Config
# import torch

# def migrate_model(old_model, new_config):
#     # Initialize a new model with the new config
#     new_model = GPT2LMHeadModel(new_config)
    
#     # Migrate parameters from old model to new model
#     for old_param_key, new_param_key in zip(old_model.state_dict(), new_model.state_dict()):
#         old_param = old_model.state_dict()[old_param_key]
#         new_param = new_model.state_dict()[new_param_key]

#         if old_param.size() == new_param.size():
#             # If the old parameter has the same size as the new one, 
#             # just copy the parameter values
#             new_model.state_dict()[new_param_key].data.copy_(old_param.data)
#         elif len(old_param.size()) == len(new_param.size()):
#             # If the old parameter and the new parameter have the same number of dimensions, 
#             # but different sizes, we can copy the original values and 
#             # initialize the rest randomly
#             slice_obj = [slice(0, min(dim_old, dim_new)) for dim_old, dim_new in zip(old_param.size(), new_param.size())]
#             new_model.state_dict()[new_param_key].data[slice_obj].copy_(old_param.data[slice_obj])
#             remaining_dims = [slice(dim_old, dim_new) for dim_old, dim_new in zip(old_param.size(), new_param.size())]
#             new_model.state_dict()[new_param_key].data[remaining_dims].normal_(0, 0.02)
#         else:
#             # In this case, the parameters are not compatible and need to be initialized from scratch
#             # An example could be the parameters of the extra layers when n_layers is increased
#             new_model.state_dict()[new_param_key].data.normal_(0, 0.02)
    
#     return new_model