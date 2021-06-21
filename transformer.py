import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, ntoken=256, ninp=512, nhead=8, nhid=256, nlayers=2, dropout=0.5):
        super(Transformer, self).__init__()
        self.L = 64
        self.hyp = {
            "ntoken": ntoken,
            "ninp": ninp,
            "nhead": nhead,
            "nhid": nhid,
            "nlayers": nlayers,
            "dropout": dropout}
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, X, has_mask=True):
        x = X[...,:-1] # shape [B, N]
        y = X[...,1:]  # shape [B, N]
        x = torch.transpose(x, -1, -2) # shape [N, B]
        y = torch.transpose(y, -1, -2) # shape [N, B]
        x = self.encoder(x) * math.sqrt(self.hyp['ninp']) # shape [N, B, E]
        x = self.pos_encoder(x) # shape [N, B, E]
        if has_mask:
            device = X.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = self._generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        x = self.transformer_encoder(x, self.src_mask) # shape [N, B, E]
        x = self.decoder(x).view(-1,self.hyp['ntoken']) # shape [N*B, K]
        y = y.view(-1) # shape [N*B]
        return self.criterion(x,y)# self.softmax(output, dim=-1)

    def probs(self, X, has_mask=True):
        if X.dim == 1:
            X = torch.unsqueeze(X, 0)
        x = X # shape [B, N]
        x = torch.transpose(x, 0, 1) # shape [N, B]
        x = self.encoder(x) * math.sqrt(self.hyp['ninp']) # shape [N, B, E]
        x = self.pos_encoder(x) # shape [N, B, E]
        if has_mask:
            device = X.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = self._generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        x = self.transformer_encoder(x, self.src_mask) # shape [N, B, E]
        x = self.decoder(x)[-1,:,:].squeeze(0) # shape [B, K]
        return self.softmax(x) # shape [B]
