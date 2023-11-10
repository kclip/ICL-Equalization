import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import numpy as np
from parameters import parameter_reading

args = parameter_reading()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(embedding_dim, n_positions,num_heads, num_layers, data_dim):
    model = TransformerModel(
        n_dims=data_dim,
        n_positions=2 *n_positions,
        n_embd=embedding_dim,
        n_layer=num_layers,
        n_head=num_heads,
    )
    return model

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd, n_layer, n_head):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, self.n_dims)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, ys_batch, xs_batch, inds=None):
        if inds is None:
            inds = torch.arange(xs_batch.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= xs_batch.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(ys_batch, xs_batch)
        zs = zs.to(torch.float32)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        '''Mapping to Constallation Symbol'''
        prediction = (torch.sigmoid(prediction)-0.5)*np.sqrt(2)
        bsize, points, dim = ys_batch.shape
        return prediction[:, ::2, :]