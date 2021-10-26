from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, LayerNorm, Identity
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros, ones, normal

from typing import Optional

from torch_scatter import scatter, segment_csr, gather_csr

from torch_geometric.utils.num_nodes import maybe_num_nodes


def normalize(src: Tensor, index: Tensor, ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None) -> Tensor:
    r"""Computes a sparsely evaluated normalization.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if ptr is None:
        N = maybe_num_nodes(index, num_nodes)
        count = scatter(torch.ones_like(src), index, dim=0, dim_size=N, reduce='sum')[index]
        diff = src - scatter(src, index, dim=0, dim_size=N, reduce='mean')[index]
        variance = scatter(diff * diff, index, dim=0, dim_size=N, reduce='sum')[index] + 1e-16
    else:
        count = gather_csr(segment_csr(torch.ones_like(src), ptr, reduce='sum'), ptr)
        diff = src - gather_csr(segment_csr(src, ptr, reduce='mean'), ptr)
        variance = gather_csr(segment_csr(diff * diff, ptr, reduce='sum'), ptr) + 1e-16
    return diff / (variance / (count + 1e-16)).sqrt()


class TransformerConv(MessagePassing):
    r"""The graph normalized attentional operator based on the paper
    `"Normalized Attention Without Probability Cage"
    <https://arxiv.org/abs/2005.09561>`
    """

    def __init__(self, model_dimension: int, heads: int = 1, model: str = 'NAP', **kwargs):
        if model == 'max':
            super(TransformerConv, self).__init__(aggr='max', node_dim=0, **kwargs)
        else:
            super(TransformerConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.model_dim = model_dimension
        self.heads = heads
        self.model = model
        if model not in ['sum', 'max']:
            self.key_layer = Linear(model_dimension, model_dimension)
            self.query_layer = Linear(model_dimension, model_dimension)
            self.value_layer = Linear(model_dimension, model_dimension)
            self.gain = Parameter(torch.Tensor(1, heads))
            self.shift = Parameter(torch.Tensor(1, heads))
            self.reset_parameters()
            self.layer_norm_1 = Identity() if model == 'BERT' else LayerNorm(model_dimension)
            self.mixing_layer = Linear(model_dimension, model_dimension)
            ff_hidden_dim = 4 * model_dimension
        else:
            factor = (12.0 * model_dimension + 8) / (2 * model_dimension + 1)
            ff_hidden_dim = round(factor * model_dimension)
        self.layer_norm_2 = LayerNorm(model_dimension)

        self.ff1 = Linear(model_dimension, ff_hidden_dim)
        self.layer_norm_3 = Identity() if model == 'BERT' else LayerNorm(ff_hidden_dim)
        self.ff2 = Linear(ff_hidden_dim, model_dimension)
        self.layer_norm_4 = LayerNorm(model_dimension)

    def reset_parameters(self):
        ones(self.gain)
        zeros(self.shift)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if self.model in ['sum', 'max']:
            keys = queries = None
            values = x
        else:
            keys = self.key_layer(x).view(-1, self.heads, self.model_dim // self.heads)
            queries = self.query_layer(x).view(-1, self.heads, self.model_dim // self.heads)
            values = self.value_layer(x).view(-1, self.heads, self.model_dim // self.heads)

        if isinstance(edge_index, Tensor):
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        elif isinstance(edge_index, SparseTensor):
            edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out1 = self.propagate(edge_index, x=(values, values), alpha=(queries, keys))
        out1 = out1.view(-1, self.model_dim)

        if self.model not in ['sum', 'max']:
            out1 = self.layer_norm_1(out1)
            if self.model != 'BERT':
                out1 = F.gelu(out1)
            out1 = self.mixing_layer(out1)

        if self.model == 'BERT':
            out1 = out1 + x
            pooling_out = self.layer_norm_2(out1)
        else:
            out1 = self.layer_norm_2(out1)
            pooling_out = out1 + x

        # ff
        ff_hidden = self.ff1(pooling_out)
        ff_hidden = self.layer_norm_3(ff_hidden)
        ff_hidden = F.gelu(ff_hidden)
        ff_hidden = self.ff2(ff_hidden)

        if self.model == 'BERT':
            ff_hidden = pooling_out + ff_hidden
            out = self.layer_norm_4(ff_hidden)
        else:
            ff_hidden = self.layer_norm_4(ff_hidden)
            out = pooling_out + ff_hidden
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if self.model in ['sum', 'max']:
            return x_j
        alpha = alpha_j * alpha_i
        alpha = alpha.sum(dim=-1)
        if self.model in ['BERT', 'MTE']:
            alpha /= np.sqrt(self.model_dim / self.heads)
            alpha = softmax(alpha, index, ptr, size_i)
        elif self.model == 'NON':
            N = maybe_num_nodes(index, size_i)
            count = scatter(torch.ones_like(alpha), index, dim=0, dim_size=N, reduce='sum')[index]
            alpha /= count.sqrt()
        else:
            alpha = normalize(alpha, index, ptr, size_i)
            alpha = self.gain * alpha + self.shift
        return x_j * alpha.unsqueeze(-1)

