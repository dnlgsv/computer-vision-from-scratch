"""Vaswani et al. (https://arxiv.org/abs/1706.03762)"""

import torch
from torch import nn


class LinearProjection(nn.Module):
    """_summary_

    Args:
        input_dim (_type_): _description_
        output_dim (_type_): _description_
    """

    def __init__(self, input_dim, output_dim):

        super(LinearProjection, self).__init__()
        self.projector = nn.Linear(input_dim, output_dim)
        # TODO: Apply weights init

    def forward(self, patches):
        return self.projector(patches)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        pass


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network.

    Args:
        dims (_type_): _description_
        ratio (int, optional): _description_. Defaults to 4.
        p (float, optional): _description_. Defaults to 0.1.
    """

    def __init__(self, dims, ratio=4, p=0.1):

        super(FeedForwardNetwork, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._p = p
        self._h_dims = int(dims * ratio)

        self.mapping = nn.Sequential(
            nn.Linear(dims, self._h_dims),
            nn.ReLu(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(self._h_dims, dims),
            nn.Dropout(p=p),
        )

        # TODO: Apply weights init

    def forward(self, x):
        x = self.mapping(x)
        return
