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
        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, seq_length: int, num_heads: int, hidden_dim: int, mlp_dim: int, norm_layer: nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads

        # Possitional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = 0

    def forward(self, input: torch.Tensor):
        x = input + self.pos_embedding
        x = self.ln_1(x)
        x, _ = self.self_attention(x)
        x = x + input
        y = self.ln_2(x)
        # y = self.mlp(y) # TODO: make normal mlp
        return x + y  # add layer norm in the end?


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        # num_layers: int,
        # num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer: nn.LayerNorm,
        num_classes: int = 10,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.norm_layer = norm_layer
