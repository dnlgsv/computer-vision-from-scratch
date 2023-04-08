"""Vaswani et al. (https://arxiv.org/abs/1706.03762)."""

from typing import Union

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Feed Forward Network."""

    def __init__(self, model_dim: int, expantion_rate: int = 4, dropout: Union[float, int] = 0):
        super().__init__()
        self.fc1: nn.Module = nn.Linear(model_dim, model_dim * expantion_rate)
        self.act: nn.Module = nn.GELU()
        self.fc2: nn.Module = nn.Linear(model_dim * expantion_rate, model_dim)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, samples: Tensor):
        samples = self.act(self.fc1(samples))
        samples = self.dropout(samples)
        samples = self.fc2(samples)
        samples = self.dropout(samples)
        return samples


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention block."""

    def __init__(self, dim: int, heads: int, prob: float):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.query_map = nn.Linear(dim, dim)
        self.key_map = nn.Linear(dim, dim)
        self.value_map = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(prob)

    def forward(self, patches):
        batch_dim, patch_dim, feature_dim = patches.shape
        queries = self.query_map(patches).reshape(batch_dim, self.heads, patch_dim, feature_dim // self.heads)
        keys = (
            self.key_map(patches)
            .reshape(batch_dim, self.heads, patch_dim, feature_dim // self.heads)
            .permute(0, 1, 2, 1)
        )
        values = self.value_map(patches).reshape(
            batch_dim,
            self.heads,
            patch_dim,
            feature_dim // self.heads,
        )  # B x 12 x 197 x 64

        attention = self.softmax(torch.bmm(queries, keys) / torch.sqrt(self.dim))  # B x 12 x 197 x 197
        output = torch.bmm(attention, values).transpose(1, 2).reshape(batch_dim, patch_dim, feature_dim)
        output = self.output_proj(output)

        return self.dropout(output)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    pos_embedding_std = 0.02

    def __init__(self, dim: int, num_heads: int, expantion_rate: int = 4):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln1 = nn.LayerNorm(dim)
        self.self_attention = MultiHeadAttention(dim, num_heads, 0)

        # MLP block
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, expantion_rate)

    def forward(self, patches: torch.Tensor):
        iternal_output = self.ln1(patches)
        iternal_output = self.self_attention(iternal_output)
        iternal_output = iternal_output + patches
        output = self.ln2(iternal_output)
        output = self.mlp(output)
        return output + iternal_output
