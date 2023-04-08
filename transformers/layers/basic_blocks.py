"""Vaswani et al. (https://arxiv.org/abs/1706.03762)."""

import torch
from torch import nn

from transformers.utils.general_utils import extract_patches


class LinearProjection(nn.Module):
    """_summary_.

    Args:
        input_dim (_type_): _description_
        output_dim (_type_): _description_
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)
        # TODO: Apply weights init

    def forward(self, patches):
        """_summary_.

        Args:
            patches (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.projector(patches)


class MultiHeadAttention(nn.Module):
    """_summary_.

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        pass


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network.

    Args:
        dims (_type_): _description_
        ratio (int, optional): _description_. Defaults to 4.
        p (float, optional): _description_. Defaults to 0.1.
    """

    def __init__(self, dims, ratio=4, p=0.1):
        super().__init__()

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
        return self.mapping(x)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    pos_embedding_std = 0.02

    def __init__(self, seq_length: int, num_heads: int, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.layern_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = None  # TODO: not implemented yet

        # MLP block
        self.layern_2 = nn.LayerNorm(hidden_dim)
        self.mlp = FeedForwardNetwork(hidden_dim)

    def forward(self, patches: torch.Tensor):
        x = self.ln_1(patches)
        x, _ = self.self_attention(x)
        x = x + patches
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y  # add layer norm in the end?


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_classes: int = 10,
    ):
        super().__init__()
        # Possitional embedding
        self.num_of_patches = image_size[1] // patch_size[0] * image_size[2] // patch_size[1]
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim).normal_(std=self.pos_embedding_std))
        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.num_of_patches, hidden_dim).normal_(std=self.pos_embedding_std)
        )
        self.mlp = nn.Linear(3 * patch_size[0] ** 2, hidden_dim)
        self.mlp_head = nn.Linear(hidden_dim, num_classes)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.encoder = EncoderBlock()
        self.softmax = nn.Softmax(dim=1)

    def forward(self):
        patches = extract_patches(image)  # or batch of images??
        patches_linear_projected = self.mlp(patches)
        patches_and_cls_token = torch.concat(patches_linear_projected, self.cls_token)
        patches_and_pos_embedding = patches_and_cls_token + self.pos_embedding
        patches_encoded = self.encoder(patches_and_pos_embedding)
        patches_mlp_head = self.mlp_head(patches_encoded)
        return self.softmax(patches_mlp_head)
