import torch
from torch import nn

from transformers.layers.blocks import EncoderBlock
from transformers.utils.general_utils import extract_patches


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int = 768,
        num_classes: int = 10,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size**2

        # Patch dim to model dim
        self.mlp = nn.Linear(patch_dim, dim)

        # Possitional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Encoder blocks
        self.num_classes = num_classes
        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(
                    dim,
                    num_heads,
                )
                for _ in range(depth)
            ],
        )

        self.mlp_head = nn.Linear(dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images: torch.Tensor):
        # Expand
        cls_token = self.cls_token.expand(images.size(0), -1, -1)

        patches = extract_patches(images)
        patches_linear_projected = self.mlp(patches)

        patches_and_cls_token = torch.cat((cls_token, patches_linear_projected), dim=1)
        patches_and_pos_embedding = patches_and_cls_token + self.pos_embedding

        patches_encoded = self.encoder_blocks(patches_and_pos_embedding)

        patches_mlp_head = self.mlp_head(patches_encoded[:, 0])
        return self.softmax(patches_mlp_head)
