import torch
from torch import nn

from transformers.utils.general_utils import extract_patches


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
