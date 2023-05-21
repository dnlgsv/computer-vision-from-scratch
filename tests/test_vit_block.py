import torch

from transformers.models.vit import VisionTransformer


def test_vit():
    images = torch.randn(4, 3, 224, 224)
    vit = VisionTransformer(image_size=224, patch_size=16, dim=768, num_classes=10, depth=12, num_heads=12)
    vit(images)
