import torch

from transformers.utils.general_utils import extract_patches


def test_extract_patches_health_check():
    random_image_square = torch.rand(3, 32, 32)
    extract_patches(random_image_square, (4, 4))


def test_extract_patches_corner_cases():
    random_image_arbitrary = torch.rand(3, 32, 37)
    extract_patches(random_image_arbitrary, (4, 4))
    extract_patches(random_image_arbitrary, (4, 6))
