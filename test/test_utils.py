import torch

from transformers.utils.general_utils import extract_patches


def test_extract_patches_health_check():
    random_image_square = torch.rand(3, 32, 32)
    extract_patches(random_image_square, (4, 4))


def test_extract_patches_correctness():
    random_image_square = torch.rand(3, 32, 32)
    patches = extract_patches(random_image_square, (4, 4))

    num_of_patches, channels, height, width = patches.shape

    assert channels == 3
    assert height == width == 4
    assert num_of_patches == (32 / 4) ** 2
