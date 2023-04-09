import torch

from transformers.utils.general_utils import extract_patches


def test_extract_patches_health_check():
    random_image_square = torch.rand(1, 3, 32, 32)
    extract_patches(random_image_square, 4)


def test_extract_patches_correctness():
    random_image_square = torch.rand(1, 3, 32, 32)
    patches = extract_patches(random_image_square, 4)

    batch, num_of_patches, patch_dim = patches.shape

    assert batch == 1
    assert num_of_patches == (32 / 4) ** 2
    assert patch_dim == 4 * 4 * 3
