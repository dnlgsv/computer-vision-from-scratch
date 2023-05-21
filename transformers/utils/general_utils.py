from torch import Tensor


def extract_patches(image: Tensor, patch_size: int = 16, is_flattened: bool = True) -> Tensor:
    """
    Takes an image and returns a sequance of patches with NxN size

    Args:
        image (Tensor): input image
        patch_size (tuple): size of each patch

    Returns:
        Tensor: _description_
    """
    assert image.shape[2] == image.shape[3]
    assert image.shape[2] % patch_size == 0

    batch_size = image.size(0)
    patches = (
        image.unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
        .reshape(batch_size, 3, -1, patch_size, patch_size)
        .transpose_(1, 2)
    )
    if is_flattened:
        return patches.flatten(2)

    return patches
