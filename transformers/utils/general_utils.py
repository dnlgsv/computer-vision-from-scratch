from torch import Tensor


def extract_patches(image: Tensor, patch_size: tuple = (16, 16), is_flattened: bool = True) -> Tensor:
    """
    Takes an image and returns a sequance of patches with NxN size

    Args:
        image (Tensor): input image
        patch_size (tuple): size of each patch

    Returns:
        Tensor: _description_
    """
    height, width = patch_size
    assert patch_size[0] == patch_size[1]
    assert image.shape[1] == image.shape[2]
    assert image.shape[1] % height == 0

    number_of_patches = image.shape[1] // height * image.shape[2] // width
    patches = (
        image.unfold(1, height, width)
        .unfold(2, height, width)
        .reshape(3, number_of_patches, height, width)
        .permute(1, 0, 2, 3)
    )
    if is_flattened:
        return patches.flatten(1)

    return patches
