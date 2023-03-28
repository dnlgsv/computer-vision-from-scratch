from torch import Tensor


def extract_patches(image: Tensor, patch_size: tuple) -> Tensor:
    """
    Takes an image and returns a sequance of patches with NxN size

    Args:
        image (Tensor): input image
        patch_size (tuple): size of each patch

    Returns:
        Tensor: _description_
    """
    # patches = list()
    height, width = patch_size
    # for i in range(0, image.shape[1], height):
    #     for j in range(0, image.shape[2], width):
    #         patches.append()

    return image.unfold(0, height, width).unfold(1, height, width)
