"""Vaswani et al. (https://arxiv.org/abs/1706.03762)."""

import torch
from torch import nn


class LinearProjection(nn.Module):
    """_summary_.

    Args:
        input_dim (_type_): _description_
        output_dim (_type_): _description_
    """

    def __init__(self, input_dim, output_dim):
        """_summary_.

        Args:
            input_dim (_type_): _description_
            output_dim (_type_): _description_
        """
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
        """_summary_.
        """

        super(MultiHeadAttention, self).__init__()
        pass


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network.

    Args:
        dims (_type_): _description_
        ratio (int, optional): _description_. Defaults to 4.
        p (float, optional): _description_. Defaults to 0.1.
    """

    def __init__(self, dims, ratio=4, p=0.1):
        """_summary_.

        Args:
            dims (_type_): _description_
            ratio (int, optional): _description_. Defaults to 4.
            p (float, optional): _description_. Defaults to 0.1.
        """
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
        x = self.mapping(x)
        return


class OutlookAttention(nn.Module):
    """_summary_.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, channels, kernel, padding=1, stride=1):
        """_summary_.

        Args:
            channels (_type_): _description_
            kernel (_type_): _description_
            padding (int, optional): _description_. Defaults to 1.
            stride (int, optional): _description_. Defaults to 1.
        """
        super().__init__()

        self.channels = channels
        self.kernel = kernel
        self.padding = padding
        self.stride = stride

        self._value_projection = nn.Linear(channels, channels)
        self._attention = nn.Linear(channels, kernel**4)
        self._unfold = nn.Unfold(kernel, padding)

    def forward(self, image):
        """_summary_.

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size, height, width, channels = image.shape
        # value_matrix = self._value_projection(image).permute(0, 3, 1, 2)
        value_matrix = image.permute(0, 3, 1, 2)
        value_matrix_unfolded = self._unfold(value_matrix)
        value_matrix_unfolded = value_matrix_unfolded.reshape(1, self.channels, self.kernel**2, height * width)
        value_matrix_unfolded = value_matrix_unfolded.permute(2, 1, 0)

        attention_matrix = self._attention(image).reshape(self.height * self.width, self.kernel**2, self.kernel**2)

        self.fold = nn.Fold(
            output_size=(height, width), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride,
        )
        return attention, value


if __name__ == "__main__":
    outlooker = OutlookAttention(channels=2, kernel=2)
    img = torch.arange(4 * 4 * 2).reshape(1, 4, 4, 2).float()
    outlooker(img)
