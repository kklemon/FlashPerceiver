import torch
import torch.nn as nn

from abc import abstractmethod, ABC
from typing import Optional
from math import pi
from einops import rearrange


T = torch.Tensor


class BasePositionalEncoding(nn.Module, ABC):
    """Base class for positional encoders.

    An implementation set values for in_dim and out_dim.

    Attributes:
        in_dim: Expected input dimensionality of the encoder.
        out_dim: Output dimensionality of the encoder.
    """
    in_dim: int
    out_dim: int

    @abstractmethod
    def forward(self, x: T) -> T:
        """foo"""
        raise NotImplementedError


class FourierPositionalEncoding(BasePositionalEncoding):
    """Projects an input by the given projection matrix before applying a sinus function.
    The input will be concatenated along the last axis.

    Args:
        proj_matrix: Projection matrix of shape ``(m, n)``.
        is_trainable: Whether the projection should be stored as trainable parameter. Default: ``False``

    Raises:
        ValueError: Raised if the given projection matrix does not have two dimensions.
    """
    def __init__(self, proj_matrix: T, is_trainable: bool = False):
        super().__init__()

        if proj_matrix.ndim != 2:
            raise ValueError(f'Expected projection matrix to have two dimensions but found {proj_matrix.ndim}')

        self.is_trainable = is_trainable

        if is_trainable:
            self.register_parameter('proj_matrix', nn.Parameter(proj_matrix))
        else:
            self.register_buffer('proj_matrix', proj_matrix)

        self.in_dim, self.out_dim = self.proj_matrix.shape

    def forward(self, x: T) -> T:
        channels = x.shape[-1]

        assert channels == self.in_dim, \
            f'Expected input to have {self.in_dim} channels but found {channels} channels instead)'

        x = torch.einsum('... i, i j -> ... j', x, self.proj_matrix)
        x = 2 * pi * x

        return torch.sin(x)


class IdentityPositionalEncoding(BasePositionalEncoding):
    """Positional encoder that returns the identity of the input."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, x: T) -> T:
        return x


class GaussianFourierFeatureTransform(FourierPositionalEncoding):
    """Implements the positional encoder proposed in (Tancik et al., 2020).

    Args:
        in_dim: Dimensionality of inputs.
        mapping_size: Dimensionality to map inputs to. Default: ``32``
        sigma: SD of the gaussian projection matrix. Default: ``1.0``
        is_trainable: Whether the projection should be stored as trainable parameter. Default: ``False``
        seed: Optional seed for the random number generator.

    Attributes:
        in_dim: Expected input dimensionality.
        out_dim: Output dimensionality (mapping_size * 2).
    """
    def __init__(
            self,
            in_dim: int,
            mapping_size: int = 32,
            sigma: float = 1.0,
            is_trainable: bool = False,
            seed: Optional[int] = None
    ):
        super().__init__(self.get_proj_matrix(in_dim, mapping_size, sigma, seed=seed), is_trainable=is_trainable)
        self.mapping_size = mapping_size
        self.sigma = sigma
        self.seed = seed

    @classmethod
    def get_proj_matrix(cls, in_dim, mapping_size, sigma, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.normal(mean=0, std=sigma, size=(in_dim, mapping_size), generator=generator)

    @classmethod
    def from_proj_matrix(cls, projection_matrix):
        in_dim, mapping_size = projection_matrix.shape
        feature_transform = cls(in_dim, mapping_size)
        feature_transform.projection_matrix.data = projection_matrix
        return feature_transform


class NeRFPositionalEncoding(FourierPositionalEncoding):
    """Implements the NeRF positional encoding from (Mildenhall et al., 2020).

    Args:
        in_dim: Dimensionality of inputs.
        num_frequency_bands: Number of frequency bands where the i-th band has frequency :math:`f_{i} = 2^{i}`.
            Default: ``10``

    Attributes:
        in_dim: Expected input dimensionality.
        out_dim: Output dimensionality (in_dim * n * 2).
    """
    def __init__(self, in_dim: int, num_frequency_bands: int = 10):
        super().__init__((2.0 ** torch.arange(num_frequency_bands))[None, :])
        self.num_frequency_bands = num_frequency_bands
        self.out_dim = num_frequency_bands * 2 * in_dim

    def forward(self, x: T) -> T:
        x = rearrange(x, '... -> ... 1') * self.proj_matrix
        x = pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = rearrange(x, '... i j -> ... (i j)')
        return x


def get_encoder(name: str, in_dim: int, **kwargs):
    encoders = {
        'identity': IdentityPositionalEncoding,
        'gaussian_fourier_features': GaussianFourierFeatureTransform,
        'nerf': NeRFPositionalEncoding
    }

    if name not in encoders:
        raise ValueError(f'Unknown encoder {name}. Must be one of {list(encoders)}.')

    return encoders[name](in_dim, **kwargs)
