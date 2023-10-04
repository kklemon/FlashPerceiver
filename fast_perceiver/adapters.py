from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torch
import torch.nn as nn

from fast_perceiver import utils
from fast_perceiver.utils.encodings import BasePositionalEncoding


class ImageAdapter(nn.Module):
    """
    Adapter for images as input to the Perceiver.

    Can optionally patch the image and add positional encodings.

    Args:
        embed_dim: Dimensionality of the final output, the Perceiver input respectively.
        num_channels: Number of channels of the input image.
        pos_encoding: Positional encoding module to use.
            Defaults to `None`, i.e. no positional encoding is applied.
        patch_size: Size of the patches to extract from the image.
            Can be a single integer or a tuple representing `(width, height)` of the patches.
            Default to `None`, i.e. no patching.
        channel_first: Whether the input image has the channels first or last.
            Defaults to `True`, i.e. channels first.
    """
    def __init__(
        self,
        embed_dim: int,
        num_channels: int = 3,
        pos_encoding: BasePositionalEncoding | None = None,
        patch_size: int | tuple[int, int ] | None = None,
        channel_first: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.pos_encoding = pos_encoding
        self.channel_first = channel_first

        if patch_size is not None:
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)

            self.patchify = Rearrange(
                'b (h p1) (w p2) c -> b h w (p1 p2 c)',
                p1=self.patch_size[0],
                p2=self.patch_size[1]
            )
            self.patch_dim = np.prod(patch_size) * num_channels
        else:
            self.patchify = None
            self.patch_dim = num_channels
        
        if pos_encoding is not None:
            self.patch_dim += pos_encoding.out_dim

        self.proj = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_grid = None

    def get_pos_grid(self, x):
        b, h, w, _ = x.shape

        if self.pos_grid is None or self.pos_grid.shape[0] < w or self.pos_grid.shape[1] < h:
            self.pos_grid = utils.meshgrid(h, w).to(x.device)

        pos_grid = self.pos_grid[:h, :w]
        pos_grid = repeat(pos_grid, 'h w c -> b h w c', b=b)

        return pos_grid
    
    def forward(self, x):
        assert x.ndim == 4, \
            f'Expected input to have four dimensions but found {x.ndim} dimensions instead'
        
        if self.channel_first:
            x = rearrange(x, 'b c h w -> b h w c')
        
        assert x.shape[-1] == self.num_channels, \
            f'Expected input to have {self.num_channels} channels but found {x.shape[-1]} channels instead'
    
        if self.patchify is not None:
            x = self.patchify(x)
        
        if self.pos_encoding is not None:
            pos_grid = self.get_pos_grid(x)
            x = torch.cat([x, self.pos_encoding(pos_grid)], dim=-1)

        x = self.proj(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        return x
