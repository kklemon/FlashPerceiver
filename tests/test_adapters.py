import pytest
import torch

from flash_perceiver.adapters import ImageAdapter
from flash_perceiver.utils.encodings import NeRFPositionalEncoding


@pytest.fixture
def image():
    return torch.randn(32, 3, 128, 128)


def test_image_adapter(image):
    _, _, w, h = image.shape

    adapter = ImageAdapter(num_channels=3, embed_dim=512)
    out = adapter(image)

    assert out.shape == (32, w * h, 512)


def test_image_adapter_patching(image):
    _, _, w, h = image.shape

    adapter = ImageAdapter(
        num_channels=3,
        embed_dim=512,
        patch_size=(16, 16)
    )
    out = adapter(image)

    assert out.shape == (32, w * h / (16 ** 2), 512)


def test_image_adapter_pos_encoding(image):
    _, _, w, h = image.shape

    pos_encoding = NeRFPositionalEncoding(2)

    adapter = ImageAdapter(
        num_channels=3,
        embed_dim=512,
        pos_encoding=pos_encoding
    )
    out = adapter(image)

    assert out.shape == (32, w * h, 512)


def test_image_adapter_pos_encoding_with_patching(image):
    _, _, w, h = image.shape

    pos_encoding = NeRFPositionalEncoding(2)

    adapter = ImageAdapter(
        num_channels=3,
        embed_dim=512,
        pos_encoding=pos_encoding,
        patch_size=(16, 16)
    )
    out = adapter(image)

    assert out.shape == (32, w * h / (16 ** 2), 512)

def test_image_adapter_pos_encoding_with_patching(image):
    _, _, w, h = image.shape

    pos_encoding = NeRFPositionalEncoding(2)

    adapter = ImageAdapter(
        num_channels=3,
        embed_dim=512,
        pos_encoding=pos_encoding,
        patch_size=(16, 16)
    )
    out = adapter(image)

    assert out.shape == (32, w * h / (16 ** 2), 512)


@pytest.mark.parametrize('channel_first', [False, True])
def test_image_adapter_channel_first(image, channel_first):
    _, _, w, h = image.shape

    if not channel_first:
        image = image.permute(0, 2, 3, 1)
    
    pos_encoding = NeRFPositionalEncoding(2)

    adapter = ImageAdapter(
        num_channels=3,
        embed_dim=512,
        pos_encoding=pos_encoding,
        patch_size=(16, 16),
        channel_first=channel_first
    )
    out = adapter(image)

    assert out.shape == (32, w * h / (16 ** 2), 512)
