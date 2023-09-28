import pytest
from sympy import N
import torch

from fast_perceiver import utils
from fast_perceiver.perceiver import PerceiverBase


@pytest.mark.parametrize('input_dim', [32, 64])
@pytest.mark.parametrize('depth', [1, 4])
@pytest.mark.parametrize('num_latents', [32])
@pytest.mark.parametrize('latent_dim', [128])
@pytest.mark.parametrize('self_per_cross_attn', [1, 2])
@pytest.mark.parametrize('input_length', [128, 256])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('mask', [False, True])
def test_model(
    input_dim,
    depth,
    num_latents,
    latent_dim,
    self_per_cross_attn,
    input_length,
    batch_size,
    mask
):
    model = PerceiverBase(
        input_dim=input_dim,
        depth=depth,
        num_latents=num_latents,
        latent_dim=latent_dim,
        self_per_cross_attn=self_per_cross_attn,
    )

    x = torch.randn(batch_size, input_length, input_dim)

    if mask:
        mask = utils.random_mask(x)
    else:
        mask = None
    
    out = model(x, mask=mask)

    assert out.shape == (batch_size, num_latents, latent_dim)


@pytest.mark.parametrize('input_dim', [64])
@pytest.mark.parametrize('depth', [4])
@pytest.mark.parametrize('cross_heads', [None, 1, 4])
@pytest.mark.parametrize('cross_head_dim', [None, 32])
@pytest.mark.parametrize('latent_heads', [None, 1, 4])
@pytest.mark.parametrize('latent_head_dim', [None, 32])
def test_attn_heads(
    input_dim,
    depth,
    cross_heads,
    cross_head_dim,
    latent_heads,
    latent_head_dim,
):
    build_model = lambda: PerceiverBase(
        input_dim=input_dim,
        depth=depth,
        cross_heads=cross_heads,
        cross_head_dim=cross_head_dim,
        latent_heads=latent_heads,
        latent_head_dim=latent_head_dim
    )

    if (
        (cross_heads is None and cross_head_dim is None) or 
        (latent_heads is None and latent_head_dim is None)
    ):
        with pytest.raises(AssertionError):
            build_model()
    else:
        build_model()



@pytest.mark.parametrize('input_dim', [64])
@pytest.mark.parametrize('depth', [4])
@pytest.mark.parametrize('cross_attn_dropout', [0.0, 0.2])
@pytest.mark.parametrize('latent_attn_dropout', [0.0, 0.2])
def test_dropout(
    input_dim,
    depth,
    cross_attn_dropout,
    latent_attn_dropout
):
    model = PerceiverBase(
        input_dim=input_dim,
        depth=depth,
        cross_attn_dropout=cross_attn_dropout,
        latent_attn_dropout=latent_attn_dropout,
    )

    input = torch.randn(32, 128, input_dim)

    pass_a = model(input)
    pass_b = model(input)

    if cross_attn_dropout > 0.0 or latent_attn_dropout > 0.0:
        assert not torch.allclose(pass_a, pass_b)
    else:
        assert torch.allclose(pass_a, pass_b)


@pytest.mark.parametrize('num_latents', [None, 128])
def test_num_latents(num_latents):
    model = PerceiverBase(
        input_dim=64,
        depth=4,
        num_latents=num_latents,
        latent_dim=128,
    )

    data = torch.randn(32, 128, 64)

    if num_latents is None:
        with pytest.raises(AssertionError):
            model(data)
    else:
        model(data)


@pytest.mark.parametrize('latent_dim', [64, 128])
def test_custom_latents(latent_dim):
    model = PerceiverBase(
        input_dim=64,
        depth=4,
        num_latents=128,
        latent_dim=64,
    )

    latents = torch.randn(128, latent_dim)
    data = torch.randn(32, 128, 64)

    if latent_dim != model.latent_dim:
        with pytest.raises(AssertionError):
            model(data, latents=latents)
    else:
        out = model(data, latents=latents)
        assert out.shape == (32, 128, latent_dim)


def test_weight_tying():
    build_model = lambda weight_tie_layers: PerceiverBase(
        input_dim=64,
        depth=4,
        weight_tie_layers=weight_tie_layers,
    )

    weight_tied_model = build_model(True)
    not_weight_tied_model = build_model(False)

    assert utils.numel(not_weight_tied_model) > utils.numel(weight_tied_model)


@pytest.mark.parametrize('input_dim', [
    64,
    [64, 64],
    [64, 128, 256]
])
@pytest.mark.parametrize('use_mask', [False, True])
def test_multi_input(input_dim, use_mask):
    mask = None

    if isinstance(input_dim, int):
        data = torch.randn(32, 128, input_dim)
        depth = 1

        if use_mask:
            mask = utils.random_mask(data)
    else:
        data = [torch.randn(32, 128, dim) for dim in input_dim]
        depth = len(input_dim)

        if use_mask:
            mask = [utils.random_mask(x) for x in data]

    model = PerceiverBase(
        input_dim=input_dim,
        depth=depth
    )

    out = model(data, mask=mask)

    assert out.shape == (32, model.num_latents, model.latent_dim)


@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_flash_attn(use_flash_attn):
    model = PerceiverBase(
        input_dim=128,
        depth=4,
        use_flash_attn=use_flash_attn,
    )

    x = torch.randn(32, 64, 128)

    out = model(x)

    assert out.shape == (32, model.num_latents, model.latent_dim)


@pytest.mark.skip(reason='rotary positional embeddings are not yet supported for cross-attention')
def test_rotary_positional_embeddings():
    model = PerceiverBase(
        input_dim=128,
        depth=4,
        cross_rotary_emb_dim=32,
    )

    x = torch.randn(32, 64, 128)

    out = model(x)

    assert out.shape == (32, model.num_latents, model.latent_dim)
