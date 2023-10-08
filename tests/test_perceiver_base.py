import pytest
import torch

from flash_perceiver import utils
from flash_perceiver.perceiver import PerceiverBase


@pytest.mark.parametrize('input_dim', [32, 64])
@pytest.mark.parametrize('depth', [1, 4])
@pytest.mark.parametrize('num_latents', [32])
@pytest.mark.parametrize('latent_dim', [128])
@pytest.mark.parametrize('self_per_cross_attn', [1, 2])
@pytest.mark.parametrize('input_length', [128, 256])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('mask', [False, True])
@pytest.mark.parametrize('use_flash_attn', [False, True])
@pytest.mark.parametrize('num_zero_tokens', [None, 0, 32])
@pytest.mark.parametrize('latent_drop', [0.0, 0.5])
def test_model(
    input_dim,
    depth,
    num_latents,
    latent_dim,
    self_per_cross_attn,
    input_length,
    batch_size,
    mask,
    use_flash_attn,
    num_zero_tokens,
    latent_drop
):
    model = PerceiverBase(
        input_dim=input_dim,
        depth=depth,
        num_latents=num_latents,
        latent_dim=latent_dim,
        self_per_cross_attn=self_per_cross_attn,
        use_flash_attn=use_flash_attn,
        num_zero_tokens=num_zero_tokens,
        latent_drop=latent_drop
    )
    model.train()

    x = torch.randn(batch_size, input_length, input_dim)

    if mask:
        mask = utils.random_mask(x)
    else:
        mask = None
    
    out = model(x, mask=mask)

    assert out.shape == (batch_size, int(num_latents * (1 - latent_drop)), latent_dim)


@pytest.mark.parametrize('input_dim', [64])
@pytest.mark.parametrize('depth', [4])
@pytest.mark.parametrize('cross_heads', [None, 1, 4])
@pytest.mark.parametrize('cross_head_dim', [None, 32])
@pytest.mark.parametrize('latent_heads', [None, 1, 4])
@pytest.mark.parametrize('latent_head_dim', [None, 32])
@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_attn_heads(
    input_dim,
    depth,
    cross_heads,
    cross_head_dim,
    latent_heads,
    latent_head_dim,
    use_flash_attn
):
    build_model = lambda: PerceiverBase(
        input_dim=input_dim,
        depth=depth,
        cross_heads=cross_heads,
        cross_head_dim=cross_head_dim,
        latent_heads=latent_heads,
        latent_head_dim=latent_head_dim,
        use_flash_attn=use_flash_attn
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
@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_dropout(
    input_dim,
    depth,
    cross_attn_dropout,
    latent_attn_dropout,
    use_flash_attn
):
    model = PerceiverBase(
        input_dim=input_dim,
        depth=depth,
        cross_attn_dropout=cross_attn_dropout,
        latent_attn_dropout=latent_attn_dropout,
        use_flash_attn=use_flash_attn
    )

    input = torch.randn(32, 128, input_dim)

    pass_a = model(input)
    pass_b = model(input)

    if cross_attn_dropout > 0.0 or latent_attn_dropout > 0.0:
        assert not torch.allclose(pass_a, pass_b)
    else:
        assert torch.allclose(pass_a, pass_b)


@pytest.mark.parametrize('num_latents', [None, 128])
@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_num_latents(num_latents, use_flash_attn):
    model = PerceiverBase(
        input_dim=64,
        depth=4,
        num_latents=num_latents,
        latent_dim=128,
        use_flash_attn=use_flash_attn
    )

    data = torch.randn(32, 128, 64)

    if num_latents is None:
        with pytest.raises(AssertionError):
            model(data)
    else:
        model(data)


@pytest.mark.parametrize('latent_dim', [64, 128])
@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_custom_latents(latent_dim, use_flash_attn):
    model = PerceiverBase(
        input_dim=64,
        depth=4,
        num_latents=128,
        latent_dim=64,
        use_flash_attn=use_flash_attn
    )

    latents = torch.randn(128, latent_dim)
    data = torch.randn(32, 128, 64)

    if latent_dim != model.latent_dim:
        with pytest.raises(AssertionError):
            model(data, latents=latents)
    else:
        out = model(data, latents=latents)
        assert out.shape == (32, 128, latent_dim)


@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_weight_tying(use_flash_attn):
    build_model = lambda weight_tie_layers: PerceiverBase(
        input_dim=64,
        depth=4,
        weight_tie_layers=weight_tie_layers,
        use_flash_attn=use_flash_attn
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
@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_multi_input(input_dim, use_mask, use_flash_attn):
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
        depth=depth,
        use_flash_attn=use_flash_attn
    )

    out = model(data, mask=mask)

    assert out.shape == (32, model.num_latents, model.latent_dim)


@pytest.mark.parametrize('use_flash_attn', [False, True])
@pytest.mark.parametrize('use_mask', [False, True])
def test_flash_attn(use_flash_attn, use_mask):
    model = PerceiverBase(
        input_dim=128,
        depth=4,
        use_flash_attn=use_flash_attn,
    )

    x = torch.randn(32, 64, 128)

    if use_mask:
        mask = utils.random_mask(x)
    else:
        mask = None

    out = model(x, mask=mask)

    assert out.shape == (32, model.num_latents, model.latent_dim)


@pytest.mark.parametrize('with_fa', [False, True])
@pytest.mark.parametrize('use_mask', [False, True])
def test_setting_fa(with_fa, use_mask):
    model = PerceiverBase(
        input_dim=128,
        depth=4,
        use_flash_attn=with_fa,
    )

    x = torch.randn(32, 64, 128)

    if use_mask:
        mask = utils.random_mask(x)
    else:
        mask = None

    out_before = model(x, mask=mask)

    model.set_flash_attn(not with_fa)

    out_after = model(x, mask=mask)

    assert out_before.shape == (32, model.num_latents, model.latent_dim)

    # Is this high tolerance reasonable?
    assert torch.allclose(out_before, out_after, atol=1e-3), \
        str(abs(out_before - out_after))


@pytest.mark.parametrize('return_attn_weights', [False, True])
@pytest.mark.parametrize('use_flash_attn', [False, True])
@pytest.mark.parametrize('use_mask', [False, True])
def test_return_attn_weights(return_attn_weights, use_flash_attn, use_mask):
    model = PerceiverBase(
        input_dim=128,
        depth=4,
        latent_dim=256,
        use_flash_attn=use_flash_attn,
    )

    x = torch.randn(32, 64, 128)

    if use_mask:
        mask = utils.random_mask(x)
    else:
        mask = None


    if return_attn_weights and use_flash_attn:
        with pytest.raises(NotImplementedError):
            out = model(x, return_attn_weights=return_attn_weights, mask=mask)
    else:
        if return_attn_weights:
            out, all_attn_weights = model(x, return_attn_weights=True, mask=mask)

            assert len(all_attn_weights) == model.num_attention_layers

            for i, attn_weights in enumerate(all_attn_weights):
                # Even layers are cross-attention, odd layers are self-attention
                if i % 2 == 0:
                    assert attn_weights.shape == (32, model.cross_heads, model.num_latents, x.shape[1])
                else:
                    assert attn_weights.shape == (32, model.latent_heads, model.num_latents, model.num_latents)
        else:
            out = model(x, return_attn_weights=False)
        
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
