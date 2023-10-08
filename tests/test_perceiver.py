import pytest
import torch

from flash_perceiver import Perceiver


@pytest.mark.parametrize('output_dim', [None, 128])
@pytest.mark.parametrize('output_mode', ['average', 'concat', 'first'])
@pytest.mark.parametrize('num_zero_tokens', [None, 0, 32])
@pytest.mark.parametrize('use_flash_attn', [False, True])
def test_output_projection(
    output_dim,
    output_mode,
    num_zero_tokens,
    use_flash_attn
):
    model = Perceiver(
        input_dim=128,
        depth=4,
        output_dim=output_dim,
        output_mode=output_mode,
        num_zero_tokens=num_zero_tokens,
        use_flash_attn=use_flash_attn
    )

    x = torch.randn(32, 64, 128)

    out = model(x)
    
    if output_dim is None:
        assert out.shape == (32, model.num_latents, model.latent_dim)
    else:
        assert out.shape == (32, output_dim)
