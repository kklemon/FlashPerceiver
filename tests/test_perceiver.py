import pytest
import torch

from fast_perceiver import Perceiver


@pytest.mark.parametrize('output_dim', [None, 128])
def test_output_projection(output_dim):
    model = Perceiver(
        input_dim=128,
        depth=4,
        output_dim=output_dim,
    )

    x = torch.randn(32, 64, 128)

    out = model(x)
    
    if output_dim is None:
        assert out.shape == (32, model.num_latents, model.latent_dim)
    else:
        assert out.shape == (32, output_dim)

