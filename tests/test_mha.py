import pytest
import torch

from flash_perceiver.perceiver import PatchedMHA


@pytest.mark.parametrize('kv_dim', [None, 128])
@pytest.mark.parametrize('with_fa', [False, True])
def test_mha_fa_change(kv_dim, with_fa):
    embed_dim = 64

    mha = PatchedMHA(
        embed_dim=embed_dim,
        kv_dim=kv_dim,
        cross_attn=kv_dim is not None,
        use_flash_attn=with_fa
    )

    x = torch.randn(32, 128, embed_dim)

    x_kv = None

    if kv_dim is not None:
        x_kv = torch.randn(32, 256, kv_dim)

    out_before = mha(x, x_kv=x_kv)

    mha.set_flash_attn(not with_fa)

    out_after = mha(x, x_kv=x_kv)

    assert out_before.shape == (32, 128, embed_dim)

    # Is this high tolerance reasonable?
    assert torch.allclose(out_before, out_after, atol=5e-4)
