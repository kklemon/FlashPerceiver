import pytest
import torch

from flash_perceiver import PerceiverIO


@pytest.mark.parametrize('query_dim', [32, 64])
@pytest.mark.parametrize('num_queries', [8, 16])
@pytest.mark.parametrize('proj_dim', [None, 128])
def test_perceiver_io(
    query_dim,
    num_queries,
    proj_dim
):
    model = PerceiverIO(
        input_dim=128,
        depth=4,
        query_dim=query_dim,
        proj_dim=proj_dim
    )

    x = torch.randn(32, 64, 128)
    queries = torch.randn(num_queries, query_dim)

    out = model(x, queries=queries)
    
    if proj_dim is None:
        assert out.shape == (32, num_queries, query_dim)
    else:
        assert out.shape == (32, num_queries, proj_dim)