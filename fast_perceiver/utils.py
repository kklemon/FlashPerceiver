import torch
import torch.nn as nn

from functools import wraps


def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


def random_mask(x):
    bs, n = x.shape[:2]

    seq_lens = torch.randint(1, n + 1, (bs,), device=x.device)
    mask = torch.arange(n, device=x.device)[None, :] < seq_lens[:, None]

    return mask


def numel(m: nn.Module, only_trainable: bool = True):
    return sum(p.numel() for p in m.parameters() if not only_trainable or p.requires_grad)
