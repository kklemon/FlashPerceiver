from typing import List, Optional
import torch

from functools import partial
from torch import nn

from einops import repeat
from einops.layers.torch import Reduce

from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.block import Block
from flash_attn.modules.mlp import Mlp, GatedMlp

from fast_perceiver.utils import cache_fn


def patched_mha(base_mha_cls):
    class PatchedMHA(base_mha_cls):
        """
        Wrapper around FA's MHA to support separate q and kv dim and more API flexibility.
        """
        def __init__(
            self,
            embed_dim: int,
            *args,
            kv_dim: Optional[int] = None,
            num_heads: Optional[int] = 8,
            head_dim: Optional[int] = None,
            **kwargs
        ):
            if num_heads is None:
                assert head_dim is not None, 'Must specify either num_heads or head_dim'
                num_heads = embed_dim // head_dim

            super().__init__(embed_dim, num_heads, *args, **kwargs)

            self.kv_dim = kv_dim or self.embed_dim

            if head_dim is not None:
                self.head_dim = head_dim
            
            inner_dim = self.num_heads * self.head_dim
            linear_cls = self.out_proj.__class__

            qkv_proj_bias = kwargs.get('qkv_proj_bias', True)
            out_proj_bias = kwargs.get('out_proj_bias', True)

            if self.cross_attn:
                self.Wq = linear_cls(self.embed_dim, inner_dim, bias=qkv_proj_bias)
                self.Wkv = linear_cls(self.kv_dim, 2 * inner_dim, bias=qkv_proj_bias)
            else:
                self.Wqkv = linear_cls(self.embed_dim, 3 * inner_dim, bias=qkv_proj_bias)

            self.out_proj = linear_cls(inner_dim, self.embed_dim, bias=out_proj_bias)

    return PatchedMHA

PatchedMHA = patched_mha(MHA)
PatchedParallelMHA = patched_mha(ParallelMHA)

T = torch.Tensor


class Perceiver(nn.Module):
    """
    Fast and memory efficient [Perceiver](https://arxiv.org/abs/2103.03206) implementation in PyTorch
    with [FlashAttention](https://arxiv.org/abs/2205.14135) as the underlying attention implementation.

    Args:
        input_dim: Dimension (last axis) of input
        depth: Number of cross-self-attention blocks. One such block corresponds to
            a cross-attention module followed by `self_per_cross_attn` self-attention modules.
            The number of overall attention modules is therefore `depth * (1 + self_per_cross_attn)`.
        out_dim: Dimension of output. If None, no output projection is applied
            and the final latents are returned.
        num_latents: Number of latent vectors.
        latent_dim: Dimension of latent vectors.
        cross_heads: Number of heads for cross-attention. Defaults to 1.
        cross_head_dim: Dimension of cross-attention heads.
        cross_rotary_emb_dim: Dimension of cross-attention rotary embeddings.
            Defaults to 0 (no rotary embeddings).
        cross_attn_dropout: Dropout for cross-attention.
        latent_heads: Number of heads for latent self-attention. Defaults to 8.
        latent_head_dim: Dimension of latent self-attention heads.
        latent_rotary_emb_dim: Dimension of latent self-attention rotary embeddings.
            Defaults to 0 (no rotary embeddings).
        latent_attn_dropout: Dropout for latent self-attention.
        weight_tie_layers: Whether to share the weights of the cross-attention and
            latent self-attention blocks. Defaults to False.
        gated_mlp: Whether to use gated MLPs. Doubles the number of parameters
            in those layers. Defaults to True.
        self_per_cross_attn: Number of self-attention blocks per cross-attention block.
            Defaults to 1.
    """
    def __init__(
        self,
        *,
        input_dim: List[int] | int,
        depth: int,
        output_dim: Optional[int] = None,
        num_latents: Optional[int] = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        cross_head_dim: int = 64,
        cross_rotary_emb_dim: int = 0,
        cross_attn_dropout: float = 0.0,
        latent_heads: int = 8,
        latent_head_dim: int = 64,
        latent_rotary_emb_dim: int = 0,
        latent_attn_dropout: float = 0.0,
        weight_tie_layers: bool = False,
        gated_mlp: bool = True,
        use_parallel_mha: bool = False,
        self_per_cross_attn: int = 1,
    ):
        super().__init__()
        
        if isinstance(input_dim, (tuple, list)):
            assert len(input_dim) == depth, 'Must specify input_dim for each layer'
            assert not weight_tie_layers, 'Cannot weight tie layers with different input dimensions'

            self.input_dims = input_dim
        else:
            self.input_dims = [input_dim] * depth

        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.depth = depth

        if self.num_latents is not None:
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        else:
            self.latents = None

        if gated_mlp:
            mlp_cls = partial(GatedMlp, hidden_features=latent_dim * 4)
        else:
            mlp_cls = Mlp

        if use_parallel_mha:
            mha_cls = PatchedParallelMHA
        else:
            mha_cls = PatchedMHA

        get_cross_attn_block = lambda in_dim: Block(
            dim=latent_dim,
            mixer_cls=partial(
                mha_cls,
                kv_dim=in_dim,
                num_heads=cross_heads,
                head_dim=cross_head_dim,
                cross_attn=True,
                dropout=cross_attn_dropout,
                qkv_proj_bias=False,
                rotary_emb_dim=cross_rotary_emb_dim,
                use_flash_attn=True,
            ),
            mlp_cls=mlp_cls
        )

        get_self_attn_block = lambda: Block(
            dim=latent_dim,
            mixer_cls=partial(
                mha_cls,
                num_heads=latent_heads,
                head_dim=latent_head_dim,
                dropout=latent_attn_dropout,
                rotary_emb_dim=latent_rotary_emb_dim,
                use_flash_attn=True
            ),
            mlp_cls=mlp_cls
        )

        get_cross_attn_block, get_self_attn_block = map(cache_fn, (get_cross_attn_block, get_self_attn_block))

        self.layers = nn.ModuleList([])

        for i, in_dim in enumerate(self.input_dims):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_idx in range(self_per_cross_attn):
                self_attns.append(get_self_attn_block(**cache_args, key=block_idx))

            self.layers.append(nn.ModuleList([
                get_cross_attn_block(in_dim=in_dim, **cache_args),
                self_attns
            ]))

        if output_dim is not None:
            self.out_proj = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, output_dim)
            )
        else:
            self.out_proj = nn.Identity()


    def _validate_data(self, data: List[T] | T, mask: List[T] | T | None = None):
        if isinstance(data, T):
            data = [data] * self.depth
            mask = [mask] * self.depth
        else:
            assert len(data) == self.depth, f'Expected {self.depth} inputs, but found {len(data)}'

            if mask is not None:
                assert isinstance(mask, (tuple, list)), \
                    'If a list of data tensors is provided, mask must have the same format'
                assert len(mask) == self.depth, f'Expected {self.depth} masks, but found {len(mask)}'
            else:
                mask = [None] * self.depth
        
        assert all(d.shape[-1] == in_dim for d, in_dim in zip(data, self.input_dims)), \
            'Data dimensions do not match cross-attention dimensions'
        
        assert len(set(d.shape[0] for d in data)) == 1, 'All data tensors must have the same batch size'

        return data, mask

    def forward(
        self,
        data: List[T] | T,
        mask: List[T] | T | None = None,
        latents: T | None = None,
        return_embeddings: bool = False
    ):
        is_multi_data = isinstance(data, (tuple, list))

        data, masks = self._validate_data(data, mask)

        batch_size = data[0].shape[0]

        if latents is None:
            assert self.latents is not None, 'Must explicitly provide latents if not initialized with num_latents'
            latents = self.latents
        else:
            assert latents.shape[-1] == self.latent_dim, f'Latents must have {self.latent_dim} dimensions, but found {latents.shape[-1]}'

        if latents.ndim == 2:
            latents = repeat(latents, 'n d -> b n d', b=batch_size)
        
        x = latents

        cross_block_kwargs = None

        for (cross_block, self_attn_blocks), datum, mask in zip(self.layers, data, masks):
            if is_multi_data or cross_block_kwargs is None:
                cross_block_kwargs = {'x_kv': datum}

                if mask is not None:
                    datum, _, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(datum, mask)

                    cross_block_kwargs = {
                        'x_kv': datum,
                        'cu_seqlens_k': cu_seqlens_k,
                        'max_seqlen_k': max_seqlen_in_batch_k
                    }

            # FlashAttention currently does not support key-value-only padding
            # We therefore have to _unpad_ the queries (aka latents) as well.
            # In the future, this could be used for a Perceiver AR implementation.
            # TODO: We could compute the dummy mask tensors for the queries directly here
            #  without calling the unpad_input function.
            if mask is not None:
                x_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                x_cross, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, x_mask)

                cross_block_kwargs.update({
                    'cu_seqlens': cu_seqlens,
                    'max_seqlen': max_seqlen_in_batch
                })
            else:
                x_cross = x

            x = cross_block(x_cross, mixer_kwargs=cross_block_kwargs)[0]

            if mask is not None:
                x = pad_input(x, indices, batch_size, self.num_latents)

            for self_attn_block in self_attn_blocks:
                x = self_attn_block(x)[0]

        if return_embeddings:
            return x

        return self.out_proj(x)
