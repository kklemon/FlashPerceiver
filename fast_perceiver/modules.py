import torch

from functools import partial
from torch import nn

from einops import repeat
from einops.layers.torch import Reduce

from flash_attn.bert_padding import unpad_input
from flash_attn.modules.mha import MHA
from flash_attn.modules.block import Block
from flash_attn.modules.mlp import Mlp, GatedMlp

from fast_perceiver.utils import cache_fn


class BetterMHA(MHA):
    """
    Wrapper around FA's MHA to support separate q and kv dim and more API flexibility.
    """
    def __init__(self, embed_dim, *args, kv_dim=None, num_heads=8, head_dim=None, **kwargs):
        if num_heads is None:
            assert head_dim is not None, 'Must specify either num_heads or head_dim'
            kwargs['num_heads'] = embed_dim // head_dim

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


class Perceiver(nn.Module):
    """
    Fast and memory efficient [Perceiver](https://arxiv.org/abs/2103.03206) implementation in PyTorch
    with [FlashAttention](https://arxiv.org/abs/2205.14135) as the underlying attention implementation.

    Args:
        input_dim: Dimension (last axis) of input
        depth: Number of self-attention (latent processing) blocks.
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
        input_dim,
        depth,
        out_dim=None,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        cross_head_dim=64,
        cross_rotary_emb_dim=0,
        cross_attn_dropout=0.0,
        latent_heads=8,
        latent_head_dim=64,
        latent_rotary_emb_dim=0,
        latent_attn_dropout=0.0,
        weight_tie_layers=False,
        gated_mlp=True,
        self_per_cross_attn=1,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        if gated_mlp:
            mlp_cls = partial(GatedMlp, hidden_features=latent_dim * 4)
        else:
            mlp_cls = Mlp

        get_cross_attn_block = lambda: Block(
            dim=latent_dim,
            mixer_cls=partial(
                BetterMHA,
                kv_dim=input_dim,
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
                BetterMHA,
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

        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_idx in range(self_per_cross_attn):
                self_attns.append(get_self_attn_block(**cache_args, key=block_idx))

            self.layers.append(nn.ModuleList([
                get_cross_attn_block(**cache_args),
                self_attns
            ]))

        if out_dim is not None:
            self.out_proj = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, out_dim)
            )
        else:
            self.out_proj = nn.Identity()


    def forward(
        self,
        data,
        mask=None,
        return_embeddings=False
    ):
        batch_size, seq_len, dim = data.shape

        assert dim == self.input_dim, f'Input must have {self.input_dim} dimensions, but found {dim}'

        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        cross_block_kwargs = {'x_kv': data}

        if mask is not None:
            data, _, cu_seqlens, max_seqlen_in_batch = unpad_input(data, ~mask)

            cross_block_kwargs = {
                'x_kv': data,
                'cu_seqlens_k': cu_seqlens,
                'max_seqlen_k': max_seqlen_in_batch
            }

            print(data.shape)

        for cross_block, self_attn_blocks in self.layers:
            x = cross_block(x, mixer_kwargs=cross_block_kwargs)[0]

            for self_attn_block in self_attn_blocks:
                x = self_attn_block(x)[0]

        if return_embeddings:
            return x

        return self.out_proj(x)
