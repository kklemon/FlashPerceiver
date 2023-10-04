import math
from typing import List, Optional
import torch

from functools import partial
from torch import nn

from einops import repeat
from einops.layers.torch import Reduce

from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.modules.mha import (
    SelfAttention,
    FlashSelfAttention,
    CrossAttention,
    FlashCrossAttention,
    MHA,
    ParallelMHA
)
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

            # Missing attributes
            self.causal = kwargs.get('causal', False)
            self.dropout = kwargs.get('dropout', 0.0)
            self.softmax_scale = kwargs.get('softmax_scale', None)

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
        
        # def set_flash_attention(self, use_flash_attn: bool):
        #     """
        #     Enable or disbale use of FlashAttention.
        #     """
        #     inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        #     inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention

        #     kwargs = dict(
        #         causal=self.causal,
        #         softmax_scale=self.softmax_scale,
        #         attention_dropout=self.dropout
        #     )

        #     self.inner_attn = inner_attn_cls(**kwargs)
        #     self.inner_cross_attn = inner_cross_attn_cls(**kwargs)

        #     self.use_flash_attn = use_flash_attn
    
    return PatchedMHA


PatchedMHA = patched_mha(MHA)
PatchedParallelMHA = patched_mha(ParallelMHA)

T = torch.Tensor


class PerceiverBase(nn.Module):
    """
    Base class for FlashAttention-based implementations of Perceiver and PerceiverIO.

    Fast and memory efficient [Perceiver](https://arxiv.org/abs/2103.03206) implementation in PyTorch
    with [FlashAttention](https://arxiv.org/abs/2205.14135) as the underlying attention implementation.

    Args:
        input_dim: Number of features of the input data. Can be a single integer or a list of integers
            to specify different input dimensions for each cross-attention block.
            `len(input_dim)` must be equal to `depth` in that case.
        depth: Number of cross-self-attention blocks. One such block corresponds to
            a cross-attention module followed by `self_per_cross_attn` self-attention modules.
            The number of overall attention modules is therefore `depth * (1 + self_per_cross_attn)`.
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
        self_per_cross_attn: int = 1,
        use_parallel_mha: bool = False,
        use_flash_attn: bool = True,
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
        self.cross_heads = 1
        self.cross_head_dim = 64
        self.latent_heads = 8
        self.latent_head_dim = 64
        self.depth = depth
        self.self_per_cross_attn = self_per_cross_attn
        self.use_flash_attn = use_flash_attn

        if self.num_latents is not None:
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        else:
            self.latents = None

        if gated_mlp:
            self.mlp_cls = partial(GatedMlp, hidden_features=latent_dim * 4)
        else:
            self.mlp_cls = Mlp

        if use_parallel_mha:
            self.mha_cls = PatchedParallelMHA
        else:
            self.mha_cls = PatchedMHA

        get_cross_attn_block = lambda in_dim: Block(
            dim=latent_dim,
            mixer_cls=partial(
                self.mha_cls,
                kv_dim=in_dim,
                num_heads=cross_heads,
                head_dim=cross_head_dim,
                cross_attn=True,
                dropout=cross_attn_dropout,
                qkv_proj_bias=False,
                rotary_emb_dim=cross_rotary_emb_dim,
                use_flash_attn=use_flash_attn,
            ),
            mlp_cls=self.mlp_cls
        )

        get_self_attn_block = lambda: Block(
            dim=latent_dim,
            mixer_cls=partial(
                self.mha_cls,
                num_heads=latent_heads,
                head_dim=latent_head_dim,
                dropout=latent_attn_dropout,
                rotary_emb_dim=latent_rotary_emb_dim,
                use_flash_attn=use_flash_attn
            ),
            mlp_cls=self.mlp_cls
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


    def set_flash_attn(self, use_flash_attn: bool):
        """
        Enable or disbale use of FlashAttention.
        """
        for cross_block, self_attn_blocks in self.layers:
            cross_block.mixer.set_flash_attn(use_flash_attn)

            for self_attn_block in self_attn_blocks:
                self_attn_block.mixer.set_flash_attn(use_flash_attn)

        self.use_flash_attn = use_flash_attn

    @property
    def num_attention_layers_per_block(self):
        return 1 + self.self_per_cross_attn
    
    @property
    def num_attention_layers(self):
        return self.depth * self.num_attention_layers_per_block
    
    @property
    def num_self_attention_layers(self):
        return self.depth * self.self_per_cross_attn
    
    @property
    def num_cross_attention_layers(self):
        return self.depth
    
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
        return_attn_weights: bool = False
    ):
        """
        Args:
            data: Input data which interacts with the latents via cross-attention.
                Must have shape `(batch_size, num_tokens, input_dim)`.
                If a single tensor is provided, it will be used for all cross-attention layers.
                To provide a separate input for each cross-attention block, a list of tensors with length `depth`
                can be provided. The different inputs can have different lengths, optional masking per tensor,
                and differ in the feature dimension as configured via `input_dim` during model initialization.
            mask: Optional boolean mask for the input data of shape `(batch_size, num_tokens)`.
                `False` indicates that a given token should not be attended.
                Can be a single tensor or a list of tensors, depending on the format of the `data` argument.
                In the multi-input case, `None` masks for some of the inputs are allowed.
            latents: Optional custom latent vectors. Must be of shape `([batch_size,] num_latents, latent_dim)`.
                If not provided, the model's learned latent vectors will be used.
                If `None` has been provided as `num_latents` argument during model initialization, custom latents
                must be provided.
            return_attn_weights: Whether to return the attention weights of the attention modules.
        """
        if self.use_flash_attn and return_attn_weights:
            raise NotImplementedError(
                'FlashAttention does not support returning attention weights. '
                'Please disable use of FA with `set_flash_attention(False)`.'
            )

        is_multi_data = isinstance(data, (tuple, list))

        data, masks = self._validate_data(data, mask)

        batch_size = data[0].shape[0]

        if latents is None:
            assert self.latents is not None, \
                'Must explicitly provide latents if not initialized with num_latents'
            latents = self.latents
        else:
            assert latents.shape[-1] == self.latent_dim, \
                f'Latents must have {self.latent_dim} dimensions, but found {latents.shape[-1]}'

        if latents.ndim == 2:
            latents = repeat(latents, 'n d -> b n d', b=batch_size)
        
        x = latents

        mixer_kwargs = {}
        cross_block_mixer_kwargs = {}
        attn_weights = []

        def handle_output(args):
            if return_attn_weights:
                assert isinstance(args, tuple) and len(args) == 2
                out, attn_weight = args
                attn_weights.append(attn_weight)
                return out
            else:
                return args

        if return_attn_weights:
            mixer_kwargs['return_attn_weights'] = True

        for (cross_block, self_attn_blocks), datum, mask in zip(self.layers, data, masks):
            if is_multi_data or not cross_block_mixer_kwargs:
                cross_block_mixer_kwargs = {'x_kv': datum}

                if mask is not None:
                    datum, _, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(datum, mask)

                    cross_block_mixer_kwargs = {
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

                cross_block_mixer_kwargs.update({
                    'cu_seqlens': cu_seqlens,
                    'max_seqlen': max_seqlen_in_batch
                })
            else:
                x_cross = x

            x = handle_output(cross_block(
                x_cross,
                mixer_kwargs={**mixer_kwargs, **cross_block_mixer_kwargs}
            ))[0]

            if mask is not None:
                x = pad_input(x, indices, batch_size, self.num_latents)

            for self_attn_block in self_attn_blocks:
                x = handle_output(self_attn_block(x, mixer_kwargs=mixer_kwargs))[0]
        
        if return_attn_weights:
            return x, attn_weights
        
        return x


class Perceiver(PerceiverBase):
    """
    Implementation of the [Perceiver architecture](https://arxiv.org/abs/2103.03206) with compute and
    memory efficient [FlashAttention](https://arxiv.org/abs/2205.14135) as underlying attention implementation

    Args:
        input_dim: Number of features of the input data. Can be a single integer or a list of integers
            to specify different input dimensions for each cross-attention block.
            `len(input_dim)` must be equal to `depth` in that case.
        depth: Number of cross-self-attention blocks. One such block corresponds to
            a cross-attention module followed by `self_per_cross_attn` self-attention modules.
            The number of overall attention modules is therefore `depth * (1 + self_per_cross_attn)`.
        output_dim: Dimension of output. If `None`, no output projection is applied
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
        output_dim: int | None = None,
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
        self_per_cross_attn: int = 1,
        use_parallel_mha: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_head_dim=cross_head_dim,
            cross_rotary_emb_dim=cross_rotary_emb_dim,
            cross_attn_dropout=cross_attn_dropout,
            latent_heads=latent_heads,
            latent_head_dim=latent_head_dim,
            latent_rotary_emb_dim=latent_rotary_emb_dim,
            latent_attn_dropout=latent_attn_dropout,
            weight_tie_layers=weight_tie_layers,
            gated_mlp=gated_mlp,
            self_per_cross_attn=self_per_cross_attn,
            use_parallel_mha=use_parallel_mha,
            use_flash_attn=use_flash_attn
        )

        self.output_dim = output_dim

        if self.output_dim is not None:
            self.out_proj = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.output_dim)
            )
        else:
            self.out_proj = nn.Identity()
    
    def forward(
        self,
        data: List[T] | T,
        mask: List[T] | T | None = None,
        latents: T | None = None,
        return_embeddings: bool = False,
        return_attn_weights: bool = False
    ):
        """
        Args:
            data: Input data which interacts with the latents via cross-attention.
                Must have shape `(batch_size, num_tokens, input_dim)`.
                If a single tensor is provided, it will be used for all cross-attention layers.
                To provide a separate input for each cross-attention block, a list of tensors with length `depth`
                can be provided. The different inputs can have different lengths, optional masking per tensor,
                and differ in the feature dimension as configured via `input_dim` during model initialization.
            mask: Optional boolean mask for the input data of shape `(batch_size, num_tokens)`.
                `False` indicates that a given token should not be attended.
                Can be a single tensor or a list of tensors, depending on the format of the `data` argument.
                In the multi-input case, `None` masks for some of the inputs are allowed.
            latents: Optional custom latent vectors. Must be of shape `([batch_size,] num_latents, latent_dim)`.
                If not provided, the model's learned latent vectors will be used.
                If `None` has been provided as `num_latents` argument during model initialization, custom latents
                must be provided.
            return_embeddings: Whether to return the final latent vectors instead of the output projection.
            return_attn_weights: Whether to return the attention weights of the attention modules.
        """
        outputs = super().forward(data, mask, latents, return_attn_weights)

        if return_attn_weights:
            x, attn_weights = outputs
        else:
            x = outputs

        def make_output(x):
            if return_attn_weights:
                return x, attn_weights
            else:
                return x

        if not return_embeddings:
            x = self.out_proj(x)

        return make_output(x)


class PerceiverIO(PerceiverBase):
    """
    Implementation of the [PerceiverIO architecture](https://arxiv.org/abs/2103.03206) with compute and
    memory efficient [FlashAttention](https://arxiv.org/abs/2205.14135) as underlying attention implementation

    Args:
        input_dim: Number of features of the input data. Can be a single integer or a list of integers
            to specify different input dimensions for each cross-attention block.
            `len(input_dim)` must be equal to `depth` in that case.
        query_dim: Dimensionality of the query vectors for decoding from the latents.
        depth: Number of self-attention blocks. Following the original paper, there is only one
            cross-attention layer at the beginning followed by `depth` self-attention layers.
        proj_dim: If provided, the final output from the query attention operation will be projected
            to `output_dim`.
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
        query_heads: Number of heads for the latent-query cross-attention. Defaults to 1.
        query_head_dim: Dimension of the latent-query cross-attention heads.
        query_rotary_emb_dim: Dimension of the rotary embeddings for the latent-query cross-attention layer.
            Defaults to 0 (no rotary embeddings).
        query_attn_dropout: Dropout for the latent-query cross-attention.
        weight_tie_layers: Whether to share the weights of the cross-attention and
            latent self-attention blocks. Defaults to False.
        gated_mlp: Whether to use gated MLPs. Doubles the number of parameters
            in those layers. Defaults to True.
    """
    def __init__(
        self,
        *,
        input_dim: List[int] | int,
        query_dim: int,
        depth: int,
        proj_dim: int | None = None,
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
        query_heads: int = 1,
        query_head_dim: int = 64,
        query_rotary_emb_dim: int = 0,
        query_attn_dropout: float = 0.0,
        weight_tie_layers: bool = False,
        gated_mlp: bool = True,
        use_parallel_mha: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            depth=1,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_head_dim=cross_head_dim,
            cross_rotary_emb_dim=cross_rotary_emb_dim,
            cross_attn_dropout=cross_attn_dropout,
            latent_heads=latent_heads,
            latent_head_dim=latent_head_dim,
            latent_rotary_emb_dim=latent_rotary_emb_dim,
            latent_attn_dropout=latent_attn_dropout,
            weight_tie_layers=weight_tie_layers,
            gated_mlp=gated_mlp,
            self_per_cross_attn=depth,
            use_parallel_mha=use_parallel_mha,
            use_flash_attn=use_flash_attn
        )

        self.query_dim = query_dim
        self.proj_dim = proj_dim

        self.query_block = Block(
            dim=query_dim,
            mixer_cls=partial(
                self.mha_cls,
                kv_dim=self.latent_dim,
                num_heads=query_heads,
                head_dim=query_head_dim,
                cross_attn=True,
                dropout=query_attn_dropout,
                qkv_proj_bias=False,
                rotary_emb_dim=query_rotary_emb_dim,
                use_flash_attn=self.use_flash_attn,
            ),
            mlp_cls=self.mlp_cls
        )

        if self.proj_dim is not None:
            self.out_proj = nn.Sequential(
                nn.LayerNorm(self.query_dim),
                nn.Linear(self.query_dim, self.proj_dim)
            )
        else:
            self.out_proj = nn.Identity()
    
    def forward(
        self,
        data: List[T] | T,
        mask: List[T] | T | None = None,
        latents: T | None = None,
        queries: T | None = None,
        query_mask: T | None = None,
        return_attn_weights: bool = False
    ):
        """
        Args:
            data: Input data which interacts with the latents via cross-attention.
                Must have shape `(batch_size, num_tokens, input_dim)`.
                If a single tensor is provided, it will be used for all cross-attention layers.
                To provide a separate input for each cross-attention block, a list of tensors with length `depth`
                can be provided. The different inputs can have different lengths, optional masking per tensor,
                and differ in the feature dimension as configured via `input_dim` during model initialization.
            mask: Optional boolean mask for the input data of shape `(batch_size, num_tokens)`.
                `False` indicates that a given token should not be attended.
                Can be a single tensor or a list of tensors, depending on the format of the `data` argument.
                In the multi-input case, `None` masks for some of the inputs are allowed.
            latents: Optional custom latent vectors. Must be of shape `([batch_size,] num_latents, latent_dim)`.
                If not provided, the model's learned latent vectors will be used.
                If `None` has been provided as `num_latents` argument during model initialization, custom latents
                must be provided.
            queries: Optional query vectors which will interact with the latents via cross-attention to produce the output.
                Must have shape `(batch_size, num_queries, query_dim)`.
            query_mask: Not supported yet.
            return_attn_weights: Whether to return the attention weights of the attention modules.
        """
        outputs = super().forward(data, mask, latents, return_attn_weights)

        if return_attn_weights:
            embeds, attn_weights = outputs
        else:
            embeds = outputs

        def make_output(x):
            if return_attn_weights:
                return x, attn_weights
            return x

        if queries is None:
            return make_output(embeds)
        
        assert query_mask is None, \
            'query_mask is not supported yet'
        
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=embeds.shape[0])
        else:
            assert queries.ndim == 3

        out = self.query_block(queries, mixer_kwargs={
            'x_kv': embeds
        })[0]
        out = self.out_proj(out)

        return make_output(out)
