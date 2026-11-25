"""Per-layer target context modules for dnest (``context_mode="per_layer"``).

Each draft layer ``j`` is a copy of target layer ``s_j`` and conditions on
``h_j`` — the extracted target hidden state that is the *input* of ``s_j``
(hidden-states-tuple tap ``s_j``; the consumer of tap ``l`` is 0-based layer
``l``, so the input of layer ``s_j`` is exactly tap ``s_j``). The context
keys/values are then computed exactly the way the target layer computes its
own: ``input_layernorm`` → ``k_proj``/``v_proj`` → ``k_norm`` (keys) → RoPE
at the context tokens' absolute positions. No target KV cache is needed.

``kv_mode`` selects who owns the key/value path:

- ``frozen_shared`` / ``trainable_shared`` — the layer's own
  ``input_layernorm`` + ``k_proj``/``v_proj``/``k_norm`` play both roles
  (context and block). No new modules; the layer simply normalises ``h_j``
  with its ``input_layernorm`` before it reaches the attention.
- ``split`` — a second, frozen copy of the four path tensors lives on the
  attention as ``self_attn.context_kv``; the block keeps the layer's own
  (trainable) path.
- ``split_trainable`` — the same second copy, but trainable: it starts as
  the source layer's exact path (context keys/values equal the target's at
  initialisation) and may then drift during training.
"""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RMSNorm

from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
)

__all__ = [
    "CONTEXT_KV_SUFFIXES",
    "ContextKVPath",
    "DNestAttention",
    "DNestDecoderLayer",
]

# State-dict suffixes of the context path in ``split``/``split_trainable``
# mode, relative to the attention module
# (``layers.{j}.self_attn.context_kv.<suffix>``). They copy 1:1 from the
# target layer (``model.layers.{s}.<suffix>``).
CONTEXT_KV_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "k_proj.weight",
    "v_proj.weight",
    "k_norm.weight",
)


class ContextKVPath(nn.Module):
    """The per-layer context key/value path (``split``/``split_trainable``).

    Mirrors the target layer's key/value computation: ``input_layernorm``
    feeding ``k_proj``/``v_proj``, and the per-head ``k_norm`` for keys. All
    four tensors are copied from the source target layer at fresh
    initialisation; ``split`` then freezes them (context keys/values exactly
    the target's), ``split_trainable`` keeps them trainable.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_norm = Qwen3RMSNorm(head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]


class DNestAttention(Qwen3DFlashAttention):
    """DFlash attention whose context K/V may come from a frozen context path.

    Identical to the base attention unless the owning layer registered a
    ``context_kv`` path (``split`` mode): then the context keys/values are
    projected by the frozen path and each part is normed with its own k_norm
    (elementwise RMSNorm, so equal to norming the concatenation with one).
    """

    def _context_kv(self, target_hidden):
        context_kv = getattr(self, "context_kv", None)
        if context_kv is None:
            return super()._context_kv(target_hidden)
        return context_kv.k_proj(target_hidden), context_kv.v_proj(target_hidden)

    def _norm_context_and_block_keys(self, k_ctx, k_noise, bsz, ctx_len, q_len):
        context_kv = getattr(self, "context_kv", None)
        if context_kv is None:
            return super()._norm_context_and_block_keys(
                k_ctx, k_noise, bsz, ctx_len, q_len
            )
        k_ctx = context_kv.k_norm(k_ctx.view(bsz, ctx_len, -1, self.head_dim))
        k_noise = self.k_norm(k_noise.view(bsz, q_len, -1, self.head_dim))
        return torch.cat([k_ctx, k_noise], dim=1)


class DNestDecoderLayer(Qwen3DFlashDecoderLayer):
    """Decoder layer normalising its per-layer context feature.

    In per-layer context modes the incoming ``target_hidden`` is the *raw*
    extracted hidden state ``h_j``; it must pass through the source layer's
    ``input_layernorm`` before the key/value projections, because that is the
    target's computation. Shared modes use the layer's own (single)
    ``input_layernorm``; ``split`` uses the frozen copy on
    ``self_attn.context_kv``.
    """

    attention_cls = DNestAttention

    def __init__(self, config: Qwen3Config, layer_idx: int, *, split: bool):
        super().__init__(config=config, layer_idx=layer_idx)
        if split:
            self.self_attn.context_kv = ContextKVPath(config)  # type: ignore[arg-type]

    def forward(self, target_hidden=None, hidden_states=None, **kwargs):
        assert hidden_states is not None  # noqa: S101
        if target_hidden is not None:
            context_kv = getattr(self.self_attn, "context_kv", None)
            context_norm = (
                context_kv.input_layernorm
                if context_kv is not None
                else self.input_layernorm
            )
            target_hidden = context_norm(target_hidden)
        return super().forward(
            target_hidden=target_hidden, hidden_states=hidden_states, **kwargs
        )
