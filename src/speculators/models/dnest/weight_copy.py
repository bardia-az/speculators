"""Target-weight initialisation for dnest drafters.

Kept as a standalone module (not inline in the model) because later phases
will grow this into per-layer context injection and frozen copied layers with
LoRA adapters; every entry point here is a pure function of its arguments so
the model class only orchestrates.
"""

import logging
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.utils.loading import load_model_layers

logger = logging.getLogger(__name__)

__all__ = [
    "COPIED_LAYER_SUFFIXES",
    "check_layer_compatibility",
    "copy_target_decoder_layers",
    "init_fusion_passthrough_deepest",
    "resolve_init_source_layers",
]

# The per-layer tensors copied from the target, as state-dict suffixes. The
# draft's Qwen3DFlashDecoderLayer and the target's Qwen3 decoder layer use
# identical submodule names, so only the prefix differs (target:
# ``model.layers.{s}.<suffix>`` -> draft: ``layers.{j}.<suffix>``).
COPIED_LAYER_SUFFIXES: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

# Config fields that must match exactly for a weight copy to be meaningful.
# Mismatches raise; they never silently reshape.
_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("hidden_size", "hidden size"),
    ("num_attention_heads", "number of attention heads"),
    ("num_key_value_heads", "number of KV heads"),
    ("intermediate_size", "MLP intermediate size"),
    ("rms_norm_eps", "RMSNorm eps"),
)

# RoPE fields compared via _rope_signature() rather than field-by-field,
# because transformers 5.x stores them under ``rope_parameters`` (with
# ``rope_theta`` as a legacy alias) depending on how the config was built.


def _resolved_head_dim(config: PretrainedConfig) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    return config.hidden_size // config.num_attention_heads


def _rope_signature(config: PretrainedConfig) -> tuple[Any, ...]:
    """A comparable signature of the RoPE settings (theta, type, partial factor)."""
    rope_params = getattr(config, "rope_parameters", None) or {}
    theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    rope_type = rope_params.get("rope_type", getattr(config, "rope_type", "default"))
    partial = getattr(config, "partial_rotary_factor", None)
    return (float(theta), str(rope_type), partial)


def _target_layer_type(config: PretrainedConfig, layer_idx: int) -> str:
    """The target's per-layer attention type ('full_attention' if undeclared)."""
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return "full_attention"
    return layer_types[layer_idx]


def check_layer_compatibility(
    draft_config: PretrainedConfig,
    target_config: PretrainedConfig,
    source_layers: list[int] | None = None,
) -> None:
    """Verify the draft layer configuration equals the target's (requirement:
    shape check), or raise a ``ValueError`` listing every mismatch.

    Only structural fields that a weight copy depends on are compared: hidden
    size, attention/KV head counts, head dim, MLP size, RMSNorm eps, attention
    bias, RoPE settings. q/k-norm presence is enforced structurally at copy
    time (a target without them lacks the tensors and the copy refuses). A
    per-layer attention-*type* difference (e.g. draft sliding vs target full)
    does not affect parameters, so it is logged as a warning instead.
    """
    mismatches: list[str] = []
    for field, label in _FIELD_LABELS:
        draft_value = getattr(draft_config, field, None)
        target_value = getattr(target_config, field, None)
        if draft_value != target_value:
            mismatches.append(f"{label}: draft={draft_value!r} target={target_value!r}")
    if _resolved_head_dim(draft_config) != _resolved_head_dim(target_config):
        mismatches.append(
            f"head dim: draft={_resolved_head_dim(draft_config)} "
            f"target={_resolved_head_dim(target_config)}"
        )
    if bool(getattr(draft_config, "attention_bias", False)) != bool(
        getattr(target_config, "attention_bias", False)
    ):
        mismatches.append(
            f"attention bias: draft={getattr(draft_config, 'attention_bias', None)!r} "
            f"target={getattr(target_config, 'attention_bias', None)!r}"
        )
    if _rope_signature(draft_config) != _rope_signature(target_config):
        mismatches.append(
            f"RoPE settings: draft={_rope_signature(draft_config)} "
            f"target={_rope_signature(target_config)}"
        )
    if mismatches:
        raise ValueError(
            "dnest cannot copy target layers: the draft layer configuration "
            "does not match the target's. No weights were copied. "
            "Mismatches: " + "; ".join(mismatches)
        )

    if source_layers:
        for draft_idx, source_idx in enumerate(source_layers):
            draft_type = getattr(draft_config, "layer_types", None)
            draft_type = (
                draft_type[draft_idx]
                if draft_type
                else "full_attention"
            )
            target_type = _target_layer_type(target_config, source_idx)
            if draft_type != target_type:
                logger.warning(
                    "dnest: draft layer %d uses %s attention but is "
                    "initialised from target layer %d which uses %s. Weight "
                    "shapes are unaffected; only the runtime attention mask "
                    "differs.",
                    draft_idx,
                    draft_type,
                    source_idx,
                    target_type,
                )


def _after_extracted_sources(
    aux_layer_ids: list[int],
    num_draft_layers: int,
    num_target_layers: int,
) -> list[int]:
    """Sources for ``target_after_extracted``: draft layer ``j`` pairs with
    the consumer of tap ``j`` -- 0-based decoder layer ``aux_layer_ids[j]``."""
    if len(aux_layer_ids) != num_draft_layers:
        raise ValueError(
            f"init_mode='target_after_extracted' pairs each draft layer "
            f"with the consumer of one extracted hidden state, so the "
            f"number of draft layers ({num_draft_layers}) must equal the "
            f"number of target_layer_ids ({len(aux_layer_ids)})."
        )
    # A tap with id == num_target_layers is the final post-norm state;
    # it has no consuming decoder layer to copy from.
    final_taps = [tap for tap in aux_layer_ids if tap >= num_target_layers]
    if final_taps:
        raise ValueError(
            "init_mode='target_after_extracted' cannot use target_layer_ids "
            f"{final_taps}: id {num_target_layers} is the verifier's final "
            "hidden state (post-norm), which no decoder layer consumes. "
            "Remove the final-layer tap or use another init mode."
        )
    return list(aux_layer_ids)


def resolve_init_source_layers(
    init_mode: str,
    aux_layer_ids: list[int],
    num_draft_layers: int,
    num_target_layers: int,
    init_source_layers: list[int] | None,
) -> list[int] | None:
    """Resolve the target source layers, or ``None`` for random mode.

    ``aux_layer_ids`` are indices into the verifier's ``hidden_states`` tuple
    (slot 0 = embedding output, slot ``i`` = output of 0-based decoder layer
    ``i - 1``), so the layer that *consumes* the tap with id ``l`` is 0-based
    decoder layer ``l``. Returned indices are 0-based decoder layer numbers.

    :raises ValueError: on invalid indices or a draft/source count mismatch.
    """
    if init_mode == "random":
        return None
    if init_mode == "target_after_extracted":
        sources = _after_extracted_sources(
            aux_layer_ids, num_draft_layers, num_target_layers
        )
    elif init_mode == "target_last":
        if num_draft_layers > num_target_layers:
            raise ValueError(
                f"init_mode='target_last' needs the target's last "
                f"{num_draft_layers} layers but the target only has "
                f"{num_target_layers}."
            )
        sources = list(
            range(num_target_layers - num_draft_layers, num_target_layers)
        )
    elif init_mode == "target_explicit":
        if not init_source_layers:
            raise ValueError(
                "init_mode='target_explicit' requires init_source_layers."
            )
        sources = list(init_source_layers)
    else:
        raise ValueError(f"Unknown dnest init_mode: {init_mode!r}")

    invalid = sorted(
        s for s in sources if s < 0 or s >= num_target_layers
    )
    if invalid:
        raise ValueError(
            f"init source layers {invalid} are not valid target layers for a "
            f"target with {num_target_layers} hidden layers (0-based)."
        )
    if len(set(sources)) != len(sources):
        raise ValueError(f"init source layers must be distinct, got {sources}.")
    if len(sources) != num_draft_layers:
        raise ValueError(
            f"The number of init source layers ({len(sources)}) must equal "
            f"the number of draft layers ({num_draft_layers})."
        )
    return sources


def copy_target_decoder_layers(
    model: nn.Module,
    verifier_path: str,
    source_layers: list[int],
    *,
    init_final_norm: bool = False,
) -> list[str]:
    """Initialise the draft decoder layers from target layers, all-or-nothing.

    Reads only the needed tensors from the target's safetensors shards (via
    the index file, on CPU), validates that every tensor exists and every
    shape matches, and only then applies the copies. The fusion projection,
    embedding, and LM head are never touched here (the embedding/head follow
    DFlash's verifier sharing; the fusion follows ``fusion_init``).

    :return: The state-dict names of the parameters that were copied.
    :raises ValueError: if any tensor is missing or shaped differently; in
        that case nothing is copied.
    """
    name_pairs: list[tuple[str, str]] = []  # (target key, draft param name)
    for draft_idx, source_idx in enumerate(source_layers):
        for suffix in COPIED_LAYER_SUFFIXES:
            name_pairs.append(
                (
                    f"model.layers.{source_idx}.{suffix}",
                    f"layers.{draft_idx}.{suffix}",
                )
            )
    if init_final_norm:
        name_pairs.append(("model.norm.weight", "norm.weight"))

    # load_model_layers reads only the shards holding these tensors, on CPU,
    # via model.safetensors.index.json; missing names are omitted with a
    # warning, which we turn into a hard error below.
    target_keys = [target_key for target_key, _ in name_pairs]
    tensors = load_model_layers(target_keys, verifier_path)
    missing = [key for key in target_keys if key not in tensors]
    if missing:
        raise ValueError(
            f"dnest cannot copy target layers: {len(missing)} of the "
            f"requested target tensors are absent from {verifier_path} "
            f"(e.g. {missing[:5]}). This usually means the target lacks "
            "q/k-norm or a component the draft expects. No weights were "
            "copied."
        )

    shape_errors = []
    for target_key, draft_name in name_pairs:
        param = model.get_parameter(draft_name)
        if tensors[target_key].shape != param.shape:
            shape_errors.append(
                f"{draft_name} {tuple(param.shape)} != "
                f"{target_key} {tuple(tensors[target_key].shape)}"
            )
    if shape_errors:
        raise ValueError(
            "dnest cannot copy target layers: tensor shapes differ "
            "(draft param != target tensor): " + "; ".join(shape_errors[:5])
            + ". No weights were copied."
        )

    for target_key, draft_name in name_pairs:
        param = model.get_parameter(draft_name)
        with torch.no_grad():
            param.copy_(
                tensors[target_key].to(device=param.device, dtype=param.dtype)
            )
    logger.info(
        "dnest: initialised draft layers %s from target layers %s "
        "(%d tensors%s).",
        list(range(len(source_layers))),
        source_layers,
        len(name_pairs),
        " + final norm" if init_final_norm else "",
    )
    return [draft_name for _, draft_name in name_pairs]


def init_fusion_passthrough_deepest(
    fc: nn.Linear, aux_layer_ids: list[int]
) -> int:
    """Make the fusion projection initially return the deepest extracted state.

    ``fc`` maps the concatenation of the extracted hidden states (in
    ``aux_layer_ids`` order) to the fused feature. This sets its weight to the
    identity on the deepest tap's slice and zero elsewhere, so the linear
    output equals that tap exactly. The ``hidden_norm`` that follows divides
    by the per-token RMS -- an exact passthrough of the raw scale cannot be
    expressed through a fixed-weight RMSNorm, so the closest equivalent is
    ``x_deepest / rms(x_deepest)`` (same direction, unit RMS). This is the
    documented behaviour for ``fusion_init='passthrough_deepest'``.

    :return: The slice index of the deepest tap.
    """
    if not aux_layer_ids:
        raise ValueError("aux_layer_ids must be non-empty for fusion init.")
    slice_idx = aux_layer_ids.index(max(aux_layer_ids))
    hidden_size = fc.weight.shape[0]
    expected_in = len(aux_layer_ids) * hidden_size
    if fc.weight.shape[1] != expected_in:
        raise ValueError(
            f"fusion fc in-features {fc.weight.shape[1]} does not match "
            f"{len(aux_layer_ids)} taps x {hidden_size} hidden."
        )
    with torch.no_grad():
        fc.weight.zero_()
        fc.weight[
            :, slice_idx * hidden_size : (slice_idx + 1) * hidden_size
        ].copy_(
            torch.eye(
                hidden_size, dtype=fc.weight.dtype, device=fc.weight.device
            )
        )
    return slice_idx
