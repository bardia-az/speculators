"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: ``torch.optim.Muon`` over the 2D weight matrices (which is all
Muon supports) and ``torch.optim.AdamW`` over everything else (norms, biases, and the
embedding / LM-head matrices, following standard Muon practice).

Muon works transparently for both single-GPU and multi-GPU (FSDP2) training: when the
model is sharded with ``fully_shard`` the parameters become ``DTensor``s and Muon's
Newton-Schulz orthogonalization dispatches across ranks automatically.
"""

import logging
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger("speculators")

# Names of parameters that are 2D but should still be optimized with AdamW rather than
# Muon, following the convention from Keller Jordan's Muon (embeddings, embedding-like
# codebooks, Markov vocabulary factors, and output heads are excluded from the
# orthogonalized update).
_ADAMW_NAME_HINTS = (
    "embed_tokens",
    "lm_head",
    "codebook",
    "markov_w1",
    "markov_w2",
)

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2


def model_lr_scales(model: Module) -> dict[str, float]:
    """Per-parameter LR multipliers a model opts into, empty for most models.

    A model that wants per-parameter learning rates (e.g. dnest, whose
    target-copied parameters can train at ``copied_lr_scale`` of the base LR)
    implements ``optimizer_param_lr_scales() -> dict[str, float]`` mapping
    state-dict parameter names to their multiplier. Entries equal to 1.0 are
    dropped so an all-defaults model is indistinguishable from one without
    the hook.
    """
    fn = getattr(model, "optimizer_param_lr_scales", None)
    if not callable(fn):
        return {}
    return {name: scale for name, scale in fn().items() if scale != 1.0}


def split_by_lr_scale(
    named_params: Iterable[tuple[str, Tensor]],
    lr_scales: dict[str, float],
) -> tuple[list[Tensor], dict[float, list[Tensor]]]:
    """Partition named parameters by LR scale: (base params, {scale: params}).

    Parameters absent from ``lr_scales`` (or with scale 1.0) form the base
    list; the rest group by their exact scale value. Names are dropped after
    partitioning -- optimizers only need the tensors.
    """
    base: list[Tensor] = []
    scaled: dict[float, list[Tensor]] = {}
    for name, param in named_params:
        scale = lr_scales.get(name)
        if scale is None:
            base.append(param)
        else:
            scaled.setdefault(float(scale), []).append(param)
    return base, scaled


def _log_group_sizes(base: int, scaled: dict[float, list], optimizer: str) -> None:
    parts = [f"{base} @ base lr"] + [
        f"{len(params)} @ {scale:.4g}x" for scale, params in sorted(scaled.items())
    ]
    logger.info("%s optimizer param groups: %s.", optimizer, ", ".join(parts))


def _scaled_param_groups(
    named_params: Iterable[tuple[str, Tensor]],
    base_lr: float,
    lr_scales: dict[str, float],
    optimizer: str,
) -> list[dict]:
    """Param groups partitioned by LR scale (see :func:`split_by_lr_scale`),
    with the group sizes logged."""
    base, scaled = split_by_lr_scale(named_params, lr_scales)
    groups: list[dict] = []
    if base:
        groups.append({"params": base, "lr": base_lr})
    for scale, params in sorted(scaled.items()):
        groups.append({"params": params, "lr": base_lr * scale})
    _log_group_sizes(len(base), scaled, optimizer)
    return groups


def split_named_params_for_muon(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding, codebook, or vocabulary-output weight;
    everything else goes to AdamW. Degenerate 2D weights (``[1, N]`` / ``[N, 1]``
    vectors) route to AdamW -- Muon orthogonalizes matrices, not vectors, and crashes
    on them under FSDP2.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params)`` tuple of named parameter lists.
    """
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == _MATRIX_NDIM
            and min(param.shape) > 1  # exclude degenerate [1, N] / [N, 1] vectors
            and not any(hint in name for hint in _ADAMW_NAME_HINTS)
        ):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """
    if config.optimizer == "adamw":
        lr_scales = model_lr_scales(model)
        if not lr_scales:
            return [
                torch.optim.AdamW(
                    model.named_parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            ]
        return [
            torch.optim.AdamW(
                _scaled_param_groups(
                    model.named_parameters(), config.lr, lr_scales, "AdamW"
                ),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        ]

    if config.optimizer == "muon":
        muon_params, adamw_params = split_named_params_for_muon(model)
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW.",
            len(muon_params),
            len(adamw_params),
        )
        lr_scales = model_lr_scales(model)

        def _groups(
            named: list[tuple[str, Tensor]], base_lr: float
        ) -> list[dict]:
            return _scaled_param_groups(named, base_lr, lr_scales, "Muon")

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(
                    _groups(muon_params, config.muon_lr),
                    lr=config.muon_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    _groups(adamw_params, config.lr),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
