"""Metrics and loss functions for DFlash draft model."""

from functools import partial
from typing import Any

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    kl_div_loss,
)
from speculators.losses.tau_rb import tau_rb_block_loss
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
    token_ids: torch.Tensor | None = None,
    target_temperature: float = 1.0,
    t2d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_config: Mapping of ``{name: (loss_fn, weight)}``; a ``tau_rb``
            entry is computed with block context instead of the per-position
            dispatch
        per_position_loss_weight: Weighting option for per-position block-drafting loss
        dpace_alpha: Smoothing constant for D-Pace loss weighting
        token_ids: Realized training tokens at each slot (verifier vocab),
            required by the ``tau_rb`` loss
        target_temperature: Temperature forming ``p`` for the ``tau_rb`` loss
        t2d: Target-to-draft vocab mapping for the ``tau_rb`` loss, or None
            when the draft uses the full vocabulary

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
            - eal: Expected Accepted Length, the mean per-block accepted run
              plus the verifier's bonus token (headline metric)
            - tau_rb_loss / tau_rb_tau_hat / per-slot tau_rb_alpha_{i},
              tau_rb_accept_{i}, tau_rb_survival_{i}: estimator diagnostics
              when the loss config contains ``tau_rb``
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    # tau_rb needs the realized tokens and block structure, so it is computed
    # here rather than through the per-position (logits, targets) dispatch.
    tau_term: torch.Tensor | None = None
    tau_stats: dict[str, torch.Tensor] | None = None
    per_position_config: LossConfig = {}
    for name, spec in loss_config.items():
        if name == "tau_rb":
            if token_ids is None:
                raise ValueError(
                    "tau_rb requires the realized token ids aligned to the "
                    "draft slots (token_ids); the DFlash forward passes them "
                    "automatically."
                )
            tau_term, tau_stats = tau_rb_block_loss(
                logits,
                targets,
                token_ids,
                loss_mask,
                block_size,
                temperature=target_temperature,
                t2d=t2d,
                sample_from_anchor=sample_from_anchor,
            )
        else:
            per_position_config[name] = spec

    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=per_position_config,
        decay_fn=decay_fn,
    )
    if tau_term is not None:
        _fn, weight = loss_config["tau_rb"]
        loss = loss + weight * tau_term

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Start position: 0 if sample_from_anchor else 1 (skip anchor)
    start_pos = 0 if sample_from_anchor else 1
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()

    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    # Counted per block so the accepted run is formed before any averaging; the
    # sum/total pair then pools across batches and ranks like every other metric.
    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(-1, block_size)[:, start_pos:],
        loss_mask.to(torch.bool).reshape(-1, block_size)[:, start_pos:],
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total

    if tau_term is not None and tau_stats is not None:
        block_valid = tau_stats["block_valid"]
        metrics["tau_rb_loss_sum"] = tau_term.detach()
        metrics["tau_rb_loss_total"] = ones.clone()
        metrics["tau_rb_tau_hat_sum"] = tau_stats["tau_hat"][block_valid].sum()
        metrics["tau_rb_tau_hat_total"] = block_valid.sum()
        slot_valid = tau_stats["slot_valid"]
        for k in range(slot_valid.shape[1]):
            m = slot_valid[:, k]
            slot = start_pos + k
            alpha_k = tau_stats["alpha"][:, k][m]
            accept_k = tau_stats["accept"][:, k][m]
            survival_k = tau_stats["survival"][:, k][m]
            metrics[f"tau_rb_alpha_{slot}_sum"] = alpha_k.sum()
            metrics[f"tau_rb_accept_{slot}_sum"] = accept_k.sum()
            metrics[f"tau_rb_survival_{slot}_sum"] = survival_k.sum()
            metrics[f"tau_rb_alpha_{slot}_total"] = m.sum()
            metrics[f"tau_rb_accept_{slot}_total"] = m.sum()
            metrics[f"tau_rb_survival_{slot}_total"] = m.sum()
    return loss, metrics
