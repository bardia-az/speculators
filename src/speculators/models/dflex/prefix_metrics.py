"""Losses and counted metrics for the extra DFlex rows."""

import torch
import torch.distributed as dist

from speculators.losses import dflash_loss_decay
from speculators.losses.utils import _LOSS_REDUCTION_EPS
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)


def prefix_metrics(
    config, layout, logits, targets, valid, loss_config, gamma, training
):
    k = config.block_size
    device = valid.device
    one = torch.ones((), device=device)
    zero = torch.zeros((), device=device)
    stats = {}

    def record(name, numerator, denominator=one):
        stats[name + "_sum"] = numerator.detach().float()
        stats[name + "_total"] = denominator.detach().float()

    record("num_full_anchors", one * layout.num_full)
    record("num_partial_anchors", one * sum(layout.histogram))
    for length in range(2, k + 1):
        record(f"partial_length_{length}", one * layout.histogram[length])

    # Only one scalar per row leaves a vocabulary kernel. Keep the no-grad
    # overlap calculation chunked so FP32 q and p never span the whole batch.
    ar_sums = zero
    draft_sums = zero
    overlap_sum = zero
    depth = layout.offset - layout.real_lengths[layout.block] + 1
    draft_slot = layout.is_partial[layout.block] & (depth >= 0)
    ar_mask = valid * ~draft_slot
    draft_mask = valid * draft_slot
    if layout.offset.numel():
        pred, teacher = logits.argmax(-1), targets.argmax(-1)
        decay = dflash_loss_decay(
            depth.clamp_min(0).float(), gamma, sample_from_anchor=True
        )
        for _, (fn, weight) in loss_config.items():
            values = fn(logits, targets).float()
            ar_sums = ar_sums + weight * (values * ar_mask).sum()
            draft_sums = draft_sums + weight * (values * draft_mask * decay).sum()
        with torch.no_grad():
            for start in range(0, logits.shape[1], 128):
                cols = slice(start, start + 128)
                q = logits[:, cols].float().softmax(-1)
                p = targets[:, cols].float().softmax(-1)
                overlap_sum = (
                    overlap_sum + (torch.minimum(p, q).sum(-1) * ar_mask[:, cols]).sum()
                )
    else:
        pred = teacher = torch.empty_like(valid, dtype=torch.long)

    ar_count, draft_count = ar_mask.float().sum(), draft_mask.float().sum()
    # DDP averages gradients. Use pooled token counts to weight each rank's
    # contribution when packed documents yield different numbers of labels.
    counts = torch.stack((ar_count, draft_count))
    ranks = 1
    if training and dist.is_initialized():
        dist.all_reduce(counts)
        ranks = dist.get_world_size()
    ar_loss, draft_loss = (
        ranks * torch.stack((ar_sums, draft_sums)) / (counts + _LOSS_REDUCTION_EPS)
    )
    record("ar_loss", ar_sums, ar_count)
    record("draft_partial_loss", draft_sums, draft_count)
    record("ar_overlap", overlap_sum, ar_count)
    correct, total = compute_accuracy_multi_step(
        pred, teacher, ar_mask, torch.zeros_like(depth), 1
    )
    record("ar_accuracy", correct[0], total[0])
    correct, total = compute_accuracy_multi_step(
        pred, teacher, draft_mask, depth.clamp(0, k - 1), k
    )
    record("partial_accuracy", correct.sum(), total.sum())
    for i in range(k):
        record(f"partial_depth_{i + 1}_accuracy", correct[i], total[i])

    rows = layout.draft_rows
    matches = (pred == teacher)[0, rows]
    supervised = valid[0, rows].bool()
    record("partial_eal", *compute_accepted_length_counts(matches, supervised))
    lengths = layout.real_lengths[layout.block[rows[:, 0]]]
    for low, high in ((2, 4), (5, 8), (9, k)):
        if low <= k:
            bucket = (lengths >= low) & (lengths <= high)
            record(
                f"partial_eal_l{low}_{min(high, k)}",
                *compute_accepted_length_counts(matches, supervised & bucket[:, None]),
            )
    loss = config.lambda_ar * ar_loss + draft_loss
    record("prefix_loss", loss)
    return loss, stats
