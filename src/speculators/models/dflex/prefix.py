"""Revealed-prefix sampling and row metadata for DFlex training."""

import math
from dataclasses import dataclass

import torch


@dataclass
class PrefixLayout:
    anchors: torch.Tensor
    real_lengths: torch.Tensor
    is_partial: torch.Tensor
    block: torch.Tensor
    offset: torch.Tensor
    draft_rows: torch.Tensor
    histogram: list[int]
    num_full: int

    @classmethod
    def from_blocks(cls, anchors, lengths, partial, k, device):
        """Build integer indices on CPU, so ragged compaction never runs on Ascend."""
        anchors = torch.as_tensor(anchors, dtype=torch.long)
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        partial = torch.as_tensor(partial, dtype=torch.bool)
        widths = lengths + partial.long() * (k - 1)
        block = torch.repeat_interleave(torch.arange(len(anchors)), widths)
        starts = widths.cumsum(0) - widths
        offset = torch.arange(block.numel()) - starts[block]
        draft_starts = (starts + lengths - 1)[partial]
        draft_rows = draft_starts[:, None] + torch.arange(k)
        histogram = torch.bincount(lengths[partial], minlength=k + 1).tolist()
        return cls(
            anchors.to(device),
            lengths.to(device),
            partial.to(device),
            block.to(device),
            offset.to(device),
            draft_rows.to(device),
            histogram,
            int((~partial).sum()),
        )

    @property
    def indices(self):
        return self.anchors[self.block] + self.offset

    @property
    def real(self):
        return self.offset < self.real_lengths[self.block]

    def input_ids(self, ids, mask_token_id):
        return torch.where(self.real, ids[:, self.indices], mask_token_id)


def sample_prefix_layout(config, loss_mask, document_ids, seed):
    """Draw lengths first, then sample eligible anchors without replacement.

    Each partial anchor is unique across lengths. Long spans get first choice
    when a short pack cannot accommodate every request. Full and partial draws
    are independent and may overlap. Document IDs in a pack are contiguous runs.
    """
    generator = torch.Generator().manual_seed(seed)
    eligible = loss_mask[0].detach().bool().cpu()
    docs = document_ids[0].detach().cpu()
    k = config.block_size

    def draw(width, count, used):
        if not count or width >= len(docs):
            return torch.empty(0, dtype=torch.long)
        fits = (docs[:-width] == docs[width:]) & (docs[:-width] >= 0)
        candidates = (fits & eligible[:-width] & ~used[:-width]).nonzero().flatten()
        order = torch.randperm(len(candidates), generator=generator)
        return candidates[order[:count]].sort().values

    used = torch.zeros_like(eligible)
    full = draw(k, config.num_full_anchors, used)
    anchors = full.tolist()
    lengths = [k] * len(full)
    if config.num_partial_anchors:
        log_weights = torch.zeros(k - 1, dtype=torch.float64)
        if config.prefix_length_dist == "geometric":
            log_weights = torch.arange(k - 1, dtype=torch.float64) * math.log(
                config.prefix_length_ratio
            )
        sampled = (
            torch.multinomial(
                log_weights.softmax(0),
                config.num_partial_anchors,
                replacement=True,
                generator=generator,
            )
            + 2
        )
        counts = torch.bincount(sampled, minlength=k + 1).tolist()
        for length in range(k, 1, -1):
            chosen = draw(length + k - 1, counts[length], used)
            used[chosen] = True
            anchors.extend(chosen.tolist())
            lengths.extend([length] * len(chosen))
    partial = [False] * len(full) + [True] * (len(anchors) - len(full))
    return PrefixLayout.from_blocks(anchors, lengths, partial, k, loss_mask.device)
