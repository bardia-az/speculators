import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def _is_real_token(block_offset: torch.Tensor) -> torch.Tensor:
    """Which in-block offsets hold real (non-mask) tokens.

    Today only the anchor (offset 0) is real: block queries are
    ``[anchor, MASK x (block_size - 1)]``. The later multi-real phase --
    several leading block positions holding real tokens -- changes this
    predicate alone; the visibility rules below are already written in terms
    of it.
    """
    return block_offset == 0


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    attention_mode: str | None = None,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to tokens in their own synthetic block j, per the
          in-block visibility rule (``attention_mode`` when set, else the
          legacy ``sliding_window_non_causal`` rule)
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        sliding_window: integer size of sliding window or None for full attn
        sliding_window_non_causal: Use non causal mask for sliding window attn
            (legacy in-block rule; ignored when ``attention_mode`` is set)
        attention_mode: dflex in-block visibility override, one of
            "bidirectional" / "causal" / "hybrid" (see the nested
            ``_in_block_visible`` predicate). ``None`` keeps the legacy
            rule verbatim so DFlash/DSpark (which never set it) are
            unchanged.

    Returns:
        mask_mod, q_len, kv_len
    """
    if attention_mode is not None and attention_mode not in (
        "bidirectional",
        "causal",
        "hybrid",
    ):
        raise ValueError(
            f"attention_mode must be bidirectional|causal|hybrid, "
            f"got {attention_mode!r}"
        )
    # Always use non_causal for full attn
    non_causal = sliding_window is None or sliding_window_non_causal

    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # For each query position, which anchor does it belong to?
    # query q in [j*block_size, (j+1)*block_size) belongs to anchor_positions[j]
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may see base-sequence tokens in the same document before the anchor.
        """
        # absolute base position
        q_anchor = query_anchor_positions[q_idx]
        # doc id for this query block
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)  # safe indexing
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        in_window = (
            (kv_base_pos >= q_anchor - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def _in_block_visible(q_offset, kv_offset, q_is_real, kv_is_real):
        """
        Pure in-block visibility predicate (dflex ``attention_mode``).

        Both sides are within-block offsets; ``same_block_mod`` gates the
        result by same-block membership, so visibility towards the base
        prefix and other blocks is unchanged in every mode:

        - ``bidirectional``: every block position attends to every position
          in its block.
        - ``causal``: block position ``i`` attends to block positions
          ``j <= i``.
        - ``hybrid``: a real token attends only to real tokens at earlier or
          equal block positions; a mask attends to all real tokens in its
          block and to all masks in its block, in both directions.
        """
        causal = kv_offset <= q_offset
        if attention_mode == "causal":
            return causal
        if attention_mode == "hybrid":
            return torch.where(
                q_is_real, kv_is_real & causal, torch.ones_like(causal)
            )
        return torch.ones_like(causal)

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend to tokens in their own synthetic block, per the
        in-block visibility rule: ``attention_mode`` when set (dflex), else
        the legacy rule -- non-causal unless non_causal=False, in which case
        only prior positions are attended.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if attention_mode is not None:
            # Within-block offsets. kv_offset is only meaningful for
            # in-block keys; the remainder clamps out-of-range flex indices
            # (same safe-indexing pattern as base_prefix_mod) and `same`
            # gates the result.
            q_offset = torch.remainder(q_idx, block_size)
            kv_offset = torch.remainder(kv_idx - total_seq_len, block_size)
            same = same & _in_block_visible(
                q_offset,
                kv_offset,
                _is_real_token(q_offset),
                _is_real_token(kv_offset),
            )
        elif not non_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len
