"""Exact expected accepted-length (``tau_rb``) loss for block drafters.

Phase 1 uses a single target rollout per context: the training sequence
itself. The realized tokens ``Z_k`` of that rollout supervise the survival
prefix; later phases can reduce gradient variance by stacking additional
continuations that share one drafter output.

For one block with drafted slots ``k = 1..K``:

    a_k     = min(1, q_k(Z_k) / p_k(Z_k))     sampled accept factor (log space)
    alpha_k = sum_v min(p_k(v), q_k(v))       exact per-slot acceptance (1 - TV)
    S_{j-1} = prod_{k<j} a_k                  survival, sampled factors only
    tau_hat = 1 + sum_j S_{j-1} * alpha_j
    loss    = -tau_hat / K

Only the LAST factor of each term uses ``alpha``; replacing just that factor
by its exact expectation keeps the estimator unbiased while preserving the
branch structure of the rollout. A product of per-slot alphas
(``prod_{k<=j} alpha_k``) is biased because whether an earlier slot is
accepted correlates with which continuation follows, and a plain sum of
alphas ignores survival entirely (that is the existing ``tv`` loss).

``p`` is computed at ``temperature`` and carries no gradient. ``Z`` must be
drawn from that same temperature for the estimator to be unbiased: greedy
(temperature-0) regenerated data corresponds to ``temperature -> 0``, where
``a_k = alpha_k = q_k(Z_k)`` and the objective degenerates into the greedy
chain loss ``1 + sum_j prod_{k<=j} q_k(Z_k)``.

Two implementations share the block math (this module's
``_tau_rb_from_row_stats``): ``tau_rb_block_loss`` computes the per-row
statistics (``alpha``, ``logq_z``, ``logp_z``) eagerly -- full-width
``[T, V]`` fp32 log-softmaxes, the numerical reference -- while
``tau_rb_fused_loss`` streams them from the Triton kernels in
``losses.fused`` without ever materializing a ``[T, V]`` intermediate. That
is the difference between OOMing and fitting at large anchor counts: the
eager graph retains ~9 full-width buffers, the fused one only the gradient.
"""

import torch

# fp32 log-space floor: exp(-80) ~ 1.8e-35, far below any meaningful
# probability, and safely above the -inf that breaks differences of logs.
_LOG_EPS = -80.0


def _validated_block_args(
    logits: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool,
    temperature: float,
) -> tuple[int, int, int]:
    """Shared shape/argument checks; returns (seq_len, start_pos, num_drafted)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if logits.dim() != 3 or logits.shape[0] != 1:
        raise ValueError(
            "logits must have shape [1, num_blocks*block_size, vocab], got "
            f"{tuple(logits.shape)}"
        )
    seq_len = logits.shape[1]
    if seq_len % block_size != 0:
        raise ValueError(
            f"sequence length {seq_len} is not a multiple of block_size {block_size}"
        )
    start_pos = 0 if sample_from_anchor else 1
    num_drafted = block_size - start_pos
    if num_drafted < 1:
        raise ValueError(
            f"block_size {block_size} leaves no drafted slot with "
            f"sample_from_anchor={sample_from_anchor}"
        )
    return seq_len, start_pos, num_drafted


def _realized_draft_indices(
    token_ids: torch.Tensor, t2d: torch.Tensor | None, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realized tokens as (draft-vocab index, in-vocab flag), each ``[T]``.

    Tokens outside the draft vocab have q = 0, so their accept factor is
    exactly 0: the block dies at that slot (-inf log factor -> zero survival
    downstream). Their index is clamped to 0 for the gather; the caller
    overwrites the gathered value via the in-vocab flag.
    """
    z = token_ids[0].long()
    if t2d is not None:
        z_map = t2d.to(device)[z]
        in_vocab = z_map > 0
        z_idx = (z_map - 1).clamp_min(0)
    else:
        in_vocab = torch.ones_like(z, dtype=torch.bool)
        z_idx = z
    return z_idx, in_vocab


def _tau_rb_from_row_stats(
    logq_z: torch.Tensor,  # [T] raw gathered draft log-prob of the realized token
    logp_z: torch.Tensor,  # [T] raw gathered target log-prob of the realized token
    alpha: torch.Tensor,  # [T] exact per-slot acceptance sum_v min(p, q)
    in_vocab: torch.Tensor,  # [T] bool
    loss_mask: torch.Tensor,  # [T]
    block_size: int,
    start_pos: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Block math shared by the eager and fused implementations.

    Everything here is ``[num_blocks, K]``-sized; the only full-vocab work
    (producing the three per-row statistics above) happens in the callers.
    """
    device = logq_z.device
    logq_z = logq_z.clamp_min(_LOG_EPS)
    logp_z = logp_z.clamp_min(_LOG_EPS)
    logq_z = torch.where(in_vocab, logq_z, torch.full_like(logq_z, float("-inf")))

    seq_len = logq_z.shape[0]
    num_blocks = seq_len // block_size

    def to_blocks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(num_blocks, block_size)[:, start_pos:]

    valid = to_blocks(loss_mask.to(torch.bool))
    # A block ends at its first invalid slot (padding, sequence end, or an
    # invalid anchor): that slot and all later slots contribute nothing.
    prefix_alive = torch.cumprod(valid.to(torch.float32), dim=1) > 0.0
    valid = valid & prefix_alive

    # log a_k = min(0, log q_k(Z_k) - log p_k(Z_k)); the clamp zeroes the
    # gradient wherever the ratio is capped at 1, which is intended.
    log_a = (to_blocks(logq_z) - to_blocks(logp_z)).clamp_max(0.0)
    log_a = torch.where(valid, log_a, torch.zeros_like(log_a))

    # S_{j-1} = exp(sum_{k<j} log a_k), with S_0 = 1: prepend a zero column to
    # the cumulative sum. Slots after a truncation or an out-of-vocab death
    # multiply survival by exactly 0 via exp(-inf) or the alive mask.
    cum = torch.cumsum(log_a, dim=1)
    log_survival = torch.cat(
        [torch.zeros(num_blocks, 1, device=device), cum[:, :-1]], dim=1
    )
    alive_before = torch.cat(
        [
            torch.ones(num_blocks, 1, device=device, dtype=torch.bool),
            prefix_alive[:, :-1],
        ],
        dim=1,
    )
    survival = torch.exp(log_survival) * alive_before.to(torch.float32)

    tau_hat = 1.0 + (survival * to_blocks(alpha) * valid).sum(dim=1)
    block_valid = valid.any(dim=1)
    denom = block_valid.to(torch.float32).sum().clamp_min(1.0)
    num_drafted = block_size - start_pos
    loss = -((tau_hat / num_drafted) * block_valid.to(torch.float32)).sum() / denom

    stats = {
        "tau_hat": tau_hat.detach(),
        "block_valid": block_valid,
        "slot_valid": valid,
        "alpha": to_blocks(alpha).detach(),
        "accept": log_a.exp().detach(),
        "survival": survival.detach(),
    }
    return loss, stats


def tau_rb_block_loss(
    logits: torch.Tensor,  # shape: [1, num_blocks*block_size, draft_vocab]
    targets: torch.Tensor,  # shape: [1, num_blocks*block_size, draft_vocab]
    token_ids: torch.Tensor,  # shape: [1, num_blocks*block_size], verifier ids
    loss_mask: torch.Tensor,  # shape: [1, num_blocks*block_size]
    block_size: int,
    *,
    temperature: float = 1.0,
    t2d: torch.Tensor | None = None,
    sample_from_anchor: bool = False,
    vocab_chunk_size: int = 8192,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the exact accepted-length loss averaged over valid blocks.

    Eager reference implementation: full-width fp32 log-softmaxes and a
    chunked vocabulary loop. Numerically validated against brute force, but
    retains ~9 full-width buffers in the autograd graph and OOMs large-anchor
    DFlash steps; prefer ``tau_rb_fused_loss`` on accelerators.

    Args:
        logits: Drafter logits per slot; ``q = softmax(logits)``.
        targets: Verifier-head logits aligned to predict the token AT each
            slot (the same tensor the ``tv``/``kl_div`` losses consume).
        token_ids: Realized training token at each slot, in verifier
            vocabulary ids; the training sequence is the target rollout.
        loss_mask: Aligned loss mask; slot 0 of each block is the anchor
            (masked when ``sample_from_anchor=False``). Invalid means padded
            or past the sequence end (data validity, not acceptance); a
            block is truncated at its first invalid slot.
        block_size: Draft block size (anchor included).
        temperature: Temperature applied to ``targets`` when forming ``p``.
        t2d: Target-to-draft vocab mapping (1-based draft index, 0 = outside
            the draft vocab), or None when the draft uses the full vocab.
        sample_from_anchor: Whether slot 0 predicts a token (DSpark-style)
            instead of holding the anchor input.
        vocab_chunk_size: Vocabulary slice size for the chunked alpha sum,
            bounding transient eager memory.

    Returns:
        Tuple of ``(loss, stats)``. ``loss`` is ``-mean(tau_hat / K)`` over
        blocks with at least one valid drafted slot. ``stats`` (all detached)
        holds ``tau_hat`` [num_blocks], ``block_valid`` [num_blocks],
        ``slot_valid``/``alpha``/``accept``/``survival`` [num_blocks, K];
        ``accept`` at invalid slots is a placeholder, and ``survival`` at a
        slot is its reach probability (exactly 0 beyond the cut) -- select
        both via ``slot_valid``.
    """
    seq_len, start_pos, _num_drafted = _validated_block_args(
        logits, block_size, sample_from_anchor, temperature
    )
    vocab_size = logits.shape[2]
    device = logits.device

    logq = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)
    logp = torch.nn.functional.log_softmax(targets[0].float() / temperature, dim=-1)

    z_idx, in_vocab = _realized_draft_indices(token_ids, t2d, device)
    logq_z = logq.gather(-1, z_idx.unsqueeze(-1)).squeeze(-1)
    logp_z = logp.gather(-1, z_idx.unsqueeze(-1)).squeeze(-1)

    # Exact per-slot acceptance alpha = sum_v min(p_v, q_v), chunked over the
    # vocabulary so the elementwise minimum never materializes full-width.
    alpha = torch.zeros(seq_len, device=device, dtype=torch.float32)
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        alpha = alpha + torch.minimum(
            logq[:, start:end].exp(), logp[:, start:end].exp()
        ).sum(dim=-1)

    return _tau_rb_from_row_stats(
        logq_z, logp_z, alpha, in_vocab, loss_mask[0], block_size, start_pos
    )


def tau_rb_fused_loss(
    logits: torch.Tensor,  # shape: [1, num_blocks*block_size, draft_vocab]
    targets: torch.Tensor,  # shape: [1, num_blocks*block_size, draft_vocab]
    token_ids: torch.Tensor,  # shape: [1, num_blocks*block_size], verifier ids
    loss_mask: torch.Tensor,  # shape: [1, num_blocks*block_size]
    block_size: int,
    *,
    temperature: float = 1.0,
    t2d: torch.Tensor | None = None,
    sample_from_anchor: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fused twin of ``tau_rb_block_loss``: Triton row stats + eager block math.

    Identical signature (minus the chunking knob, which bounds eager-only
    memory) and identical semantics. The per-row statistics stream from
    ``losses.fused`` without materializing any ``[T, V]`` fp32 intermediate,
    which is what makes 512-anchor steps fit; the survival chain and tau_hat
    run eagerly on ``[num_blocks, K]`` tensors. Requires an accelerator with
    Triton; parity with the eager reference is pinned by the unit tests.
    """
    try:
        from speculators.losses.fused import fused_tau_rb_row_stats  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "The fused tau_rb loss requires Triton; use --loss-implementation "
            "eager for compatibility testing."
        ) from error

    seq_len, start_pos, _num_drafted = _validated_block_args(
        logits, block_size, sample_from_anchor, temperature
    )
    z_idx, in_vocab = _realized_draft_indices(token_ids, t2d, logits.device)

    alpha, logq_z, logp_z = fused_tau_rb_row_stats(
        logits.contiguous().view(seq_len, -1),
        targets.contiguous().view(seq_len, -1),
        z_idx,
        temperature,
    )
    return _tau_rb_from_row_stats(
        logq_z, logp_z, alpha, in_vocab, loss_mask[0], block_size, start_pos
    )


def tau_rb_loss(
    logits: torch.Tensor,  # noqa: ARG001
    targets: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    """Registry placeholder so ``--loss-fn`` accepts the ``tau_rb`` name.

    The real computation needs the realized token ids and block structure,
    which the per-position ``(logits, targets)`` dispatch does not carry; it
    is executed by ``compute_metrics`` whenever ``tau_rb`` appears in the
    loss config.
    """
    raise RuntimeError(
        "tau_rb is a block-level loss: it needs the realized token ids and "
        "block structure that the per-position (logits, targets) dispatch "
        "does not carry. It is computed inside compute_metrics() whenever "
        "'tau_rb' appears in the loss config."
    )
