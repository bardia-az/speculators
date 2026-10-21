"""Fused Triton kernels for the dense draft/target logit losses.

One kernel pair serves every fused loss configured in ``losses.utils``: forward
streams the selected reduction from online-softmax statistics; backward
recomputes probabilities from five saved scalars per row and applies the
closed-form gradient. No ``[T, V]`` intermediate is ever materialized or
saved -- the memory win over the eager losses. ``OP`` picks the reduction at
Triton specialization time; ``nla`` and ``lk_hybrid`` are ``[1, T]`` torch
compositions of the primitives instead of kernel branches.

Per-row gradients (dp = softmax(logits), draft; tp = softmax(targets),
treated as constant), L the row loss:

    kl_div  dp - tp
    rkl     dp * (log dp - log tp - L)
    jsd     0.5 * dp * (log(dp / mix) - KL(dp || mix)),  mix = (dp + tp) / 2
    ce      dp - onehot(argmax tp)
    tv      -dp * (1[dp <= tp] - s_s),  s_s = sum of dp where dp <= tp

A second, OP-less kernel pair (``tau_rb_stats_kernel``/``tau_rb_grad_kernel``)
serves the block-level ``tau_rb`` loss the same way: forward emits the per-row
statistics ``(alpha, logq_z, logp_z)`` the block math needs, and backward
applies the closed-form gradient those statistics induce. The block logic
itself (survival chain, tau_hat) is [num_blocks, K]-sized eager math in
``losses.tau_rb``, never full-vocab.

Derived from SpecForge specforge/core/loss.py (Apache-2.0, Unsloth/Liger
lineage); TV gradient sign convention matches Liger ops/tvd.py.
"""

# Triton kernels idiomatically use uppercase constexpr/dim names (B, T, V,
# BLOCK_SIZE); exempt this file from the pep8-naming lints rather than fight
# that convention.
# ruff: noqa: N803, N806

import torch
import triton
import triton.language as tl

from speculators.utils.util import is_npu_available

# Ascend NPU's Unified Buffer (~192 KB) cannot fit the double-row load these
# kernels perform (logits + targets per block) beyond 4096 elements per block;
# triton-ascend raises "ub overflow" at BLOCK_SIZE >= 8192. CE is single-row
# but shares the same cap so all _FusedLoss OPs use one path.
MAX_FUSED_SIZE_NPU = 4096

# Search space for the row-per-program tile, autotuned per (vocab, OP) on the
# first launch. Each program keeps several fp32 vectors of BLOCK_SIZE / threads
# elements live at once, so what bounds the tile is elements *per thread*, not
# BLOCK_SIZE: register use grows at roughly two per element, exhausting the
# 255-register budget around 128, after which Triton spills to local memory.
# Measured registers/spill-bytes per thread, forward kernel, V=151936, A100:
#
#     BLOCK_SIZE     warps=4      warps=8     warps=16     warps=32
#          4096     54/0         32/0         31/0         27/0
#          8192     96/6         48/0         32/0         31/0
#         16384    253/0        127/0         64/0         32/0
#         32768    255/126      254/0         97/0         53/0
#         65536     32/1222     255/152      128/38        64/38
#        131072     32/3748      32/1162      32/532       32/234
#
# Spilling is not disqualifying on its own: 8192 at 4 warps sits inside the
# bound, spills 6 B/thread and still runs within 8% of the winner. The cutoff
# is drawn where spills stop being incidental. Every tile above it spills at
# least 38 B/thread and lands between 1.3x and 22x the winner (kl, V=151936,
# 2048 rows, A100) -- among them the fixed BLOCK_SIZE=131072 this file used to
# select for any vocab > 65536, which at 234 B/thread ran the forward at
# 98 GB/s, 6% of A100 HBM peak. Bounding the space up front beats rediscovering
# that at every startup; test_autotuner_never_picks_a_spilling_tile times the
# unbounded space and checks that nothing above the cutoff wins.
# Which tile *within* that region wins is machine-dependent, which is what the
# autotuner is for; on an A100 it lands on 8192/8 warps at a 151936 vocab and
# 16384/16 at 32000.
_MAX_ELEMS_PER_THREAD = 128
_MIN_ELEMS_PER_THREAD = 4
_BLOCK_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
_THREADS_PER_PROGRAM = (128, 256, 512, 1024)

# tl.constexpr instances: Triton kernels may only read globals wrapped this way.
_LOG2 = tl.constexpr(0.6931471805599453)

# Reduction selector, baked into the kernel at specialization time. Pass
# ``.value`` (a plain int) across the autograd boundary -- Dynamo cannot
# represent a constexpr object as an autograd.Function argument and would
# graph-break; the ``OP: tl.constexpr`` parameter re-wraps it in the kernel.
_OP_KL = tl.constexpr(0)
_OP_RKL = tl.constexpr(1)
_OP_JSD = tl.constexpr(2)
_OP_CE = tl.constexpr(3)
_OP_TV = tl.constexpr(4)

# stats [_N_STATS, n_rows]: [0]/[1] draft row max / sum-exp, [2]/[3] same for
# targets (unused by ce), [4] per-OP -- kl_div/rkl: row loss L | jsd:
# KL(dp||mix) | tv: s_s | ce: argmax as float32 (exact: vocab < 2**24).
_N_STATS = 5

# stats [_TAU_N_STATS, n_rows] for the tau_rb row statistics below: [0]/[1]
# draft row max / sum-exp, [2]/[3] same for temperature-scaled targets, [4]
# alpha = sum_v min(dp, tp), [5] S = sum_v dp * w_v (the min subgradient's
# softmax-correction weight; see tau_rb_stats_kernel).
_TAU_N_STATS = 6


def _tile_configs():
    """Register-resident (BLOCK_SIZE, num_warps) candidates for the autotuner."""
    if is_npu_available():
        # triton-ascend is Unified-Buffer bound rather than register bound:
        # exactly one tile fits, so there is nothing to search.
        return [triton.Config({"BLOCK_SIZE": MAX_FUSED_SIZE_NPU}, num_warps=4)]

    # AMD wavefronts are 64-wide, so a thread count is half as many warps.
    warp_size = 64 if getattr(torch.version, "hip", None) is not None else 32
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=threads // warp_size)
        for threads in _THREADS_PER_PROGRAM
        for block in _BLOCK_SIZES
        if _MIN_ELEMS_PER_THREAD <= block // threads <= _MAX_ELEMS_PER_THREAD
    ]


def _prune_oversized_tiles(configs, nargs, **_):
    """Drop tiles wider than the vocabulary; those only add masked-off lanes.

    Dynamo traces this callback when the launch is inside a compiled region, so
    it has to stay to simple list operations -- a ``min(..., key=...)`` here
    fails to trace.
    """
    cap = triton.next_power_of_2(nargs["n_cols"])
    keep = [config for config in configs if config.kwargs["BLOCK_SIZE"] <= cap]
    if keep:
        return keep
    # Every tile is oversized for a very narrow vocab. They all still mask
    # correctly, but only the narrowest are worth compiling.
    floor = min(config.kwargs["BLOCK_SIZE"] for config in configs)
    return [config for config in configs if config.kwargs["BLOCK_SIZE"] == floor]


# Keyed on OP as well as the vocab: CE streams the row once where the
# distribution losses stream it three times, so they do not share an optimum.
_autotune_tile = triton.autotune(
    configs=_tile_configs(),
    key=["n_cols", "OP"],
    prune_configs_by={"early_config_prune": _prune_oversized_tiles},
)

# The tau_rb kernels carry no OP selector (one reduction each), so their key
# is the vocab alone.
_autotune_tau_tile = triton.autotune(
    configs=_tile_configs(),
    key=["n_cols"],
    prune_configs_by={"early_config_prune": _prune_oversized_tiles},
)


@triton.jit
def _online_stats(row_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Online max (m) and sum-exp (d) over one row."""
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        block_max = tl.max(tl.where(mask, x, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
        m = m_new
    return m, d


@triton.jit
def _log_mix(ldp, ltp):
    """log((exp(ldp) + exp(ltp)) / 2): the JSD mixture, stable in log space."""
    mx = tl.maximum(ldp, ltp)
    return mx + tl.log(tl.exp(ldp - mx) + tl.exp(ltp - mx)) - _LOG2


@_autotune_tile
@triton.jit
def loss_forward_kernel(  # noqa: C901 -- constexpr OP branches, pruned per instance
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One row per program: online softmax stats, then the OP's reduction.

    Masked tail lanes can hold inf/nan before ``tl.where`` selects; they
    never enter an accumulator.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols

    m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    lse_d = m_d + tl.log(z_d)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)

    if OP == _OP_CE:
        # Only the target argmax is needed; no target softmax stats.
        best_val = float("-inf")
        best_idx = 0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            y = tl.load(targets_ptr + offsets, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            block_max = tl.max(y)
            block_arg = tl.argmax(y, axis=0)
            # strict > keeps the first occurrence, the torch.argmax convention
            update = block_max > best_val
            best_idx = tl.where(update, i + block_arg, best_idx)
            best_val = tl.where(update, block_max, best_val)
        logit_at_target = tl.load(logits_ptr + best_idx).cast(tl.float32)
        tl.store(loss_ptr + pid, lse_d - logit_at_target)
        # mypy reads traced Triton code as Python and types best_idx as int
        tl.store(stats_ptr + 4 * stats_row + pid, best_idx.to(tl.float32))  # type: ignore[attr-defined]
    else:
        m_t, z_t = _online_stats(targets_ptr, n_cols, BLOCK_SIZE)
        lse_t = m_t + tl.log(z_t)
        tl.store(stats_ptr + 2 * stats_row + pid, m_t)
        tl.store(stats_ptr + 3 * stats_row + pid, z_t)

        acc = 0.0
        extra = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ldp = x - lse_d
            ltp = y - lse_t
            dp = tl.exp(ldp)
            tp = tl.exp(ltp)
            if OP == _OP_KL:
                acc += tl.sum(tl.where(mask, tp * (ltp - ldp), 0.0))
            elif OP == _OP_RKL:
                acc += tl.sum(tl.where(mask, dp * (ldp - ltp), 0.0))
            elif OP == _OP_JSD:
                lmp = _log_mix(ldp, ltp)
                acc += tl.sum(tl.where(mask, dp * (ldp - lmp), 0.0))
                extra += tl.sum(tl.where(mask, tp * (ltp - lmp), 0.0))
            elif OP == _OP_TV:
                acc += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
                extra += tl.sum(tl.where(mask & (dp <= tp), dp, 0.0))

        # Lint exemptions: `in`/merged compares aren't constexpr-foldable here.
        if OP == _OP_KL or OP == _OP_RKL:  # noqa: PLR1714, SIM109
            tl.store(loss_ptr + pid, acc)
            tl.store(stats_ptr + 4 * stats_row + pid, acc)
        elif OP == _OP_JSD:
            tl.store(loss_ptr + pid, 0.5 * (acc + extra))
            tl.store(stats_ptr + 4 * stats_row + pid, acc)
        elif OP == _OP_TV:
            tl.store(loss_ptr + pid, 1.0 - acc)
            tl.store(stats_ptr + 4 * stats_row + pid, extra)


@_autotune_tile
@triton.jit
def loss_backward_kernel(
    logits_ptr,
    targets_ptr,
    grad_in_ptr,
    grad_out_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Recompute probabilities from the saved stats and apply the OP gradient."""
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go = tl.load(grad_out_ptr + pid).cast(tl.float32)
    if go == 0.0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
        return

    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    extra = tl.load(stats_ptr + 4 * stats_row + pid)
    if OP != _OP_CE:
        # CE passes None for targets_ptr.
        targets_ptr += pid * n_cols
        m_t = tl.load(stats_ptr + 2 * stats_row + pid)
        z_t = tl.load(stats_ptr + 3 * stats_row + pid)
        lse_t = m_t + tl.log(z_t)
    else:
        target_idx = extra.to(tl.int32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        ldp = x - lse_d
        dp = tl.exp(ldp)
        if OP == _OP_CE:
            grad = dp - (offsets == target_idx).to(tl.float32)
        else:
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ltp = y - lse_t
            tp = tl.exp(ltp)
            if OP == _OP_KL:
                grad = dp - tp
            elif OP == _OP_RKL:
                grad = dp * (ldp - ltp - extra)
            elif OP == _OP_JSD:
                grad = 0.5 * dp * (ldp - _log_mix(ldp, ltp) - extra)
            elif OP == _OP_TV:
                grad = -dp * ((dp <= tp).to(tl.float32) - extra)
        tl.store(grad_in_ptr + offsets, go * grad, mask=mask)


@triton.jit
def _online_stats_scaled(row_ptr, n_cols, temperature, BLOCK_SIZE: tl.constexpr):
    """Online max (m) and sum-exp (d) over one row of ``y / temperature``.

    A sibling of ``_online_stats`` kept separate so the shared helper (and the
    losses already validated on it) never pays for the division. The
    temperature is cast to fp32 up front: under torch.compile, Inductor passes
    Python floats as fp64, and an fp64 divisor would promote the loop-carried
    accumulators out of fp32.
    """
    temp = temperature.to(tl.float32)
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        y = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        y = y / temp
        block_max = tl.max(tl.where(mask, y, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(y - m_new), 0.0))
        m = m_new
    return m, d


# min(p, q) routes its subgradient to q where q < p and splits it evenly at
# exact ties -- torch.minimum's autograd rule (0.5 to each operand), which the
# block loss's eager reference inherits. The backward needs that weight, so the
# forward stores S = sum_v dp * w_v alongside alpha:
#
#     w_v            = 1[dp_v < tp_v] + 0.5 * 1[dp_v == tp_v]
#     d alpha/dx_v   = dp_v * (w_v - S)              (softmax correction)
#     d logq_z/dx_v  = (v == z) - dp_v               (gather + correction)
_TAU_TIE_WEIGHT = tl.constexpr(0.5)


@_autotune_tau_tile
@triton.jit
def tau_rb_stats_kernel(
    logits_ptr,
    targets_ptr,
    z_ptr,
    alpha_ptr,
    logq_z_ptr,
    logp_z_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """One row per program: tau_rb row statistics for the block loss.

    Two passes per row: ``min(dp, tp)`` has no shared max-shift that keeps both
    operands normalized, so pass 1 finishes the online-softmax normalizers and
    pass 2 reduces on probabilities normalized by them. Pass 2 also gathers the
    realized token's draft and target logits. Masked tail lanes hold 0.0 and
    never enter an accumulator.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols
    temp = temperature.to(tl.float32)

    m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    lse_d = m_d + tl.log(z_d)
    m_t, z_t = _online_stats_scaled(targets_ptr, n_cols, temp, BLOCK_SIZE)
    lse_t = m_t + tl.log(z_t)

    alpha = 0.0
    extra = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        dp = tl.exp(x - lse_d)
        tp = tl.exp(y / temp - lse_t)
        alpha += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
        w = tl.where(dp < tp, 1.0, tl.where(dp == tp, _TAU_TIE_WEIGHT, 0.0))
        extra += tl.sum(tl.where(mask, dp * w, 0.0))

    z = tl.load(z_ptr + pid)
    x_z = tl.load(logits_ptr + z).cast(tl.float32)
    y_z = tl.load(targets_ptr + z).cast(tl.float32)
    tl.store(alpha_ptr + pid, alpha)
    tl.store(logq_z_ptr + pid, x_z - lse_d)
    tl.store(logp_z_ptr + pid, y_z / temp - lse_t)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)
    tl.store(stats_ptr + 2 * stats_row + pid, m_t)
    tl.store(stats_ptr + 3 * stats_row + pid, z_t)
    tl.store(stats_ptr + 4 * stats_row + pid, alpha)
    tl.store(stats_ptr + 5 * stats_row + pid, extra)


@_autotune_tau_tile
@triton.jit
def tau_rb_grad_kernel(
    logits_ptr,
    targets_ptr,
    grad_alpha_ptr,
    grad_qz_ptr,
    grad_in_ptr,
    stats_ptr,
    stats_row,
    z_ptr,
    n_cols,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """Recompute probabilities from the saved stats and apply the tau_rb gradient.

    Both row-gradient scalars (through alpha and through the gathered logq_z)
    arrive from the eager block math; rows where both are zero take the
    early-out and return exact zeros from an uninitialized buffer.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols
    temp = temperature.to(tl.float32)

    go_a = tl.load(grad_alpha_ptr + pid)
    go_z = tl.load(grad_qz_ptr + pid)
    if go_a == 0.0 and go_z == 0.0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
        return

    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    m_t = tl.load(stats_ptr + 2 * stats_row + pid)
    z_t = tl.load(stats_ptr + 3 * stats_row + pid)
    lse_t = m_t + tl.log(z_t)
    extra = tl.load(stats_ptr + 5 * stats_row + pid)
    z = tl.load(z_ptr + pid).to(tl.int32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        dp = tl.exp(x - lse_d)
        tp = tl.exp(y / temp - lse_t)
        w = tl.where(dp < tp, 1.0, tl.where(dp == tp, _TAU_TIE_WEIGHT, 0.0))
        grad = go_a * dp * (w - extra) + go_z * ((offsets == z).to(tl.float32) - dp)
        tl.store(grad_in_ptr + offsets, grad, mask=mask)


class _FusedLoss(torch.autograd.Function):
    """Shared autograd wrapper; ``op`` selects the loss reduction.

    Targets intentionally receive no gradient; use the eager implementation when
    target gradients are required.
    """

    @staticmethod
    def forward(ctx, logits, targets, op):
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(B * T, V)
        targets_flat = targets.contiguous().view(B * T, V)
        loss = torch.empty(B * T, device=logits.device, dtype=torch.float32)
        stats = torch.empty(_N_STATS, B * T, device=logits.device, dtype=torch.float32)
        loss_forward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            loss,
            stats,
            stats.stride(0),
            V,
            OP=op,
        )
        # CE backward only needs the target argmax cached in stats.
        ctx.save_for_backward(
            logits_flat, None if op == _OP_CE.value else targets_flat, stats
        )
        ctx.op = op
        ctx.shape = (B, T, V)
        return loss.view(B, T)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, targets_flat, stats = ctx.saved_tensors
        B, T, V = ctx.shape
        grad_in = torch.empty_like(logits_flat)
        loss_backward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            grad_in,
            grad_output.contiguous().view(-1),
            stats,
            stats.stride(0),
            V,
            OP=ctx.op,
        )
        return grad_in.view(B, T, V), None, None


_NLA_EPS = 1e-5


def fused_kl_div_loss(logits, targets):
    """Per-position forward KL ``[1, T]``; fused twin of ``kl_div_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_KL.value)


def fused_reverse_kl_div_loss(logits, targets):
    """Per-position reverse KL ``[1, T]``; fused twin of ``reverse_kl_div_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_RKL.value)


def fused_js_div_loss(logits, targets):
    """Per-position JSD ``[1, T]``; fused twin of ``js_div_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_JSD.value)


def fused_ce_loss(logits, targets):
    """Per-position CE vs argmax(targets) ``[1, T]``; fused twin of ``ce_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_CE.value)


def fused_tv_loss(logits, targets):
    """Per-position TV distance ``[1, T]`` from draft/target logits (fused Triton)."""
    return _FusedLoss.apply(logits, targets, _OP_TV.value)


def fused_nla_loss(logits, targets):
    """Per-position negative-log-acceptance ``[1, T] = -log(alpha)``; composes on TV."""
    return -torch.log((1.0 - fused_tv_loss(logits, targets)).clamp_min(_NLA_EPS))


def fused_lk_hybrid_loss(logits, targets, eta: float = 3.0):
    """Per-position hybrid LK ``[1, T]``; fused KL + TV composed as in
    ``lk_hybrid_loss`` (blend weight detached).
    """
    kl = fused_kl_div_loss(logits, targets)
    tv = fused_tv_loss(logits, targets)
    weight = torch.exp(-eta * (1.0 - tv).detach())
    return weight * kl + (1.0 - weight) * tv


class _TauRbRowStats(torch.autograd.Function):
    """Autograd wrapper for the tau_rb per-row statistics.

    Like ``_FusedLoss``, targets intentionally receive no gradient; use the
    eager implementation when target gradients are required. ``logp_z`` is a
    constant of the logits and marked non-differentiable; the caller's block
    math consumes it without tracking.
    """

    @staticmethod
    def forward(ctx, logits_flat, targets_flat, z_idx, temperature):
        n_rows, vocab_size = logits_flat.shape
        device = logits_flat.device
        alpha = torch.empty(n_rows, device=device, dtype=torch.float32)
        logq_z = torch.empty_like(alpha)
        logp_z = torch.empty_like(alpha)
        stats = torch.empty(_TAU_N_STATS, n_rows, device=device, dtype=torch.float32)
        tau_rb_stats_kernel[(n_rows,)](
            logits_flat,
            targets_flat,
            z_idx,
            alpha,
            logq_z,
            logp_z,
            stats,
            stats.stride(0),
            vocab_size,
            temperature,
        )
        ctx.save_for_backward(logits_flat, targets_flat, stats, z_idx)
        ctx.temperature = temperature
        ctx.mark_non_differentiable(logp_z)
        return alpha, logq_z, logp_z

    @staticmethod
    def backward(ctx, grad_alpha, grad_logq_z, _grad_logp_z):
        logits_flat, targets_flat, stats, z_idx = ctx.saved_tensors
        if grad_alpha is None and grad_logq_z is None:
            return None, None, None, None
        n_rows, vocab_size = logits_flat.shape
        device = logits_flat.device
        if grad_alpha is None:
            grad_alpha = torch.zeros(n_rows, device=device, dtype=torch.float32)
        if grad_logq_z is None:
            grad_logq_z = torch.zeros(n_rows, device=device, dtype=torch.float32)
        grad_in = torch.empty_like(logits_flat)
        tau_rb_grad_kernel[(n_rows,)](
            logits_flat,
            targets_flat,
            grad_alpha.contiguous(),
            grad_logq_z.contiguous(),
            grad_in,
            stats,
            stats.stride(0),
            z_idx,
            vocab_size,
            ctx.temperature,
        )
        return grad_in, None, None, None


def fused_tau_rb_row_stats(logits, targets, z_idx, temperature):
    """Per-row ``(alpha, logq_z, logp_z)`` statistics for the tau_rb block loss.

    Args:
        logits: Drafter logits ``[rows, vocab]`` (any float dtype; cast to
            fp32 on load like the other fused losses).
        targets: Verifier-head logits ``[rows, vocab]``, divided by
            ``temperature`` in-kernel (no full-width copy is materialized).
        z_idx: One draft-vocab index per row (int64); entries for tokens
            outside the draft vocab are valid indices whose gathered values
            the caller overwrites.
        temperature: Target temperature forming ``p`` (must be > 0).

    Returns:
        Tuple of fp32 ``[rows]`` tensors: ``alpha`` (per-row exact acceptance
        ``sum_v min(p, q)``), ``logq_z``/``logp_z`` (log-probabilities of the
        realized token under draft and target). Gradients flow into
        ``logits`` through ``alpha`` and ``logq_z``; ``targets`` receives
        none.
    """
    return _TauRbRowStats.apply(logits, targets, z_idx, temperature)
