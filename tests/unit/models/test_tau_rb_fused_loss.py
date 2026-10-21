"""Fused tau_rb loss vs its eager reference (speculators.losses.fused).

The fused path must match ``tau_rb_block_loss`` in loss value, per-slot
statistics, and logits-gradient across the regimes that catch distinct bugs:
fp32 with saturated point-mass rows (log-space underflow), bf16 (the training
dtype), and the 151936-wide vocab (multi-block streaming, non-power-of-2
tail). The gradient formulas are additionally pinned by finite differences,
and an all-tie row (logits == targets) exercises the min subgradient's tie
branch without near-tie fp ambiguity.

The 512-anchor memory smoke is the motivating case: eager tau_rb retains ~9
full-width fp32 buffers in the autograd graph and OOMs an L40 at that shape
during training; the fused path must run it holding little beyond the logits
gradient it cannot avoid allocating.

The accelerator is auto-detected: CUDA when present, otherwise Ascend NPU
(via triton-ascend), like test_fused_losses.py.
"""

import random

import pytest
import torch

from speculators.losses.tau_rb import tau_rb_block_loss, tau_rb_fused_loss
from speculators.utils.util import is_npu_available


def _accelerator_device() -> str | None:
    """Pick the accelerator the fused kernels can run on, or None to skip."""
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    return None


DEVICE = _accelerator_device()
requires_accelerator = pytest.mark.skipif(
    DEVICE is None,
    reason="fused tau_rb requires a CUDA or Ascend NPU accelerator",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="memory and spill tests use torch.cuda APIs",
)

# House tolerances from test_fused_losses.py: losses are fp32 on both paths;
# gradients allow bf16 1-ulp rounding at the leaf. Stats are bounded [0, K]
# quantities compared tighter than the loss. A wrong formula errs by orders
# of magnitude more than any of these bounds.
LOSS_TOL = {"atol": 1e-4, "rtol": 1e-3}
STATS_TOL = {"atol": 1e-5, "rtol": 1e-4}
GRAD_TOL = {"atol": 1e-3, "rtol": 1e-2}


def _assert_fused_matches_eager(
    logits,
    targets,
    token_ids,
    loss_mask,
    block_size,
    *,
    temperature=1.0,
    t2d=None,
    sample_from_anchor=False,
    grad_tol=GRAD_TOL,
):
    """Fused == eager: loss, every stat, and the logits gradient."""
    le = logits.detach().clone().requires_grad_(True)
    lf = logits.detach().clone().requires_grad_(True)
    kwargs = {
        "temperature": temperature,
        "t2d": t2d,
        "sample_from_anchor": sample_from_anchor,
    }
    out_e, stats_e = tau_rb_block_loss(
        le, targets, token_ids, loss_mask, block_size, **kwargs
    )
    out_f, stats_f = tau_rb_fused_loss(
        lf, targets, token_ids, loss_mask, block_size, **kwargs
    )
    torch.testing.assert_close(out_f, out_e, **LOSS_TOL)
    for key in ("tau_hat", "alpha", "accept", "survival"):
        torch.testing.assert_close(stats_f[key], stats_e[key], **STATS_TOL)
    for key in ("block_valid", "slot_valid"):
        assert torch.equal(stats_f[key], stats_e[key]), key

    out_e.backward()
    out_f.backward()
    assert le.grad is not None
    assert lf.grad is not None
    assert torch.isfinite(lf.grad).all()
    torch.testing.assert_close(lf.grad.float(), le.grad.float(), **grad_tol)


def _random_case(seq_len, vocab, block_size, seed, device=DEVICE, dtype=torch.float32):
    """Seeded (logits, targets, token_ids, loss_mask) with truncated blocks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logits = (torch.randn(1, seq_len, vocab, generator=gen) * 3).to(device, dtype)
    targets = (torch.randn(1, seq_len, vocab, generator=gen) * 3).to(device, dtype)
    token_ids = torch.randint(0, vocab, (1, seq_len), generator=gen).to(device)
    drafted = torch.arange(seq_len) % block_size != 0
    valid = (torch.rand(seq_len, generator=gen) > 0.15) & drafted
    loss_mask = valid.float().unsqueeze(0).to(device)
    return logits, targets, token_ids, loss_mask


@requires_accelerator
@pytest.mark.parametrize("temperature", [1.0, 0.01, 2.0])
def test_parity_small_fp32(temperature):
    """Fused == eager on dense fp32 data, across target temperatures."""
    torch.manual_seed(0)
    logits, targets, token_ids, loss_mask = _random_case(64, 512, 8, 0)
    _assert_fused_matches_eager(
        logits, targets, token_ids, loss_mask, 8, temperature=temperature
    )


@requires_accelerator
def test_parity_point_mass_rows():
    """Saturated +-30 rows stress the log-space reductions and underflow."""
    torch.manual_seed(1)
    logits, targets, token_ids, loss_mask = _random_case(64, 512, 8, 1)
    logits[0, -2:] = -30.0
    targets[0, -2:] = -30.0
    logits[0, -2:, 0] = 30.0
    targets[0, -2, 0] = 30.0  # one agreeing point mass, then a disagreeing one
    targets[0, -1, 7] = 30.0
    _assert_fused_matches_eager(logits, targets, token_ids, loss_mask, 8)


@requires_accelerator
def test_parity_bf16():
    """The training dtype: both paths quantize at the leaf."""
    torch.manual_seed(2)
    logits, targets, token_ids, loss_mask = _random_case(
        64, 512, 8, 2, dtype=torch.bfloat16
    )
    _assert_fused_matches_eager(logits, targets, token_ids, loss_mask, 8)


@requires_accelerator
def test_parity_qwen_vocab():
    """Qwen3's 151936 vocab: multi-block streaming plus a masked tail."""
    torch.manual_seed(3)
    logits, targets, token_ids, loss_mask = _random_case(48, 151936, 16, 3)
    _assert_fused_matches_eager(logits, targets, token_ids, loss_mask, 16)


@requires_accelerator
def test_parity_t2d_out_of_vocab_death():
    """Tokens outside the draft vocab kill the block identically on both paths.

    ``t2d`` carries 1-based draft indices; ids mapping to 0 are out of vocab,
    so logq_z is forced to -inf and survival past that slot is exactly 0.
    """
    torch.manual_seed(4)
    seq_len, full_vocab, draft_vocab, block_size = 64, 1000, 512, 8
    logits, targets, _full_token_ids, loss_mask = _random_case(
        seq_len, draft_vocab, block_size, 4
    )
    # token ids live in verifier space; half the draft vocab is unmapped
    t2d = torch.zeros(full_vocab, dtype=torch.long)
    t2d[: draft_vocab // 2] = torch.arange(1, draft_vocab // 2 + 1)
    token_ids = torch.randint(0, full_vocab, (1, seq_len)).to(DEVICE)
    _assert_fused_matches_eager(
        logits, targets, token_ids, loss_mask, block_size, t2d=t2d
    )
    # sanity: the setup really produces out-of-vocab deaths
    _, stats = tau_rb_fused_loss(
        logits, targets, token_ids, loss_mask, block_size, t2d=t2d
    )
    assert (stats["survival"] == 0).any()


@requires_accelerator
def test_parity_sample_from_anchor():
    """DSpark-style blocks draft slot 0 too; the fused path must agree."""
    torch.manual_seed(5)
    logits, targets, token_ids, loss_mask = _random_case(64, 512, 8, 5)
    loss_mask = torch.ones_like(loss_mask)  # slot 0 is drafted, not an anchor
    _assert_fused_matches_eager(
        logits, targets, token_ids, loss_mask, 8, sample_from_anchor=True
    )


@requires_accelerator
def test_all_tie_rows_match_eager():
    """logits == targets ties every min at exactly dp == tp.

    The tie branch of the subgradient weight (torch.minimum's 0.5/0.5 split,
    mirrored by the kernel) must not poison the gradient; the alpha-path
    contribution cancels to ~0 and both implementations must agree on that.
    """
    torch.manual_seed(6)
    logits, targets, token_ids, loss_mask = _random_case(64, 512, 8, 6)
    targets = logits.clone()
    _assert_fused_matches_eager(
        logits, targets, token_ids, loss_mask, 8, grad_tol={"atol": 1e-5, "rtol": 1e-3}
    )


@requires_accelerator
def test_masked_rows_have_exact_zero_gradient():
    """Dead blocks and anchor slots take the backward early-out.

    Their rows receive gradient exactly 0 from a buffer torch.empty_like
    left uninitialized -- the kernel must zero-fill the early-out path, not
    only where it computes.
    """
    torch.manual_seed(7)
    logits, targets, token_ids, loss_mask = _random_case(64, 512, 8, 7)
    loss_mask[0, 4 * 8 : 8 * 8] = 0.0  # four fully dead blocks
    lf = logits.detach().clone().requires_grad_(True)
    loss, _ = tau_rb_fused_loss(lf, targets, token_ids, loss_mask, 8)
    loss.backward()
    assert (lf.grad[0, 4 * 8 : 8 * 8] == 0).all()
    assert (lf.grad[0, ::8] == 0).all()  # anchor rows are sliced out of the loss
    assert lf.grad[0, 1:].abs().sum() > 0


@requires_accelerator
@pytest.mark.parametrize("implementation", ["eager", "fused"])
def test_gradient_matches_finite_differences(implementation):
    """Central differences pin the gradient formula itself (both impls).

    The loss is piecewise smooth (clamped accept factors, min kinks); sampled
    coordinates sit away from kinks with overwhelming probability, so the
    assertion tolerates a small share of near-kink outliers while requiring
    the bulk to be FD-tight. Running eager through the same check validates
    the tolerance.
    """
    torch.manual_seed(8)
    seq_len, vocab, block_size = 16, 12, 4
    logits, targets, token_ids, loss_mask = _random_case(seq_len, vocab, block_size, 8)
    lf = logits.detach().clone().requires_grad_(True)
    fn = tau_rb_fused_loss if implementation == "fused" else tau_rb_block_loss
    loss, _ = fn(lf, targets, token_ids, loss_mask, block_size)
    loss.backward()
    grad = lf.grad[0]

    base = logits.detach().clone()
    eps = 1e-3
    coords = [
        (t, v)
        for t in range(seq_len)
        for v in range(vocab)
        if (t % block_size != 0) and (loss_mask[0, t] != 0)
    ]
    coords = random.Random(0).sample(coords, 60)
    errs = []
    with torch.no_grad():
        for t, v in coords:
            orig = base[0, t, v].item()
            base[0, t, v] = orig + eps
            loss_plus, _ = fn(base, targets, token_ids, loss_mask, block_size)
            base[0, t, v] = orig - eps
            loss_minus, _ = fn(base, targets, token_ids, loss_mask, block_size)
            base[0, t, v] = orig
            fd = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            errs.append(abs(fd - grad[t, v].item()))
    errs.sort()
    assert errs[len(errs) // 2] < 1e-4, errs
    assert errs[-6] < 1e-3, f"more than 10% of samples off: {errs[-6:]}"
    assert errs[-1] < 5e-2, f"FD outlier too large: {errs[-1]}"


@requires_accelerator
def test_compiles_fullgraph():
    """torch.compile(fullgraph=True) must trace the fused block loss.

    The DFlash forward (which calls compute_metrics) is wrapped in
    torch.compile; the autograd.Function crossing with an int64 index tensor
    and a float temperature must not graph-break.
    """
    logits, targets, token_ids, loss_mask = _random_case(32, 512, 8, 9)
    lf = logits.detach().clone().requires_grad_(True)

    def run(a, b, c, d):
        loss, _ = tau_rb_fused_loss(a, b, c, d, 8)
        return loss

    compiled = torch.compile(run, fullgraph=True)
    compiled(lf, targets, token_ids, loss_mask).backward()
    assert lf.grad is not None
    assert torch.isfinite(lf.grad).all()


@requires_cuda
@pytest.mark.slow
def test_512_anchor_memory_and_speed():
    """The shape that OOMs eager: 512 anchors x block 16 x Qwen3 vocab.

    Eager tau_rb saves two full-width fp32 log-softmax outputs for backward
    (~10 GiB at this vocab) before any autograd temporary, and the training
    graph around it OOMs an L40. The fused path must complete forward +
    backward holding little beyond the bf16 logits gradient (2.5 GiB) it
    cannot avoid. Eager is also run at half the anchors as a value-parity
    witness at production vocab, and to compare peak memory and wall time.
    """
    vocab, block = 151936, 16

    def make(num_anchors, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        seq_len = num_anchors * block
        logits = (
            torch.randn(1, seq_len, vocab, generator=gen)
            .to("cuda", torch.bfloat16)
            .requires_grad_(True)
        )
        targets = torch.randn(1, seq_len, vocab, generator=gen).to(
            "cuda", torch.bfloat16
        )
        token_ids = torch.randint(0, vocab, (1, seq_len), generator=gen).cuda()
        valid = torch.rand(1, seq_len, generator=gen) > 0.1
        loss_mask = valid.float() * (torch.arange(seq_len) % block != 0)
        return logits, targets, token_ids, loss_mask.cuda()

    def run_pair(num_anchors, implementation, seed):
        logits, targets, token_ids, loss_mask = make(num_anchors, seed)
        fn = tau_rb_fused_loss if implementation == "fused" else tau_rb_block_loss
        logits.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss, _ = fn(logits, targets, token_ids, loss_mask, block)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        grad = logits.grad.detach().clone()
        assert torch.isfinite(grad).all()
        return loss.detach().clone(), grad, peak

    loss_f512, _grad_f512, peak_f = run_pair(512, "fused", 10)
    grad_bytes = 512 * block * vocab * 2  # bf16 logits gradient
    assert peak_f < grad_bytes + 1.5e9, (
        f"fused peak {peak_f / 2**30:.2f} GiB exceeds the gradient buffer "
        f"({grad_bytes / 2**30:.2f} GiB) plus slack"
    )

    # value + gradient parity at production vocab, and a peak the fused path
    # beats by a wide margin, at the largest shape eager fits comfortably.
    loss_f, grad_f, peak_f256 = run_pair(256, "fused", 10)
    loss_e, grad_e, peak_e = run_pair(256, "eager", 10)
    torch.testing.assert_close(loss_f, loss_e, **LOSS_TOL)
    torch.testing.assert_close(grad_f, grad_e, **GRAD_TOL)
    eager_min_half = 2 * 256 * block * vocab * 4  # logq + logp saved, fp32
    assert peak_e > eager_min_half, (
        "eager peak does not reflect the full-width buffers this test uses "
        "as its memory bound"
    )
    assert peak_f256 < peak_e / 2, (
        f"fused peak {peak_f256 / 2**30:.2f} GiB is not decisively below the "
        f"eager peak at the same shape ({peak_e / 2**30:.2f} GiB)"
    )

    # wall clock: fused should not be slower than eager at equal shape
    from triton.testing import do_bench  # noqa: PLC0415

    logits, targets, token_ids, loss_mask = make(256, 11)

    def step_fused():
        loss, _ = tau_rb_fused_loss(logits, targets, token_ids, loss_mask, block)
        loss.backward()

    def step_eager():
        loss, _ = tau_rb_block_loss(logits, targets, token_ids, loss_mask, block)
        loss.backward()

    ms_f = do_bench(step_fused, warmup=5, rep=20)
    ms_e = do_bench(step_eager, warmup=5, rep=20)
    assert ms_f <= 1.5 * ms_e, (
        f"fused step {ms_f:.2f} ms is not competitive with eager {ms_e:.2f} ms"
    )


@requires_cuda
def test_tau_kernels_register_resident():
    """The autotuner's chosen tile for each tau kernel spills nothing.

    The tau kernels hold more live vectors than the per-position losses
    (two-pass streaming plus the tie weight), so the register-resident
    assumption behind _tile_configs() needs its own check here. Which tile
    wins is machine-dependent; that it spills nothing is the invariant.
    """
    fused_losses = pytest.importorskip("speculators.losses.fused")
    rows, vocab = 64, 151936
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16)
    targets = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16)
    z = torch.randint(0, vocab, (rows,), device="cuda", dtype=torch.int64)
    alpha = torch.empty(rows, device="cuda", dtype=torch.float32)
    logq_z = torch.empty_like(alpha)
    logp_z = torch.empty_like(alpha)
    stats = torch.empty(
        fused_losses._TAU_N_STATS, rows, device="cuda", dtype=torch.float32
    )
    grad_in = torch.empty_like(logits)
    ones = torch.ones(rows, device="cuda", dtype=torch.float32)

    # trigger the autotuner, then replay its chosen tile explicitly to read
    # the compiler's spill count
    fused_losses.tau_rb_stats_kernel[(rows,)](
        logits,
        targets,
        z,
        alpha,
        logq_z,
        logp_z,
        stats,
        stats.stride(0),
        vocab,
        1.0,
    )
    best = fused_losses.tau_rb_stats_kernel.best_config
    compiled = fused_losses.tau_rb_stats_kernel.fn[(rows,)](
        logits,
        targets,
        z,
        alpha,
        logq_z,
        logp_z,
        stats,
        stats.stride(0),
        vocab,
        1.0,
        BLOCK_SIZE=best.kwargs["BLOCK_SIZE"],
        num_warps=best.num_warps,
    )
    assert compiled.n_spills == 0, (
        f"stats kernel best tile {best.kwargs['BLOCK_SIZE']}/{best.num_warps}w "
        f"spills {compiled.n_spills} B/thread"
    )

    fused_losses.tau_rb_grad_kernel[(rows,)](
        logits,
        targets,
        ones,
        ones,
        grad_in,
        stats,
        stats.stride(0),
        z,
        vocab,
        1.0,
    )
    best = fused_losses.tau_rb_grad_kernel.best_config
    compiled = fused_losses.tau_rb_grad_kernel.fn[(rows,)](
        logits,
        targets,
        ones,
        ones,
        grad_in,
        stats,
        stats.stride(0),
        z,
        vocab,
        1.0,
        BLOCK_SIZE=best.kwargs["BLOCK_SIZE"],
        num_warps=best.num_warps,
    )
    assert compiled.n_spills == 0, (
        f"grad kernel best tile {best.kwargs['BLOCK_SIZE']}/{best.num_warps}w "
        f"spills {compiled.n_spills} B/thread"
    )
