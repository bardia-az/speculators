"""Unit tests for the tau_rb exact accepted-length loss.

Every exactness property (unbiasedness vs brute force, worked example, greedy
limit, gradient capping, masking) runs against BOTH implementations: the eager
reference (CPU or accelerator) and the fused Triton twin (accelulator legs are
collected only when one is present, matching test_fused_losses.py). The fused
implementation must satisfy the same analytic expectations, not merely agree
with eager -- agreement is pinned separately in test_tau_rb_fused_loss.py.
"""

import pytest
import torch

from speculators.losses.tau_rb import (
    tau_rb_block_loss,
    tau_rb_fused_loss,
    tau_rb_loss,
)
from speculators.utils.util import is_npu_available

_PROB_FLOOR = 1e-12


def _accelerator_device() -> str | None:
    """Pick the accelerator the fused kernels can run on, or None to skip."""
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    return None


DEVICE = _accelerator_device()
IMPLEMENTATIONS = ["eager"] + (["fused"] if DEVICE else [])


def _block_loss(
    implementation,
    logits,
    targets,
    token_ids,
    loss_mask,
    block_size,
    **kwargs,
):
    """Run one tau_rb implementation; fused legs run on the accelerator.

    Tensors are built on CPU (seeded generators stay deterministic that way)
    and moved for the fused legs, so both implementations see identical
    inputs bit for bit.
    """
    fn = tau_rb_fused_loss if implementation == "fused" else tau_rb_block_loss
    if implementation == "fused":

        def move(t):
            return t.to(DEVICE) if isinstance(t, torch.Tensor) else t

        logits, targets, token_ids, loss_mask = map(
            move, (logits, targets, token_ids, loss_mask)
        )
        kwargs = {k: move(v) for k, v in kwargs.items()}
    return fn(logits, targets, token_ids, loss_mask, block_size, **kwargs)


def _probs_to_logits(probs: list[float]) -> torch.Tensor:
    """Distribution -> logits whose softmax recovers it exactly."""
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.log(p.clamp_min(_PROB_FLOOR))


def _run_block(
    q_rows: torch.Tensor,
    target_rows: torch.Tensor,
    path: list[int],
    loss_mask: list[float] | None = None,
    implementation: str = "eager",
    **kwargs,
):
    """Run one anchor block (slot 0 = anchor, slots 1..K drafted)."""
    num_drafted, vocab_size = q_rows.shape
    block_size = num_drafted + 1
    logits = torch.zeros(1, block_size, vocab_size)
    logits[0, 1:] = q_rows
    targets = torch.zeros(1, block_size, vocab_size)
    targets[0, 1:] = target_rows
    token_ids = torch.tensor([[0, *path]])
    if loss_mask is None:
        loss_mask = [0.0] + [1.0] * num_drafted
    loss_mask = torch.tensor([loss_mask])
    return _block_loss(
        implementation,
        logits,
        targets,
        token_ids,
        loss_mask,
        block_size,
        **kwargs,
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
class TestUnbiasedness:
    def test_matches_brute_force(self, implementation):
        """Averaging tau_hat over sampled rollouts recovers the enumerated
        exact expected accepted length within Monte Carlo error."""
        vocab = 3
        p1 = [0.5, 0.3, 0.2]
        p2_cond = {
            0: [0.6, 0.3, 0.1],
            1: [0.1, 0.5, 0.4],
            2: [0.2, 0.2, 0.6],
        }
        q1 = [0.4, 0.4, 0.2]
        q2 = [0.5, 0.25, 0.25]

        # Exact value by enumeration:
        # P(slots 1..j accepted) = sum over paths of prod min(p_k(z_k), q_k(z_k))
        p_acc1 = sum(min(p1[v], q1[v]) for v in range(vocab))
        p_acc12 = sum(
            min(p1[z1], q1[z1]) * sum(min(p2_cond[z1][v], q2[v]) for v in range(vocab))
            for z1 in range(vocab)
        )
        tau_exact = 1.0 + p_acc1 + p_acc12

        # Sample rollouts from the target conditionals, one block each.
        gen = torch.Generator().manual_seed(0)
        n = 5000
        z1s = torch.multinomial(
            torch.tensor(p1), n, replacement=True, generator=gen
        ).tolist()
        z2s = [
            int(torch.multinomial(torch.tensor(p2_cond[z1]), 1, generator=gen))
            for z1 in z1s
        ]
        q1_row = _probs_to_logits(q1)
        q2_row = _probs_to_logits(q2)
        logits = torch.zeros(1, n * 3, vocab)
        logits[0, 1::3] = q1_row
        logits[0, 2::3] = q2_row
        targets = torch.zeros(1, n * 3, vocab)
        targets[0, 1::3] = _probs_to_logits(p1)
        for b, z1 in enumerate(z1s):
            targets[0, b * 3 + 2] = _probs_to_logits(p2_cond[z1])
        token_ids = torch.zeros(1, n * 3, dtype=torch.long)
        token_ids[0, 1::3] = torch.tensor(z1s)
        token_ids[0, 2::3] = torch.tensor(z2s)
        loss_mask = torch.zeros(1, n * 3)
        loss_mask[0, 1::3] = 1.0
        loss_mask[0, 2::3] = 1.0

        loss, stats = _block_loss(
            implementation, logits, targets, token_ids, loss_mask, block_size=3
        )

        tau = stats["tau_hat"]
        tau_mean = tau.mean().item()
        stderr = tau.std(unbiased=True).item() / n**0.5
        assert abs(tau_mean - tau_exact) < 5 * stderr
        assert stderr < 0.01  # the MC check is tight enough to be meaningful
        # loss is the mean over blocks of -tau_hat / K
        expected = torch.tensor(-tau_mean / 2, device=loss.device)
        assert torch.allclose(loss, expected, atol=1e-6)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
class TestWorkedExample:
    """Target says "of course" or "no problem" 50/50.

    Drafter slot 1 = (of 0.9, no 0.1). Slot 2 drafts "course" or "problem"
    with probability 1. tau_exact is 2.1 vs 1.7; the biased product-of-alpha
    variant gives 1.9 for both, so the test distinguishes the two.
    """

    OF, COURSE, NO, PROBLEM = 0, 1, 2, 3
    P1 = [0.5, 0.0, 0.5, 0.0]
    P2_GIVEN_OF = [0.0, 1.0, 0.0, 0.0]
    P2_GIVEN_NO = [0.0, 0.0, 0.0, 1.0]
    Q1 = [0.9, 0.0, 0.1, 0.0]

    def _brute_force(self, q2):
        vocab = 4
        p_acc1 = sum(min(self.P1[v], self.Q1[v]) for v in range(vocab))
        p_acc12 = sum(
            min(self.P1[z1], self.Q1[z1]) * sum(min(p2[v], q2[v]) for v in range(vocab))
            for z1, p2 in ((self.OF, self.P2_GIVEN_OF), (self.NO, self.P2_GIVEN_NO))
        )
        return 1.0 + p_acc1 + p_acc12

    def _biased_product(self, q2):
        """1 + alpha_1 + alpha_1 * alpha_2(marginal): the forbidden variant."""
        vocab = 4
        alpha1 = sum(min(self.P1[v], self.Q1[v]) for v in range(vocab))
        p2_marginal = [
            0.5 * self.P2_GIVEN_OF[v] + 0.5 * self.P2_GIVEN_NO[v] for v in range(vocab)
        ]
        alpha2 = sum(min(p2_marginal[v], q2[v]) for v in range(vocab))
        return 1.0 + alpha1 + alpha1 * alpha2

    def _expected_tau_hat(self, q2, implementation):
        """E[tau_hat] over rollout paths, weighted by path probability."""
        values = []
        for z1, p2 in ((self.OF, self.P2_GIVEN_OF), (self.NO, self.P2_GIVEN_NO)):
            z2 = self.COURSE if z1 == self.OF else self.PROBLEM
            q_rows = torch.stack([_probs_to_logits(self.Q1), _probs_to_logits(q2)])
            t_rows = torch.stack([_probs_to_logits(self.P1), _probs_to_logits(p2)])
            _, stats = _run_block(
                q_rows, t_rows, [z1, z2], implementation=implementation
            )
            values.append((0.5, stats["tau_hat"].item()))
        return sum(weight * value for weight, value in values)

    @pytest.mark.parametrize(
        "q2, expected",
        [([0.0, 1.0, 0.0, 0.0], 2.1), ([0.0, 0.0, 0.0, 1.0], 1.7)],
    )
    def test_brute_force_and_estimator_agree(self, q2, expected, implementation):
        assert self._brute_force(q2) == pytest.approx(expected, abs=1e-6)
        assert self._expected_tau_hat(q2, implementation) == pytest.approx(
            expected, abs=1e-5
        )

    @pytest.mark.parametrize(
        "q2, expected",
        [([0.0, 1.0, 0.0, 0.0], 2.1), ([0.0, 0.0, 0.0, 1.0], 1.7)],
    )
    def test_mc_averaged_estimator(self, q2, expected, implementation):
        gen = torch.Generator().manual_seed(3)
        n = 20000
        z1s = torch.multinomial(
            torch.tensor([0.5, 0.5]), n, replacement=True, generator=gen
        ).tolist()
        z1s = [self.OF if z == 0 else self.NO for z in z1s]
        q_rows = torch.stack([_probs_to_logits(self.Q1), _probs_to_logits(q2)])
        logits = torch.zeros(1, n * 3, 4)
        logits[0, 1::3] = q_rows[0]
        logits[0, 2::3] = q_rows[1]
        targets = torch.zeros(1, n * 3, 4)
        token_ids = torch.zeros(1, n * 3, dtype=torch.long)
        loss_mask = torch.zeros(1, n * 3)
        for b, z1 in enumerate(z1s):
            p2 = self.P2_GIVEN_OF if z1 == self.OF else self.P2_GIVEN_NO
            z2 = self.COURSE if z1 == self.OF else self.PROBLEM
            targets[0, b * 3 + 1] = _probs_to_logits(self.P1)
            targets[0, b * 3 + 2] = _probs_to_logits(p2)
            token_ids[0, b * 3 + 1] = z1
            token_ids[0, b * 3 + 2] = z2
            loss_mask[0, b * 3 + 1] = 1.0
            loss_mask[0, b * 3 + 2] = 1.0
        _, stats = _block_loss(
            implementation, logits, targets, token_ids, loss_mask, block_size=3
        )
        assert stats["tau_hat"].mean().item() == pytest.approx(expected, abs=5e-3)

    @pytest.mark.parametrize("q2", [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    def test_biased_variant_gives_1_9_for_both(self, q2, implementation):
        assert self._biased_product(q2) == pytest.approx(1.9, abs=1e-6)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
class TestGreedyLimit:
    def test_one_hot_target_equals_chain_loss(self, implementation):
        gen = torch.Generator().manual_seed(1)
        vocab, num_drafted = 5, 3
        q = torch.softmax(torch.randn(num_drafted, vocab, generator=gen), -1)
        path = torch.randint(0, vocab, (num_drafted,), generator=gen).tolist()
        target_rows = torch.zeros(num_drafted, vocab)
        target_rows.scatter_(-1, torch.tensor(path).unsqueeze(-1), 100.0)

        _, stats = _run_block(
            torch.log(q.clamp_min(_PROB_FLOOR)),
            target_rows,
            path,
            implementation=implementation,
        )

        survival, chain = 1.0, 1.0
        for k in range(num_drafted):
            survival *= q[k, path[k]].item()
            chain += survival
        assert stats["tau_hat"].item() == pytest.approx(chain, abs=1e-4)

    def test_small_temperature_converges_to_chain_loss(self, implementation):
        """temperature -> 0 turns p one-hot, so tau_hat -> the chain loss."""
        gen = torch.Generator().manual_seed(2)
        vocab, num_drafted = 5, 3
        q = torch.softmax(torch.randn(num_drafted, vocab, generator=gen), -1)
        target_logits = torch.randn(num_drafted, vocab, generator=gen)
        path = target_logits.argmax(dim=-1).tolist()  # greedy data path

        _, stats = _run_block(
            torch.log(q.clamp_min(_PROB_FLOOR)),
            target_logits,
            path,
            temperature=1e-3,
            implementation=implementation,
        )

        survival, chain = 1.0, 1.0
        for k in range(num_drafted):
            survival *= q[k, path[k]].item()
            chain += survival
        assert stats["tau_hat"].item() == pytest.approx(chain, abs=1e-3)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
class TestGradients:
    def _single_block_case(self, q1, p1, q2, p2, z1, z2):
        vocab = len(q1)
        logits = torch.zeros(1, 3, vocab)
        logits[0, 1] = _probs_to_logits(q1)
        logits[0, 2] = _probs_to_logits(q2)
        targets = torch.zeros(1, 3, vocab)
        targets[0, 1] = _probs_to_logits(p1)
        targets[0, 2] = _probs_to_logits(p2)
        token_ids = torch.tensor([[0, z1, z2]])
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])
        return logits, targets, token_ids, loss_mask

    def test_finite_everywhere(self, implementation):
        gen = torch.Generator().manual_seed(4)
        vocab = 4
        logits = torch.randn(1, 3, vocab, generator=gen)
        targets = torch.randn(1, 3, vocab, generator=gen)
        token_ids = torch.randint(0, vocab, (1, 3), generator=gen)
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])
        logits.requires_grad_(True)
        loss, _ = _block_loss(implementation, logits, targets, token_ids, loss_mask, 3)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[0, 1:].abs().sum() > 0

    def test_capped_accept_factor_has_zero_gradient(self, implementation):
        # Slot 1: q(z1) = 0.9 > p(z1) = 0.4, so a_1 is capped at 1 and its
        # gradient must be zero. Reference rebuilds tau_hat with the a_1
        # factor detached; gradients must match (bit-tight for the eager
        # legs; the fused legs differ from the eager rebuild only by fp
        # reduction order).
        q1 = [0.05, 0.05, 0.9]
        p1 = [0.3, 0.3, 0.4]
        q2 = [0.2, 0.5, 0.3]
        p2 = [0.6, 0.2, 0.2]
        z1, z2 = 2, 1

        logits, targets, token_ids, loss_mask = self._single_block_case(
            q1, p1, q2, p2, z1, z2
        )
        logits.requires_grad_(True)
        loss_real, _ = _block_loss(
            implementation, logits, targets, token_ids, loss_mask, 3
        )
        loss_real.backward()
        grad_real = logits.grad.clone()

        # sanity: the ratio really is capped at slot 1
        assert q1[z1] > p1[z1]

        device = logits.device
        logits_ref, targets, token_ids, loss_mask = self._single_block_case(
            q1, p1, q2, p2, z1, z2
        )
        logits_ref = logits_ref.to(device)
        targets = targets.to(device)
        token_ids = token_ids.to(device)
        loss_mask = loss_mask.to(device)
        logits_ref.requires_grad_(True)
        logq = torch.nn.functional.log_softmax(logits_ref[0].float(), dim=-1)
        logp = torch.nn.functional.log_softmax(targets[0].float(), dim=-1)
        log_a1 = (logq[1, z1] - logp[1, z1]).clamp_max(0.0).detach()
        alpha = torch.minimum(logq.exp(), logp.exp()).sum(dim=-1)
        tau_ref = 1.0 + alpha[1] + log_a1.exp() * alpha[2]
        loss_ref = -tau_ref / 2
        loss_ref.backward()

        tol = 1e-7 if implementation == "eager" else 1e-5
        assert torch.allclose(grad_real.to(device), logits_ref.grad, atol=tol)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
class TestMasking:
    def test_block_truncated_at_first_invalid_slot(self, implementation):
        gen = torch.Generator().manual_seed(5)
        vocab = 4
        q_rows = torch.randn(3, vocab, generator=gen)
        t_rows = torch.randn(3, vocab, generator=gen)
        path = torch.randint(0, vocab, (3,), generator=gen).tolist()

        # Full block (slots 1..3 drafted) with slot 3 invalid: sequence ended.
        loss_full, stats_full = _run_block(
            q_rows,
            t_rows,
            path,
            loss_mask=[0.0, 1.0, 1.0, 0.0],
            implementation=implementation,
        )
        # Reference: the same block cut to its valid prefix (slots 1..2).
        loss_ref, stats_ref = _run_block(
            q_rows[:2],
            t_rows[:2],
            path[:2],
            loss_mask=[0.0, 1.0, 1.0],
            implementation=implementation,
        )

        assert stats_full["tau_hat"].item() == pytest.approx(
            stats_ref["tau_hat"].item(), abs=1e-6
        )
        # K is the block's drafted capacity (3 here), shared by truncated and
        # full blocks alike so the loss scale stays comparable.
        assert loss_full.item() == pytest.approx(
            -stats_full["tau_hat"].item() / 3, abs=1e-6
        )
        assert loss_ref.item() == pytest.approx(
            -stats_ref["tau_hat"].item() / 2, abs=1e-6
        )
        assert not stats_full["slot_valid"][0, 2]
        # The cut slot itself is still reachable (survival = a_1 * a_2 > 0)
        # but contributes nothing -- the tau_hat equality above pins that.
        # Slots beyond the cut have survival exactly 0 via the alive mask.
        assert stats_full["survival"][0, 2] > 0.0
        loss_deep, stats_deep = _run_block(
            q_rows,
            t_rows,
            path,
            loss_mask=[0.0, 1.0, 0.0, 0.0],
            implementation=implementation,
        )
        loss_deep_ref, stats_deep_ref = _run_block(
            q_rows[:1],
            t_rows[:1],
            path[:1],
            loss_mask=[0.0, 1.0],
            implementation=implementation,
        )
        assert stats_deep["tau_hat"].item() == pytest.approx(
            stats_deep_ref["tau_hat"].item(), abs=1e-6
        )
        assert stats_deep["survival"][0, 1] > 0.0
        assert stats_deep["survival"][0, 2] == 0.0

    def test_fully_invalid_block_excluded(self, implementation):
        gen = torch.Generator().manual_seed(6)
        vocab = 3
        q_rows = torch.randn(2, vocab, generator=gen)
        t_rows = torch.randn(2, vocab, generator=gen)
        path = torch.randint(0, vocab, (2,), generator=gen).tolist()

        # One dead block (no valid drafted slot) plus one live block.
        block_size = 3
        logits = torch.zeros(1, 2 * block_size, vocab)
        targets = torch.zeros(1, 2 * block_size, vocab)
        logits[0, block_size + 1 :] = q_rows
        targets[0, block_size + 1 :] = t_rows
        token_ids = torch.zeros(1, 2 * block_size, dtype=torch.long)
        token_ids[0, block_size + 1 :] = torch.tensor(path)
        loss_mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

        loss, stats = _block_loss(
            implementation, logits, targets, token_ids, loss_mask, block_size
        )
        loss_ref, stats_ref = _run_block(
            q_rows, t_rows, path, implementation=implementation
        )
        assert loss.item() == pytest.approx(loss_ref.item(), abs=1e-6)
        assert stats["tau_hat"][0] == 1.0  # dead block contributes tau = 1
        assert not stats["block_valid"][0]
        assert stats["block_valid"][1]


class TestRegistry:
    def test_per_position_dispatch_raises(self):
        logits = torch.randn(1, 4, 5)
        with pytest.raises(RuntimeError, match="block-level"):
            tau_rb_loss(logits, logits)
