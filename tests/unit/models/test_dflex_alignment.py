"""dflex label/target/decay/anchor alignment tests.

dflex is DFlash with next-token (anchor-inclusive) indexing: with the anchor
at sequence position ``a`` and ``B = block_size`` input positions
``[anchor, MASK x (B-1)]``, the output at block position ``i`` predicts the
token at sequence position ``a + i + 1`` — the anchor's own output is the
first draft token, so a block drafts ``B`` tokens. DFlash instead reads
positions ``1..B-1`` (position ``k`` predicts token ``a + k``) and drafts
``B - 1`` tokens.

These tests pin, on toy sequences:

- the exact label vector per block for both models (targets argmax + the
  loss mask gathered at the label positions), including blocks cut short by
  document boundaries / loss masks and the in-bounds guarantee at sequence
  end;
- that the target distribution gathered for every dflex slot is one-hot on
  that slot's label, for every eager loss that consumes the verifier
  distribution;
- the exact position-decay weight vector;
- anchor-sampling bounds: no sampled anchor yields a label outside the valid
  range;
- the DFlash/DSpark loss-mask gather is unchanged by the dflex refactor
  (bit-identity of the base-class path).
"""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses.utils import _EAGER_LOSS_FN_MAP, dflash_loss_decay
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.utils import select_anchors
from speculators.models.dflex import DFlexDraftModel, DFlexSpeculatorConfig

HIDDEN = 32
VOCAB = 160
SEQ_LEN = 24
BLOCK = 4
TOKEN_BASE = 100  # toy token ids 100..123


def _tiny_dflash(sample_from_anchor: bool = False) -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
    )
    model = DFlashDraftModel(config)
    return _one_hot_verifier(model).eval()


def _tiny_dflex() -> DFlexDraftModel:
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlexSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
    )
    model = DFlexDraftModel(config)
    return _one_hot_verifier(model).eval()


def _one_hot_verifier(model):
    """Make the verifier head emit, at position ``t``, a one-hot row on
    ``token[t + 1]`` (the standard next-token semantics of target logits).

    ``verifier_last_hidden_states`` is the t-th basis vector at position t,
    ``verifier_norm`` is the identity scale, and ``verifier_lm_head.weight``
    maps basis vector ``e_t`` to ``e_{token[t+1]}``. The final position has no
    successor and is never read as a dflex/DFlash target source.
    """
    input_ids = _toy_ids()
    weight = torch.zeros(VOCAB, HIDDEN)
    weight[input_ids[0, 1:], torch.arange(SEQ_LEN - 1)] = 1.0
    with torch.no_grad():
        model.verifier_lm_head.weight.copy_(weight)
        torch.nn.init.ones_(model.verifier_norm.weight)
        # the tiny harness never loads verifier-owned weights; re-init the
        # draft modules that default to uninitialized (NaN) tensors so the
        # draft logits are finite
        torch.nn.init.normal_(model.embed_tokens.weight)
        torch.nn.init.normal_(model.lm_head.weight)
    return model


def _toy_ids() -> torch.Tensor:
    return (torch.arange(SEQ_LEN) + TOKEN_BASE).unsqueeze(0)


def _toy_batch(loss_mask: torch.Tensor):
    torch.manual_seed(0)
    hidden_states = torch.randn(1, SEQ_LEN, 2 * HIDDEN)
    # position t carries basis vector e_t
    verifier_last_hidden_states = torch.eye(HIDDEN)[:SEQ_LEN].unsqueeze(0)
    document_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    return (
        hidden_states,
        _toy_ids(),
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
    )


def _mask_from(valid_positions) -> torch.Tensor:
    """Loss mask with 1s exactly on ``valid_positions``.

    Keep position 0 invalid: padded (unsampled) anchor slots live at position
    0, so excluding it keeps real anchors distinguishable. Keep the valid
    pool at least as large as the ``max_anchors`` the test passes so every
    block returned by ``_backbone_forward`` is a real sampled anchor.
    """
    loss_mask = torch.zeros(1, SEQ_LEN)
    loss_mask[0, list(valid_positions)] = 1.0
    return loss_mask


def _run_blocks(model, loss_mask, max_anchors=4):
    """Run ``_backbone_forward`` and return per-block views.

    Returns ``(anchors, target_rows, mask_rows)`` where ``anchors[j]`` holds
    the block's sequence positions ``[a, a+1, ..., a+B-1]`` and the other two
    are the model's targets / aligned loss mask for that block.
    """
    batch = _toy_batch(loss_mask)
    with torch.no_grad():
        _, _, targets, aligned_mask, anchored = model._backbone_forward(
            *batch, max_anchors=max_anchors
        )
    anchors = anchored.reshape(-1, BLOCK)
    assert (anchors[:, 0] > 0).all()  # no padding blocks: pool >= max_anchors
    return (
        anchors,
        targets.reshape(-1, BLOCK, VOCAB),
        aligned_mask.reshape(-1, BLOCK),
    )


class TestLabelAlignment:
    """Exact label vectors and loss masks for dflex vs DFlash."""

    def test_dflex_labels_are_next_tokens(self):
        """dflex slots 0..B-1 predict tokens a+1..a+B, and the loss mask is
        gathered at those same label positions."""
        model = _tiny_dflex()
        loss_mask = _mask_from(range(1, 12))
        anchors, target_rows, mask_rows = _run_blocks(model, loss_mask)
        ids = _toy_ids()[0]
        for block, block_anchor in enumerate(anchors[:, 0]):
            a = block_anchor.item()
            label_positions = torch.arange(a + 1, a + BLOCK + 1)
            assert target_rows[block].argmax(-1).tolist() == (
                ids[label_positions].tolist()
            )
            assert torch.equal(mask_rows[block], loss_mask[0, label_positions])

    def test_dflex_loss_mask_cut_short_by_boundary(self):
        """A label past the last loss-masked position is masked at its own
        label position (not the block position one step earlier)."""
        model = _tiny_dflex()
        # valid run 1..9, invalid labels 10..11, valid again 12..23
        loss_mask = _mask_from(list(range(1, 10)) + list(range(12, SEQ_LEN)))
        anchors, target_rows, mask_rows = _run_blocks(model, loss_mask)
        for block, block_anchor in enumerate(anchors[:, 0]):
            a = block_anchor.item()
            label_positions = torch.arange(a + 1, a + BLOCK + 1)
            # targets stay one-hot on the would-be labels (alignment is
            # positional); only the loss is masked
            assert target_rows[block].argmax(-1).tolist() == (
                _toy_ids()[0][label_positions].tolist()
            )
            assert torch.equal(mask_rows[block], loss_mask[0, label_positions])

    def test_dflex_sequence_end_in_bounds(self):
        """select_anchors leaves room for B labels: even the largest valid
        anchor (T-B-1) keeps every label index within the sequence."""
        model = _tiny_dflex()
        # valid pool 16..23; the tail rule keeps only 16..T-B-1 = 16..19
        loss_mask = _mask_from(range(16, SEQ_LEN))
        anchors, target_rows, mask_rows = _run_blocks(model, loss_mask)
        assert sorted(anchors[:, 0].tolist()) == [16, 17, 18, 19]
        last_block = (anchors[:, 0] == 19).nonzero().item()
        # the last block's labels are the final four tokens of the sequence
        ids = _toy_ids()[0]
        assert target_rows[last_block].argmax(-1).tolist() == ids[-BLOCK:].tolist()
        assert mask_rows[last_block].tolist() == [1.0] * BLOCK

    def test_dflash_labels_unchanged(self):
        """DFlash slots 1..B-1 predict tokens a+1..a+B-1 and slot 0 is
        masked — identical to the pre-dflex behavior."""
        model = _tiny_dflash()
        loss_mask = _mask_from(range(1, 16))
        anchors, target_rows, mask_rows = _run_blocks(model, loss_mask)
        ids = _toy_ids()[0]
        for block, block_anchor in enumerate(anchors[:, 0]):
            a = block_anchor.item()
            label_positions = torch.arange(a + 1, a + BLOCK)  # slots 1..B-1
            assert target_rows[block, 1:].argmax(-1).tolist() == (
                ids[label_positions].tolist()
            )
            expected = torch.cat([torch.zeros(1), loss_mask[0, label_positions]])
            assert torch.equal(mask_rows[block], expected)

    def test_dflash_loss_mask_gather_is_bit_identical(self):
        """The dflex refactor must not change the DFlash/DSpark gather: the
        aligned mask equals the pre-refactor inline expression exactly."""
        for model in (_tiny_dflash(), _tiny_dflash(sample_from_anchor=True)):
            loss_mask = torch.ones(1, SEQ_LEN)
            loss_mask[0, 10:12] = 0.0  # interior invalid run
            batch = _toy_batch(loss_mask)
            with torch.no_grad():
                _, _, _, aligned_mask, anchored = model._backbone_forward(
                    *batch, max_anchors=8
                )
            expected = loss_mask.clone()[:, anchored]
            if not model.config.sample_from_anchor:
                expected[:, ::BLOCK] = 0
            assert torch.equal(aligned_mask, expected)


class TestTargetAlignment:
    """The verifier distribution consumed by every loss is one-hot on the
    dflex slot's label."""

    # tau_rb is excluded: its registry entry is a deliberate stub for the
    # per-position dispatch (a block-level loss, computed inside
    # compute_metrics from the realized token ids) — it gets the dedicated
    # test right below instead of this sweep.
    @pytest.mark.parametrize(
        "loss_name", [n for n in sorted(_EAGER_LOSS_FN_MAP) if n != "tau_rb"]
    )
    def test_every_eager_loss_sees_aligned_targets(self, loss_name):
        model = _tiny_dflex()
        loss_mask = _mask_from(range(1, 12))
        anchors, target_rows, _ = _run_blocks(model, loss_mask)
        ids = _toy_ids()[0]
        a = anchors[0, 0].item()
        expected_labels = ids[a + 1 : a + BLOCK + 1]
        targets_block = target_rows[0]
        # the gathered target rows are one-hot on each slot's label (the
        # verifier head emits basis vectors scaled by the RMS norm; compare
        # row-normalized so the scale drops out)
        one_hot = torch.nn.functional.one_hot(expected_labels, VOCAB).float()
        rows = targets_block / targets_block.max(dim=-1, keepdim=True).values
        torch.testing.assert_close(rows, one_hot, atol=1e-5, rtol=0)
        # a drafter that predicts exactly the labels minimizes every loss:
        # feeding the aligned one-hots as (scaled) logits gives ~0 for every
        # verifier-consuming loss, and strictly less than a mis-aligned
        # (rolled) prediction
        loss_fn = _EAGER_LOSS_FN_MAP[loss_name]
        aligned = (one_hot * 40.0).unsqueeze(0)
        shifted = torch.nn.functional.one_hot(expected_labels.roll(1), VOCAB).float()
        misaligned = (shifted * 40.0).unsqueeze(0)
        assert torch.all(loss_fn(aligned, aligned) < 1e-3)
        # jsd is bounded by log 2, so use a margin every loss clears
        assert torch.all(
            loss_fn(misaligned, aligned) >= loss_fn(aligned, aligned) + 0.1
        )

    def test_tau_rb_sees_next_token_realized_ids(self, monkeypatch):
        """tau_rb is block-level: the per-position registry entry is a stub
        that raises by design, and the block path receives, at every dflex
        slot, the realized token at that slot's LABEL position — slot i's
        ``a + i + 1``, the token a dflex draft actually proposes there (and
        that gets accepted/rejected by the verifier)."""
        import speculators.models.dflash.metrics as dflash_metrics

        captured = {}
        real_tau = dflash_metrics.tau_rb_block_loss

        def spy(logits, targets, token_ids, *args, **kwargs):
            captured["token_ids"] = token_ids
            return real_tau(logits, targets, token_ids, *args, **kwargs)

        monkeypatch.setattr(dflash_metrics, "tau_rb_block_loss", spy)

        model = _tiny_dflex()
        loss_mask = _mask_from(range(1, 12))
        batch = _toy_batch(loss_mask)  # seeds the RNG; anchors reproduce below
        loss_config = {
            "kl_div": (_EAGER_LOSS_FN_MAP["kl_div"], 1.0),
            "tau_rb": (_EAGER_LOSS_FN_MAP["tau_rb"], 1.0),
        }
        _, loss, metrics = model.forward(
            *batch,
            loss_config=loss_config,
            max_anchors=4,
            per_position_loss_weight="none",
            loss_implementation="eager",
        )

        # the stub documents the block-level design
        with pytest.raises(RuntimeError, match="block-level"):
            _EAGER_LOSS_FN_MAP["tau_rb"](
                torch.zeros(1, BLOCK, VOCAB), torch.zeros(1, BLOCK, VOCAB)
            )

        # the block path got the realized ids one past each block position
        # (same seeded batch -> the same sampled anchors)
        anchors, _, _ = _run_blocks(model, loss_mask)
        ids = _toy_ids()[0]
        got = captured["token_ids"][0].reshape(-1, BLOCK)
        assert got.shape == anchors.shape
        for row, a in zip(got, anchors[:, 0]):
            assert row.tolist() == ids[a + 1 : a + BLOCK + 1].tolist()

        assert torch.isfinite(loss)
        assert "tau_rb_loss_sum" in metrics
        assert "tau_rb_tau_hat_sum" in metrics
        assert metrics["tau_rb_loss_total"].item() > 0

    def test_dflex_full_forward_metrics(self):
        """forward() wiring: slot 0 is a real prediction and all per-position
        accuracies are reported for positions 0..B-1.

        max_anchors matches the valid pool so no eager-mode padding-anchor
        rows are produced (fully-masked rows softmax to NaN; the flex path
        used in real training does not run here).
        """
        model = _tiny_dflex()
        loss_mask = torch.ones(1, SEQ_LEN)
        loss_mask[0, 0] = 0.0
        batch = _toy_batch(loss_mask)
        loss_config = {"kl_div": (_EAGER_LOSS_FN_MAP["kl_div"], 1.0)}
        _, loss, metrics = model.forward(
            *batch,
            loss_config=loss_config,
            max_anchors=4,
            per_position_loss_weight="fixed-exp-decay",
        )
        assert torch.isfinite(loss)
        for pos in range(BLOCK):
            assert f"position_{pos}_acc_sum" in metrics
        assert "eal_sum" in metrics


class TestDecayWeights:
    def test_dflex_decay_weights(self):
        """w_i = exp(-i/gamma) with the anchor prediction weighted 1."""
        pos_idx = torch.arange(BLOCK, dtype=torch.float32)
        weights = dflash_loss_decay(pos_idx, gamma=2.0, sample_from_anchor=True)
        expected = torch.exp(-pos_idx / 2.0)
        torch.testing.assert_close(weights, expected, atol=0, rtol=0)
        assert weights[0].item() == 1.0

    def test_dflash_decay_weights_unchanged(self):
        """DFlash keeps slot 0 at weight 0 and decays from slot 1."""
        pos_idx = torch.arange(BLOCK, dtype=torch.float32)
        weights = dflash_loss_decay(pos_idx, gamma=2.0, sample_from_anchor=False)
        expected = torch.exp(-((pos_idx - 1).clamp(min=0)) / 2.0)
        expected = expected * (pos_idx != 0).to(torch.float32)
        torch.testing.assert_close(weights, expected, atol=0, rtol=0)
        assert weights[0].item() == 0.0


class TestAnchorSampling:
    def test_no_anchor_yields_label_outside_valid_range(self):
        """All sampled anchors leave room for B labels inside the sequence:
        a <= T-B-1, hence a+B <= T-1."""
        torch.manual_seed(0)
        for _ in range(20):
            loss_mask = torch.ones(1, SEQ_LEN)
            loss_mask[0, torch.rand(SEQ_LEN) < 0.3] = 0.0
            anchors, anchor_valid = select_anchors(loss_mask, 16, BLOCK)
            valid = anchors[anchor_valid]
            assert (valid <= SEQ_LEN - BLOCK - 1).all()
            assert loss_mask[0, valid].min().item() == 1.0
            assert (valid + BLOCK <= SEQ_LEN - 1).all()  # every label in bounds

    def test_tail_positions_never_anchor(self):
        loss_mask = torch.ones(1, SEQ_LEN)
        anchors, anchor_valid = select_anchors(loss_mask, 64, BLOCK)
        sampled = anchors[anchor_valid]
        # every position except the last block_size is a valid anchor
        assert anchor_valid.sum().item() == SEQ_LEN - BLOCK
        assert sampled.max().item() <= SEQ_LEN - BLOCK - 1


class TestTrainingArgsAndConfig:
    def test_dflex_pins_sample_from_anchor(self):
        config = DFlexSpeculatorConfig(draft_vocab_size=VOCAB, block_size=BLOCK)
        assert config.sample_from_anchor
        assert config.speculators_model_type == "dflex"

    def test_dflex_config_rejects_sample_from_anchor_false(self):
        with pytest.raises(ValueError, match="sample_from_anchor=True"):
            DFlexSpeculatorConfig(
                draft_vocab_size=VOCAB,
                block_size=BLOCK,
                sample_from_anchor=False,
            )

    def test_checkpoint_config_records_indexing(self):
        """The saved config carries both the dflex type and the explicit
        indexing flag, so inference cannot misread it."""
        config = DFlexSpeculatorConfig(draft_vocab_size=VOCAB, block_size=BLOCK)
        dumped = config.to_diff_dict()
        assert dumped["speculators_model_type"] == "dflex"
        assert dumped["sample_from_anchor"] is True

    def test_build_base_config_kwargs_dflex(self):
        verifier_config = Qwen3Config(hidden_size=HIDDEN, num_hidden_layers=36)
        kwargs = DFlexDraftModel._build_base_config_kwargs(
            "dflex",
            verifier_config,
            draft_vocab_size=VOCAB,
            block_size=15,
            verifier_name_or_path="Qwen/Qwen3-4B",
            target_layer_ids=[2, 10, 18, 26, 34],
        )
        assert kwargs["sample_from_anchor"] is True
        # a dflex block of B positions drafts B tokens
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 15
        assert kwargs["speculators_config"].algorithm == "dflex"

    def test_from_training_args_rejects_explicit_false(self):
        with pytest.raises(ValueError, match="sample_from_anchor=True"):
            DFlexDraftModel.from_training_args(
                verifier_config=Qwen3Config(hidden_size=HIDDEN),
                verifier_name_or_path="Qwen/Qwen3-4B",
                draft_vocab_size=VOCAB,
                block_size=BLOCK,
                sample_from_anchor=False,
            )
