"""dflex ``attention_mode``: in-block visibility (bidirectional/causal/hybrid).

Within a draft block every position is either a real token (today only the
anchor at block offset 0) or a mask. Context visibility -- target-context
features strictly before the anchor, never other blocks -- is unchanged in
every mode; only the in-block rule differs:

- ``bidirectional`` (default): every block position attends to every
  position in its block.
- ``causal``: block position ``i`` attends to block positions ``j <= i``.
- ``hybrid``: a real token attends only to real tokens at earlier or equal
  block positions; a mask attends to all real tokens and all masks in its
  block, in both directions.

These tests pin, on hand-written boolean matrices and tiny eager models:

- the exact full [q, kv] mask per mode (single block + context; two packed
  blocks from different documents);
- that ``bidirectional`` is bit-identical to the pre-feature legacy path on
  full-attention layers (mask, logits and loss), while DFlash/DSpark keep
  the legacy rule untouched (their configs never declare the field);
- that ``causal`` reproduces the legacy SWA-causal in-block behaviour and
  that a set mode overrides the legacy rule on sliding-window layers;
- causal independence (perturbing block input at offset j leaves outputs at
  offsets < j unchanged; bidirectional must change them -- the self-check)
  and the block-size extension property (first B outputs of a causal block
  are invariant to B -> B+8);
- hybrid sanity: the anchor is invariant to mask perturbations, each mask
  is sensitive to other masks' perturbations;
- config plumbing: the field round-trips through the checkpoint, the CLI
  derivation defaults dflex to bidirectional, and the vLLM serving-causality
  hint is stamped per mode (absent + warned for hybrid).
"""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import resolve_loss_config
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflex import DFlexDraftModel, DFlexSpeculatorConfig

MODES = ("bidirectional", "causal", "hybrid")

HIDDEN = 32
VOCAB = 160
SEQ_LEN = 24
BLOCK = 4
TOKEN_BASE = 100  # toy token ids 100..123


# ---------------------------------------------------------------------------
# Mask-level: exact hand-written boolean matrices
# ---------------------------------------------------------------------------


def _dense_mask_mod(mask_mod, q_len, kv_len):
    zero = torch.zeros((), dtype=torch.long)
    dense = torch.zeros(q_len, kv_len, dtype=torch.bool)
    for q in range(q_len):
        for kv in range(kv_len):
            dense[q, kv] = bool(mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv)))
    return dense


def _build_mask(document_ids, total_seq_len, anchors, block_size, **kwargs):
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=torch.tensor(anchors),
        block_size=block_size,
        **kwargs,
    )
    return _dense_mask_mod(mask_mod, q_len, kv_len)


# One block (B=4, anchor at 2) plus a single-document context of length 8:
# base columns 0..7, own-block columns 8..11. Every query of the block sees
# exactly context columns {0, 1} (same doc, strictly before the anchor).
_SINGLE_CTX_ROWS = {
    "bidirectional": [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1],
    "causal": [
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1],
    ],
    "hybrid": [
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1],
    ],
}


class TestExactMasks:
    def test_single_block_with_context(self):
        """B=4, anchor 2, one document: full [4, 12] matrix per mode."""
        doc = torch.zeros(8, dtype=torch.long)
        for mode in MODES:
            dense = _build_mask(doc, 8, [2], 4, attention_mode=mode)
            expected = torch.tensor(_SINGLE_CTX_ROWS[mode], dtype=torch.bool)
            if mode == "bidirectional":
                expected = expected.unsqueeze(0).expand(4, -1)
            assert torch.equal(dense, expected), mode

    def test_single_block_create_mask_consumer(self):
        """The sdpa consumer (flex create_mask) materializes the same matrix."""
        doc = torch.zeros(8, dtype=torch.long)
        for mode in MODES:
            mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
                document_ids=doc,
                total_seq_len=8,
                anchor_positions=torch.tensor([2]),
                block_size=4,
                attention_mode=mode,
            )
            dense = create_mask(
                mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=doc.device
            )
            assert torch.equal(
                dense[0, 0].bool(),
                _build_mask(doc, 8, [2], 4, attention_mode=mode),
            ), mode

    def test_two_packed_blocks_two_documents(self):
        """Docs of lengths [5, 3], anchors 3 (doc 0) and 6 (doc 1): full
        [8, 16] matrix per mode. Doc 0's block sees context {0, 1, 2}, doc
        1's sees {5}; neither block sees the other."""
        doc = torch.tensor([0] * 5 + [1] * 3, dtype=torch.long)
        base_b0 = [1, 1, 1, 0, 0, 0, 0, 0]
        base_b1 = [0, 0, 0, 0, 0, 1, 0, 0]
        other_b0 = [0, 0, 0, 0]
        tri = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
        full = [1, 1, 1, 1]
        anchor_row = [1, 0, 0, 0]
        for mode in MODES:
            dense = _build_mask(doc, 8, [3, 6], 4, attention_mode=mode)
            rows = []
            for block, base in ((0, base_b0), (1, base_b1)):
                # block 0's own columns are 8..11 (other block 12..15);
                # block 1's own columns are 12..15 (other block 8..11)
                own = (
                    lambda pattern: pattern + other_b0
                    if block == 0
                    else other_b0 + pattern
                )
                if mode == "bidirectional":
                    rows += [base + own(full)] * 4
                elif mode == "causal":
                    rows += [base + own(tri[i]) for i in range(4)]
                else:  # hybrid
                    rows += [base + own(anchor_row)]
                    rows += [base + own(full)] * 3
            assert torch.equal(dense, torch.tensor(rows, dtype=torch.bool)), mode

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="bidirectional\\|causal\\|hybrid"):
            create_anchor_block_mask_mod(
                document_ids=torch.zeros(4, dtype=torch.long),
                total_seq_len=4,
                anchor_positions=torch.tensor([1]),
                block_size=2,
                attention_mode="bogus",
            )


# ---------------------------------------------------------------------------
# Mask-level: relation to the legacy rule
# ---------------------------------------------------------------------------


class TestLegacyRelation:
    def test_bidirectional_equals_legacy_full_attention(self):
        """Default regression at the mask level: on full-attention layers the
        legacy rule (attention_mode=None) is always non-causal, identical to
        attention_mode='bidirectional'."""
        doc = torch.zeros(12, dtype=torch.long)
        for anchors in ([2], [3, 8]):
            legacy = _build_mask(doc, 12, anchors, 4)  # sliding_window=None
            mode = _build_mask(doc, 12, anchors, 4, attention_mode="bidirectional")
            assert torch.equal(legacy, mode), anchors

    def test_causal_equals_legacy_swa_causal(self):
        """causal reproduces the legacy SWA-causal in-block behaviour (the
        pre-2026-10-23 scratch recipe) -- with the same window, the full
        matrices coincide."""
        doc = torch.zeros(12, dtype=torch.long)
        kwargs = dict(sliding_window=4, sliding_window_non_causal=False)
        legacy = _build_mask(doc, 12, [3, 8], 4, **kwargs)
        mode = _build_mask(
            doc, 12, [3, 8], 4, attention_mode="causal", **kwargs
        )
        assert torch.equal(legacy, mode)

    def test_mode_overrides_legacy_swa_rule(self):
        """A set mode overrides sliding_window_non_causal on SWA layers:
        bidirectional over an swnc=False window opens the block up (and
        matches the full-attention mask's in-block part)."""
        doc = torch.zeros(12, dtype=torch.long)
        swa_causal = _build_mask(
            doc, 12, [6], 4, sliding_window=4, sliding_window_non_causal=False
        )
        swa_bidi = _build_mask(
            doc,
            12,
            [6],
            4,
            sliding_window=4,
            sliding_window_non_causal=False,
            attention_mode="bidirectional",
        )
        assert not torch.equal(swa_causal, swa_bidi)
        # in-block submatrix (columns 12..15) is all-ones under the mode
        assert bool(swa_bidi[:, 12:].all())
        assert not bool(swa_causal[:, 12:].all())


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def _toy_tl_config(num_layers: int = 1, impl: str = "eager") -> Qwen3Config:
    return Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=VOCAB,
        _attn_implementation=impl,  # type: ignore[call-arg]
    )


def _toy_speculators_config() -> dict:
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    return dict(
        speculators_config=SpeculatorsConfig(
            algorithm="dflex",
            default_proposal_method="greedy",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=BLOCK)
            ],
            verifier=VerifierConfig(
                name_or_path="toy", architectures=["Qwen3ForCausalLM"]
            ),
        )
    )


class TestConfigPlumbing:
    def test_default_is_bidirectional_and_dflex_only(self):
        assert (
            DFlexSpeculatorConfig.model_fields["attention_mode"].default
            == "bidirectional"
        )
        # DFlash/DSpark/DFlash2 never declare the field: their configs and
        # checkpoints are untouched, and the shared model reads None.
        assert "attention_mode" not in DFlashSpeculatorConfig.model_fields
        model = DFlashDraftModel(
            DFlashSpeculatorConfig(
                transformer_layer_config=_toy_tl_config(),
                draft_vocab_size=VOCAB,
                block_size=BLOCK,
                aux_hidden_state_layer_ids=[0, 1],
                mask_token_id=0,
            )
        )
        assert model.attention_mode is None

    def test_checkpoint_round_trip(self, tmp_path):
        cfg = DFlexSpeculatorConfig(
            transformer_layer_config=_toy_tl_config(),
            draft_vocab_size=VOCAB,
            block_size=BLOCK,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            attention_mode="causal",
            **_toy_speculators_config(),
        )
        cfg.save_pretrained(str(tmp_path))
        loaded = DFlexSpeculatorConfig.from_pretrained(str(tmp_path))
        assert loaded.attention_mode == "causal"
        model = DFlexDraftModel(loaded)
        assert model.attention_mode == "causal"

    def test_training_args_default_derives_bidirectional(self, tmp_path):
        """_build_base_config_kwargs derives 'bidirectional' for dflex and
        passes the key for dflex configs only."""
        _toy_tl_config().save_pretrained(str(tmp_path))
        base = DFlashDraftModel._build_base_config_kwargs(
            "dflex",
            _toy_tl_config(),
            draft_vocab_size=VOCAB,
            verifier_name_or_path=str(tmp_path),
            target_layer_ids=[0, 1],
            mask_token_id=0,
        )
        assert base["attention_mode"] == "bidirectional"
        dflash_base = DFlashDraftModel._build_base_config_kwargs(
            "dflash",
            _toy_tl_config(),
            draft_vocab_size=VOCAB,
            verifier_name_or_path=str(tmp_path),
            target_layer_ids=[0, 1],
            mask_token_id=0,
        )
        assert "attention_mode" not in dflash_base

    @pytest.mark.parametrize(
        ("mode", "expected_is_causal"),
        [("bidirectional", False), ("causal", True), ("hybrid", None)],
    )
    def test_serving_causal_hint(self, mode, expected_is_causal, caplog):
        cfg = DFlexSpeculatorConfig(
            transformer_layer_config=_toy_tl_config(),
            draft_vocab_size=VOCAB,
            block_size=BLOCK,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            attention_mode=mode,
        )
        if mode == "hybrid":
            # a stale hint (e.g. finetune re-selecting the mode) must not
            # survive an explicit hybrid
            cfg.transformer_layer_config.is_causal = True
        DFlexDraftModel._apply_serving_causal_hint(cfg)
        got = cfg.transformer_layer_config.__dict__.get("is_causal")
        assert got == expected_is_causal, mode
        if mode == "hybrid":
            assert "cannot be served" in caplog.text


# ---------------------------------------------------------------------------
# Model-level: tiny eager dflex
# ---------------------------------------------------------------------------


def _tiny_dflex(attention_mode: str | None = "bidirectional",
                block_size: int = BLOCK,
                impl: str = "eager") -> DFlexDraftModel:
    # ``None`` (the legacy rule, kept verbatim for configs that predate the
    # field) is a model-level state, not a config value: the pydantic field
    # only accepts the three modes. Simulate a field-less config by
    # overriding the resolved attribute after construction.
    torch.manual_seed(1234)  # identical decoder weights across constructions
    config = DFlexSpeculatorConfig(
        transformer_layer_config=_toy_tl_config(impl=impl),
        draft_vocab_size=VOCAB,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        attention_mode="bidirectional" if attention_mode is None else attention_mode,
    )
    model = DFlexDraftModel(config)
    if attention_mode is None:
        model.attention_mode = None
    return _finite_drafter(model).eval()


def _finite_drafter(model):
    """Make every weight finite (torch skips some init) and give the
    verifier head a deterministic next-token map."""
    torch.manual_seed(0)
    with torch.no_grad():
        torch.nn.init.normal_(model.embed_tokens.weight)
        torch.nn.init.normal_(model.lm_head.weight)
        torch.nn.init.ones_(model.verifier_norm.weight)
        ids = _toy_ids()
        weight = torch.zeros(VOCAB, HIDDEN)
        weight[ids[0, 1:], torch.arange(SEQ_LEN - 1)] = 1.0
        model.verifier_lm_head.weight.copy_(weight)
    return model


def _toy_ids() -> torch.Tensor:
    return (torch.arange(SEQ_LEN) + TOKEN_BASE).unsqueeze(0)


def _toy_batch(loss_mask: torch.Tensor):
    torch.manual_seed(0)
    hidden_states = torch.randn(1, SEQ_LEN, 2 * HIDDEN)
    verifier_last_hidden_states = torch.eye(HIDDEN)[:SEQ_LEN].unsqueeze(0)
    document_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    return hidden_states, _toy_ids(), loss_mask, verifier_last_hidden_states, document_ids


def _mask_from(valid_positions) -> torch.Tensor:
    """Sparse loss mask: anchor selection is random, so pin it by making the
    valid pool tiny and max_anchors large -- every call then samples the same
    sorted anchor set."""
    loss_mask = torch.zeros(1, SEQ_LEN)
    loss_mask[0, list(valid_positions)] = 1.0
    return loss_mask


def _run_logits(model, loss_mask, max_anchors: int = 8) -> torch.Tensor:
    with torch.no_grad():
        _, logits, _, _, _ = model._backbone_forward(
            *_toy_batch(loss_mask), max_anchors=max_anchors
        )
    return logits.reshape(-1, model.block_size, VOCAB)


class _OffsetPerturbedEmbed(torch.nn.Module):
    """Adds a fixed delta to the block input at ``offset`` of every block.

    The drafter embeds its block inputs as ``[anchor, MASK x (B-1)]`` per
    block in one flat ``[1, n_anchors * B]`` id tensor, so patching
    ``embed_tokens`` perturbs exactly one input position per block (the
    anchor token at offset 0, the shared mask embedding at offset > 0)
    without touching anything else in the forward.
    """

    def __init__(self, base, block_size: int, offset: int, delta: torch.Tensor):
        super().__init__()
        self.base = base
        self.block_size = block_size
        self.offset = offset
        self.delta = delta

    def forward(self, ids):
        emb = self.base(ids).clone()
        cols = torch.arange(self.offset, ids.shape[1], self.block_size)
        emb[:, cols] += self.delta
        return emb


def _perturbed(model, offset: int) -> DFlexDraftModel:
    torch.manual_seed(7)
    delta = torch.randn(HIDDEN)
    model.embed_tokens = _OffsetPerturbedEmbed(
        model.embed_tokens, model.block_size, offset, delta
    )
    return model


class TestDefaultRegression:
    def test_bidirectional_bit_identical_to_legacy(self):
        """Same weights and inputs: attention_mode='bidirectional' produces
        the same logits AND loss as the pre-feature legacy path (mode=None
        on full-attention layers). One model, attribute toggled between
        runs -- two separately built models would differ in their unseeded
        layer weights."""
        loss_mask = _mask_from({4, 10})
        model = _tiny_dflex("bidirectional")
        mode_logits = _run_logits(model, loss_mask)
        model.attention_mode = None  # the pre-feature state
        legacy_logits = _run_logits(model, loss_mask)
        assert torch.equal(mode_logits, legacy_logits)

        # loss through the public forward, bit-identical as well
        loss_config = resolve_loss_config("ce", "eager")
        model = _tiny_dflex("bidirectional")
        with torch.no_grad():
            _, mode_loss, _ = model(
                *_toy_batch(loss_mask), loss_config=loss_config, max_anchors=8
            )
        model.attention_mode = None
        with torch.no_grad():
            _, legacy_loss, _ = model(
                *_toy_batch(loss_mask), loss_config=loss_config, max_anchors=8
            )
        assert torch.equal(mode_loss, legacy_loss)


class TestCausalBehaviour:
    LOSS_MASK = _mask_from({4, 10})

    def test_perturbation_independence(self):
        """Perturbing the block input at offset j=2 leaves outputs at offsets
        0, 1 unchanged in causal mode (bitwise: the masked key never enters
        their computation), while offsets 2, 3 do change. In bidirectional
        mode every offset changes -- this checks the check itself."""
        clean = _run_logits(_tiny_dflex("causal"), self.LOSS_MASK)
        pert = _run_logits(
            _perturbed(_tiny_dflex("causal"), offset=2), self.LOSS_MASK
        )
        torch.testing.assert_close(pert[:, :2], clean[:, :2], rtol=0, atol=0)
        assert not torch.allclose(pert[:, 2:], clean[:, 2:])

        clean_b = _run_logits(_tiny_dflex("bidirectional"), self.LOSS_MASK)
        pert_b = _run_logits(
            _perturbed(_tiny_dflex("bidirectional"), offset=2), self.LOSS_MASK
        )
        assert not torch.allclose(pert_b[:, :2], clean_b[:, :2])

    def test_block_size_extension(self):
        """Causal blocks are prefix-stable: the first B outputs of a causal
        block of size B equal those of a block of size B+8 (same weights,
        same anchors), to numerical tolerance."""
        small = _tiny_dflex("causal", block_size=BLOCK)
        big = _tiny_dflex("causal", block_size=BLOCK + 8)
        big.load_state_dict(small.state_dict(), strict=True)
        # anchors valid for both block sizes (position < SEQ_LEN - (B + 8))
        loss_mask = _mask_from({4, 10})
        assert SEQ_LEN - (BLOCK + 8) > 10
        small_logits = _run_logits(small, loss_mask)  # [n_blocks, B, V]
        big_logits = _run_logits(big, loss_mask)  # [n_blocks, B+8, V]
        torch.testing.assert_close(
            big_logits[:, :BLOCK], small_logits, rtol=0, atol=1e-6
        )


class TestHybridBehaviour:
    LOSS_MASK = _mask_from({4, 10})

    def test_anchor_invariant_to_mask_perturbation(self):
        """The anchor (offset 0, the only real token) never attends masks:
        perturbing any mask offset leaves its output unchanged."""
        clean = _run_logits(_tiny_dflex("hybrid"), self.LOSS_MASK)
        for offset in (1, 2, 3):
            pert = _run_logits(
                _perturbed(_tiny_dflex("hybrid"), offset=offset), self.LOSS_MASK
            )
            torch.testing.assert_close(pert[:, 0], clean[:, 0], rtol=0, atol=0)

    def test_masks_sensitive_to_other_masks(self):
        """Each mask attends every other mask: perturbing one mask position
        changes the outputs at the other mask positions."""
        clean = _run_logits(_tiny_dflex("hybrid"), self.LOSS_MASK)
        pert = _run_logits(
            _perturbed(_tiny_dflex("hybrid"), offset=1), self.LOSS_MASK
        )
        for offset in (2, 3):
            assert not torch.allclose(pert[:, offset], clean[:, offset])


class TestSmokeTraining:
    STEPS = 200

    @pytest.mark.parametrize("mode", MODES)
    def test_loss_decreases(self, mode):
        """A few hundred optimizer steps on a tiny dflex (frozen verifier
        head, one-hot next-token targets, fixed batch): the training loss
        decreases in every mode. CPU + sdpa dense masks -- the flex kernel
        consumes the same mask_mod (pinned by the exact-mask tests), and the
        production recipe is smoke-tested by the training script."""
        torch.manual_seed(0)
        model = _tiny_dflex(mode, impl="sdpa")
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "tiny dflex must expose trainable parameters"
        before = [p.detach().clone() for p in trainable]
        opt = torch.optim.AdamW(trainable, lr=3e-3)
        loss_config = resolve_loss_config("ce", "eager")
        # anchor pool 1..15 with max_anchors 16: every step samples the same
        # sorted anchor set, so losses are comparable across steps
        loss_mask = _mask_from(range(1, 16))
        batch = _toy_batch(loss_mask)

        first = last = None
        for _ in range(self.STEPS):
            opt.zero_grad()
            _, loss, _ = model(*batch, loss_config=loss_config, max_anchors=16)
            assert torch.isfinite(loss), mode
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
            last = loss.item()
        assert last < 0.75 * first, (mode, first, last)
        changed = any(
            not torch.equal(p.detach(), b) for p, b in zip(trainable, before)
        )
        assert changed, mode
