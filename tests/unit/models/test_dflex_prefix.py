"""CPU tests for multi-real DFlex training, with deterministic aligned teacher rows."""

from argparse import Namespace

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflex import DFlexDraftModel, DFlexSpeculatorConfig
from speculators.models.dflex.prefix import PrefixLayout, sample_prefix_layout
from speculators.models.dflex.prefix_metrics import prefix_metrics
from speculators.train.cli import build_draft_model
from speculators.train.config import TrainConfig
from tests.unit.models.test_dflex_attention_mode import (
    BLOCK,
    SEQ_LEN,
    _dense_mask_mod,
    _tiny_dflex,
    _toy_batch,
    _toy_speculators_config,
)


def configured(mode="causal", full=2, partial=3):
    model = _tiny_dflex(mode, impl="sdpa")
    model.config.num_full_anchors = full
    model.config.num_partial_anchors = partial
    return model


def layout(anchors, lengths, partial):
    return PrefixLayout.from_blocks(anchors, lengths, partial, BLOCK, "cpu")


def batch():
    return _toy_batch(torch.ones(1, SEQ_LEN))


def test_rows_and_objective():
    packed = layout([2, 5, 1], [4, 2, 4], [False, True, True])
    model = configured()
    hs, ids, mask, last, docs = batch()
    mask[0, 7] = 0
    assert packed.input_ids(ids, 0).tolist()[0] == [
        102,
        103,
        104,
        105,
        105,
        106,
        0,
        0,
        0,
        101,
        102,
        103,
        104,
        0,
        0,
        0,
    ]
    assert packed.draft_rows.tolist() == [[5, 6, 7, 8], [12, 13, 14, 15]]
    labels = model._loss_mask_gather_indices(packed.indices)
    assert ids[0, labels].tolist() == [
        103,
        104,
        105,
        106,
        106,
        107,
        108,
        109,
        110,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
    ]
    logits, teacher = model._prefix_forward(packed, hs, ids, last, docs, None)
    assert torch.equal(teacher.argmax(-1), ids[:, labels])
    valid = mask[:, labels]
    assert valid[0].tolist() == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    loss_config = resolve_loss_config('{"ce":0.3,"tv":0.7}', "eager")
    loss, metrics = prefix_metrics(
        model.config, packed, logits, teacher, valid, loss_config, 2.0, False
    )
    values = sum(w * fn(logits, teacher)[0] for fn, w in loss_config.values())
    ar = [0, 1, 2, 3, 4, 9, 10, 11]
    ar_loss = values[ar].sum() / (len(ar) + 1e-5)
    draft = packed.draft_rows
    decay = torch.exp(-torch.arange(4) / 2)
    expected = (values[draft] * valid[0, draft] * decay).sum() / (6 + 1e-5)
    torch.testing.assert_close(loss, ar_loss + expected)
    assert metrics["ar_accuracy_total"] == len(ar)
    assert metrics["partial_accuracy_total"] == 6
    overlap = torch.minimum(logits.softmax(-1), teacher.softmax(-1)).sum(-1)
    torch.testing.assert_close(metrics["ar_overlap_sum"], overlap[0, ar].sum())


@pytest.mark.parametrize("dist", ["uniform", "geometric"])
def test_sampler_bounds_mask_and_empty(dist):
    model = configured(full=50, partial=60)
    model.config.prefix_length_dist = dist
    docs = torch.tensor([[0] * 7 + [1] * 12 + [-1] * 5])
    valid = torch.ones_like(docs)
    valid[:, 4] = 0
    before = torch.get_rng_state()
    p = sample_prefix_layout(model.config, valid, docs, 117)
    again = sample_prefix_layout(model.config, valid, docs, 117)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(p.anchors, again.anchors)
    assert (valid[0, p.anchors] > 0).all()
    assert (docs[0, p.indices + 1] == docs[0, p.anchors[p.block]]).all()
    assert (docs[0, p.indices + 1] >= 0).all()
    assert len(set(p.anchors[p.is_partial].tolist())) == sum(p.histogram)
    assert 0 < p.num_full < 50
    assert 0 < sum(p.histogram) < 60
    empty = sample_prefix_layout(model.config, valid * 0, docs, 117)
    assert empty.offset.numel() == empty.num_full == sum(empty.histogram) == 0
    _, loss, metrics = model(
        *_toy_batch(valid * 0),
        loss_config=resolve_loss_config("ce", "eager"),
        max_anchors=2,
    )
    assert torch.isfinite(loss)
    assert metrics["ar_loss_total"] == 0


@pytest.mark.parametrize("mode", ["causal", "hybrid"])
@pytest.mark.parametrize(("real", "partial"), [(4, False), (2, True), (4, True)])
def test_prefix_visibility_and_perturbations(mode, real, partial):
    model = _tiny_dflex(mode, impl="sdpa", num_layers=2)
    hs, ids, _, last, docs = batch()
    a = 3
    p = layout([a], [real], [partial])
    n = real + (3 if partial else 0)
    pred, nq, nk = create_anchor_block_mask_mod(
        docs[0], SEQ_LEN, p.anchors, BLOCK, attention_mode=mode, prefix_layout=p
    )
    matrix = _dense_mask_mod(pred, nq, nk)
    expected = torch.ones(n, n, dtype=torch.bool).tril()
    if mode == "hybrid":
        expected[real:] = True
    assert torch.equal(matrix[:, SEQ_LEN:], expected)
    assert matrix[:, :a].all()
    assert not matrix[:, a:SEQ_LEN].any()
    base = model._prefix_forward(p, hs, ids, last, docs, None)[0]
    for j in range(1, real):
        changed = ids.clone()
        changed[0, a + j] += 1
        actual = model._prefix_forward(p, hs, changed, last, docs, None)[0]
        assert torch.equal(base[:, :j], actual[:, :j])
    changed = hs.clone()
    changed[:, a:] += 32
    actual = model._prefix_forward(p, changed, ids, last, docs, None)[0]
    assert torch.equal(base, actual)


def test_ragged_blocks_cannot_see_each_other():
    docs = torch.tensor([0] * 8 + [1] * 16)
    p = layout([1, 9], [4, 2], [False, True])
    pred, nq, nk = create_anchor_block_mask_mod(
        docs, 24, p.anchors, BLOCK, attention_mode="hybrid", prefix_layout=p
    )
    matrix = _dense_mask_mod(pred, nq, nk)
    assert not matrix[:4, 28:].any()
    assert not matrix[4:, 24:28].any()
    assert matrix[:4, :1].all()
    assert not matrix[:4, 1:24].any()
    assert matrix[4:, 8:9].all()
    assert not matrix[4:, :8].any()
    assert not matrix[4:, 9:24].any()
    expected = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert torch.equal(matrix[4:, 28:], torch.tensor(expected, dtype=torch.bool))


def test_causal_anchor_row_and_boundary_diagnostic():
    model = configured()
    hs, ids, mask, last, docs = batch()
    mask.zero_()
    mask[:, 3] = 1
    packed = layout([3], [4], [False])
    blind = model._backbone_forward(hs, ids, mask, last, docs, max_anchors=1)[1]
    full = model._prefix_forward(packed, hs, ids, last, docs, None)[0]
    assert torch.equal(blind[:, 0], full[:, 0])
    docs[:, 5:] = 1
    mask[:, 5] = 1
    _, _, stats = model(
        hs,
        ids,
        mask,
        last,
        docs,
        max_anchors=3,
        loss_config=resolve_loss_config("ce", "eager"),
    )
    assert stats["blind_cross_doc_blocks_sum"] == 1
    assert stats["blind_cross_doc_blocks_total"] == 2
    assert (
        stats["blind_cross_doc_tokens_sum"]
        == stats["blind_cross_doc_tokens_total"]
        == 1
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_blind_loss_logits_rng_and_gradient_identity(enabled):
    model = configured(full=2 if enabled else 0, partial=3 if enabled else 0)
    inputs = batch()
    kwargs = {"loss_config": resolve_loss_config("ce", "eager"), "max_anchors": 4}
    captured = []

    def capture(module, args, out):
        out.retain_grad()
        captured.append(out)

    hook = model.lm_head.register_forward_hook(capture)
    try:
        torch.manual_seed(81)
        reference = []
        for _ in range(2):
            model.zero_grad()
            captured.clear()
            _, loss, _ = DFlashDraftModel.forward(model, *inputs, **kwargs)
            loss.backward()
            reference.append(
                (
                    loss.detach(),
                    captured[0].detach(),
                    captured[0].grad.clone(),
                    [p.grad.clone() for p in model.parameters() if p.grad is not None],
                )
            )
        state = torch.get_rng_state()
        torch.manual_seed(81)
        for step, (old, logits, grad, param_grads) in enumerate(reference):
            model.zero_grad()
            captured.clear()
            _, loss, stats = model(*inputs, prefix_step=step, **kwargs)
            loss.backward()
            assert torch.equal(old, stats["blind_loss_sum"])
            assert torch.equal(logits, captured[0])
            assert torch.equal(grad, captured[0].grad)
            if not enabled:
                assert torch.equal(old, loss)
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert all(
                    torch.equal(a, b) for a, b in zip(grads, param_grads, strict=True)
                )
        assert torch.equal(state, torch.get_rng_state())
    finally:
        hook.remove()


def test_config_and_tau_guard(tmp_path):
    cfg = configured().config
    cfg.speculators_config = _toy_speculators_config()["speculators_config"]
    cfg.save_pretrained(str(tmp_path))
    loaded = DFlexSpeculatorConfig.from_pretrained(str(tmp_path))
    assert loaded.num_partial_anchors == 3
    assert loaded.num_full_anchors == 2
    saved = cfg.model_dump()
    with pytest.raises(ValueError, match="causal or hybrid"):
        DFlexSpeculatorConfig(**(saved | {"attention_mode": "bidirectional"}))
    with pytest.raises(ValueError, match="tau_rb"):
        configured()(
            *batch(), loss_config=resolve_loss_config('{"ce":1,"tau_rb":1}', "eager")
        )


def test_trainer_config_and_config_only_reload(tmp_path, monkeypatch):
    cfg = configured().config
    cfg.speculators_config = _toy_speculators_config()["speculators_config"]
    cfg.num_blind_anchors = 9
    cfg.save_pretrained(str(tmp_path))
    args = Namespace(
        **TrainConfig.resolve(
            [
                "--speculator-type",
                "dflex",
                "--from-pretrained",
                str(tmp_path),
                "--verifier-name-or-path",
                "toy",
                "--draft-attn-impl",
                "sdpa",
                "--lambda-ar",
                "0.3",
                "--num-partial-anchors",
                "7",
            ]
        ).flatten()
    )
    monkeypatch.setattr(DFlexDraftModel, "load_verifier_weights", lambda self: None)
    model = build_draft_model(args, DFlexDraftModel, None, None, cfg.draft_vocab_size)
    assert model.config.num_blind_anchors == 9
    assert model.config.num_full_anchors == 2
    assert model.config.num_partial_anchors == 7
    assert model.config.lambda_ar == 0.3
    assert model.config.lambda_prefix == 1
    assert model.config.transformer_layer_config.is_causal


def test_fixed_eval_losses_decrease(record_property):
    model = configured(full=8, partial=8)
    inputs = batch()
    kwargs = {"max_anchors": 16, "loss_config": resolve_loss_config("ce", "eager")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    keys = ("blind_loss", "ar_loss", "draft_partial_loss", "prefix_loss", "total_loss")
    first, last = [], []
    for step in range(200):
        model.train()
        optimizer.zero_grad()
        loss = model(*inputs, prefix_step=step, **kwargs)[1]
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad.isfinite().all() for p in model.parameters() if p.grad is not None
        )
        optimizer.step()
        if step < 5 or step >= 195:
            model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                stats = model(*inputs, prefix_step=0, **kwargs)[2]
                values = [stats[k + "_sum"] / stats[k + "_total"] for k in keys]
                (first if step < 5 else last).append(torch.stack(values))
    early, late = torch.stack(first).mean(0), torch.stack(last).mean(0)
    for name, before, after in zip(keys, early, late, strict=True):
        record_property(name + "_early", float(before))
        record_property(name + "_late", float(after))
    assert (late < early).all()
