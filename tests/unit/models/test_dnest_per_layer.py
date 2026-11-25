"""Unit tests for dnest per-layer target context (context_mode="per_layer").

Maps to the per-layer context test plan:
1. index convention -- the extracted tap s_j is the actual input of target
   layer s_j (pre-forward hook on a real Qwen3 module),
2. exact context keys/values -- frozen modes reproduce the target's own
   cache at layer s_j (input_layernorm -> k/v_proj -> k_norm -> RoPE),
3. freezing -- frozen tensors stay bit-identical through optimizer steps,
   trainable ones move, frozen ones carry no optimizer state,
4. split separation -- context and block paths start equal, then diverge,
5. learning-rate groups -- per kv_mode each group holds exactly the
   intended parameters at the intended learning rate,
6. validation -- frozen modes with random init, inconsistent pairing, and
   misplaced per-layer knobs each raise a clear error,
7. fused regression -- context_mode="fused" stays bit-identical to the
   pre-per-layer dnest (golden captured on unmodified main).
"""

import json
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from speculators.losses import resolve_loss_config
from speculators.models.dnest import DNestDraftModel, DNestSpeculatorConfig
from speculators.models.dnest.model_definitions import CONTEXT_KV_SUFFIXES
from speculators.models.dnest.weight_copy import (
    BLOCK_KV_SUFFIXES,
    COPIED_LAYER_SUFFIXES,
    KV_PATH_SUFFIXES,
)
from speculators.train.data import check_tap_layers_manifest
from speculators.train.optimizers import (
    build_optimizers,
    split_by_lr_scale,
    split_named_params_for_muon,
)

from .generate_dnest_fused_golden import (
    DRAFT_KWARGS,
    TARGET_KWARGS,
    build_batch,
    build_model as build_golden_model,
    build_target,
    run_model as run_golden_model,
)

SOURCES = [1, 3]  # taps (1, 3) -> consumers are 0-based layers 1 and 3


@pytest.fixture(scope="module")
def target_dir(tmp_path_factory):
    torch.manual_seed(0)
    target = Qwen3ForCausalLM(Qwen3Config(**TARGET_KWARGS))
    target.eval()
    path = tmp_path_factory.mktemp("dnest_pl_target")
    target.save_pretrained(path, max_shard_size="200kB")
    return target, path


def _build_per_layer(
    target_path,
    *,
    kv_mode="frozen_shared",
    init_mode="target_after_extracted",
    seed=123,
    **extra,
):
    torch.manual_seed(seed)
    kwargs = dict(
        verifier_name_or_path=str(target_path),
        draft_vocab_size=256,
        block_size=4,
        target_layer_ids=list(SOURCES),
        mask_token_id=255,
        draft_attn_impl="eager",
        init_mode=init_mode,
        context_mode="per_layer",
        kv_mode=kv_mode,
    )
    kwargs.update(extra)
    return DNestDraftModel.from_training_args(
        verifier_config=Qwen3Config(**DRAFT_KWARGS), **kwargs
    )


def _target_forward(target, seq_len=16):
    """A real Qwen3 forward with hidden states and KV cache captured."""
    torch.manual_seed(5)
    ids = torch.randint(0, TARGET_KWARGS["vocab_size"], (1, seq_len))
    with torch.no_grad():
        return ids, target(
            ids, use_cache=True, output_hidden_states=True
        )


# --- 1. index convention ----------------------------------------------------


def test_tap_is_the_input_of_its_consumer_layer(target_dir):
    """Tuple tap s equals the actual input of 0-based decoder layer s.

    A pre-forward hook on the real target module captures what layer s
    receives; the hidden_states tuple entry with index s must match it
    exactly. This is the invariant per_layer conditioning relies on.
    """
    target, _ = target_dir
    for s in SOURCES:
        captured = {}

        def hook(module, args, kwargs, s=s, captured=captured):
            captured["input"] = (
                kwargs.get("hidden_states")
                if kwargs.get("hidden_states") is not None
                else args[0]
            )
            return None

        handle = target.model.layers[s].register_forward_pre_hook(
            hook, with_kwargs=True
        )
        try:
            _, out = _target_forward(target)
        finally:
            handle.remove()
        assert torch.equal(out.hidden_states[s], captured["input"]), (
            f"hidden_states[{s}] is not the input of decoder layer {s}"
        )


# --- 2. exact context keys/values -------------------------------------------


@pytest.mark.parametrize(
    "kv_mode", ["frozen_shared", "split", "split_trainable", "trainable_shared"]
)
def test_context_kv_equals_target_cache(target_dir, kv_mode):
    """The per-layer context K/V path reproduces the target's own cache.

    For every draft layer j (source s_j): context K == target cache keys at
    layer s_j (post k_norm, post RoPE) and context V == target cache values,
    within float tolerance; the frozen modes hold exactly (copied weights),
    the trainable modes (split_trainable, trainable_shared) hold at
    initialisation. RoPE is applied at the context tokens' absolute
    positions on both sides.
    """
    target, path = target_dir
    ids, out = _target_forward(target)
    seq_len = ids.shape[1]

    model = _build_per_layer(path, kv_mode=kv_mode)
    model.eval()

    from speculators.models.dflash.model_definitions import (
        apply_rotary_pos_emb,
    )

    for j, s in enumerate(SOURCES):
        layer = model.layers[j]
        ctx_kv = getattr(layer.self_attn, "context_kv", None)
        norm = ctx_kv.input_layernorm if ctx_kv is not None else layer.input_layernorm
        k_proj = ctx_kv.k_proj if ctx_kv is not None else layer.self_attn.k_proj
        v_proj = ctx_kv.v_proj if ctx_kv is not None else layer.self_attn.v_proj
        k_norm = ctx_kv.k_norm if ctx_kv is not None else layer.self_attn.k_norm

        h_j = out.hidden_states[s]  # the extracted tap
        with torch.no_grad():
            k = k_proj(norm(h_j)).view(1, seq_len, -1, layer.self_attn.head_dim)
            k = k_norm(k).transpose(1, 2)
            v = v_proj(norm(h_j)).view(1, seq_len, -1, layer.self_attn.head_dim)
            v = v.transpose(1, 2)
            pos = torch.arange(seq_len).unsqueeze(0)
            cos, sin = model.rotary_emb(h_j, pos)
            _, k_rope = apply_rotary_pos_emb(torch.zeros_like(k), k, cos, sin)

        tgt_k = out.past_key_values.layers[s].keys
        tgt_v = out.past_key_values.layers[s].values
        torch.testing.assert_close(k_rope, tgt_k, rtol=0, atol=1e-5)
        torch.testing.assert_close(v, tgt_v, rtol=0, atol=1e-5)


def test_context_kv_integration_through_full_forward(target_dir):
    """The full training forward routes h_j through layer j's context path.

    Monkeypatching the attention's key-norm hook records the concatenated
    keys; their context part must equal the target's cache keys at s_j (the
    same computation as above, but end-to-end through _backbone_forward's
    slicing and the layer's input_layernorm).
    """
    target, path = target_dir
    ids, out = _target_forward(target)
    seq_len = ids.shape[1]
    concat = torch.cat([out.hidden_states[s] for s in SOURCES], dim=-1)

    model = _build_per_layer(path, kv_mode="frozen_shared")
    model.eval()

    recorded = {}
    for j, layer in enumerate(model.layers):
        original = layer.self_attn._norm_context_and_block_keys

        def wrap(k_ctx, k_noise, bsz, ctx_len, q_len, original=original, j=j):
            result = original(k_ctx, k_noise, bsz, ctx_len, q_len)
            recorded[j] = result.detach().clone()
            return result

        layer.self_attn._norm_context_and_block_keys = wrap
    try:
        with torch.no_grad():
            model._backbone_forward(  # noqa: SLF001
                hidden_states=concat,
                input_ids=ids,
                loss_mask=torch.zeros(1, seq_len),
                verifier_last_hidden_states=out.hidden_states[-1],
                document_ids=torch.zeros(1, seq_len, dtype=torch.long),
                max_anchors=4,
            )
    finally:
        for layer in model.layers:
            del layer.self_attn._norm_context_and_block_keys

    for j, s in enumerate(SOURCES):
        # context part = first seq_len positions, pre-RoPE; apply the draft's
        # own RoPE at the base positions to compare with the target cache
        # recorded[j]: [bsz, ctx+q, H_kv, D] -> [H_kv, L, D]
        k_normed = recorded[j][0, :seq_len].permute(1, 0, 2)
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = model.rotary_emb(recorded[j], pos)
        from speculators.models.dflash.model_definitions import (
            apply_rotary_pos_emb,
        )

        _, k_rope = apply_rotary_pos_emb(torch.zeros_like(k_normed), k_normed, cos, sin)
        tgt_k = out.past_key_values.layers[s].keys
        torch.testing.assert_close(k_rope, tgt_k, rtol=0, atol=1e-5)


# --- 3. freezing ------------------------------------------------------------


@pytest.mark.parametrize("kv_mode", ["frozen_shared", "split"])
def test_frozen_tensors_never_move(target_dir, kv_mode):
    """Several optimizer steps: frozen tensors bit-identical, trainable
    tensors changed, frozen tensors absent from the optimizer state."""
    _, path = target_dir
    model = _build_per_layer(path, kv_mode=kv_mode)
    model.train()

    if kv_mode == "frozen_shared":
        frozen_names = {
            f"layers.{j}.{suffix}"
            for j in range(2)
            for suffix in KV_PATH_SUFFIXES
        }
    else:
        frozen_names = {
            f"layers.{j}.self_attn.context_kv.{suffix}"
            for j in range(2)
            for suffix in CONTEXT_KV_SUFFIXES
        }
    frozen_before = {
        name: model.get_parameter(name).detach().clone() for name in frozen_names
    }
    trainable_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and name not in frozen_names
    }

    assert all(
        not model.get_parameter(name).requires_grad for name in frozen_names
    )

    config = SimpleNamespace(optimizer="adamw", lr=1e-3, weight_decay=0.0)
    optimizer = build_optimizers(model, config)[0]

    torch.manual_seed(9)
    batch = build_batch()
    batch["hidden_states"] = torch.randn(1, 24, 2 * 64)
    batch["verifier_last_hidden_states"] = torch.randn(1, 24, 64)
    loss_config = resolve_loss_config("ce", "eager")
    hidden, logits, targets, aligned_loss_mask, block_idx = (
        model._backbone_forward(max_anchors=8, **batch)  # noqa: SLF001
    )
    from speculators.models.dflash.metrics import compute_metrics

    loss, _ = compute_metrics(
        logits, targets, aligned_loss_mask, model.block_size,
        gamma=4.0, loss_config=loss_config,
        per_position_loss_weight="fixed-exp-decay", dpace_alpha=0.5,
        sample_from_anchor=model.config.sample_from_anchor,
        token_ids=batch["input_ids"][:, block_idx],
        target_temperature=1.0, t2d=None, loss_implementation="eager",
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for name, before in frozen_before.items():
        after = model.get_parameter(name).detach()
        assert torch.equal(before, after), f"{name} moved despite being frozen"
    moved = [
        name
        for name, before in trainable_before.items()
        if not torch.equal(before, model.get_parameter(name).detach())
    ]
    assert moved, "no trainable tensor changed"
    # frozen tensors are absent from the optimizer: no state, not held
    opt_params = {
        id(p) for g in optimizer.param_groups for p in g["params"]
    }
    for name in frozen_names:
        assert id(model.get_parameter(name)) not in opt_params
        assert name not in getattr(optimizer, "state", {}) or not optimizer.state
    state_param_ids = {id(p) for p in optimizer.state}
    for name in frozen_names:
        assert id(model.get_parameter(name)) not in state_param_ids


@pytest.mark.parametrize(
    "kv_mode,init_mode",
    [
        ("split", "target_after_extracted"),
        ("split_trainable", "target_after_extracted"),
        ("split_trainable", "random"),
    ],
)
def test_per_layer_roundtrip_preserves_config_weights_and_freezing(
    target_dir, tmp_path, kv_mode, init_mode
):
    """save_pretrained -> from_pretrained: the per-layer config round-trips,
    the context tensors are bit-identical, re-frozen exactly in split mode
    (and trainable after load in split_trainable), and the optimizer LR
    groups are rebuilt identically (resume identity)."""
    from speculators.model import SpeculatorModel

    _, path = target_dir
    model = _build_per_layer(
        path,
        kv_mode=kv_mode,
        init_mode=init_mode,
        lr_scale_new=0.5,
        lr_scale_copied=0.1,
        lr_scale_block_kv=0.2,
        **({"lr_scale_context_kv": 0.15} if kv_mode == "split_trainable" else {}),
    )
    ctx_before = {
        f"layers.{j}.self_attn.context_kv.{suffix}"
        for j in range(2)
        for suffix in CONTEXT_KV_SUFFIXES
    }
    weights_before = {
        name: model.get_parameter(name).detach().clone() for name in ctx_before
    }
    scales_before = model.optimizer_param_lr_scales()

    save_dir = tmp_path / f"perlayer-roundtrip-{kv_mode}"
    model.save_pretrained(save_dir)
    reloaded = SpeculatorModel.from_pretrained(str(save_dir))

    assert reloaded.config.context_mode == "per_layer"
    assert reloaded.config.kv_mode == kv_mode
    assert reloaded.config.lr_scale_block_kv == 0.2
    for name, before in weights_before.items():
        after = reloaded.get_parameter(name).detach()
        assert torch.equal(before, after), f"{name} changed across save/load"
        if kv_mode == "split":
            assert not reloaded.get_parameter(name).requires_grad, (
                f"{name} not re-frozen after load"
            )
        else:
            assert reloaded.get_parameter(name).requires_grad, (
                f"{name} frozen after load in split_trainable mode"
            )
    assert reloaded.optimizer_param_lr_scales() == scales_before


# --- 4. split separation ----------------------------------------------------


@pytest.mark.parametrize("kv_mode", ["split", "split_trainable"])
def test_split_paths_start_equal_then_diverge(target_dir, kv_mode):
    """split family: context and block paths start as the same target copy;
    after steps they diverge -- the block path always moves, the context
    path only in split_trainable (frozen in split)."""
    _, path = target_dir
    model = _build_per_layer(path, kv_mode=kv_mode)
    model.train()

    for j in range(2):
        ctx = model.layers[j].self_attn.context_kv
        assert torch.equal(
            ctx.k_proj.weight, model.layers[j].self_attn.k_proj.weight
        )
        assert torch.equal(
            ctx.v_proj.weight, model.layers[j].self_attn.v_proj.weight
        )
        assert torch.equal(
            ctx.k_norm.weight, model.layers[j].self_attn.k_norm.weight
        )
        assert torch.equal(
            ctx.input_layernorm.weight,
            model.layers[j].input_layernorm.weight,
        )

    config = SimpleNamespace(optimizer="adamw", lr=1e-2, weight_decay=0.0)
    optimizer = build_optimizers(model, config)[0]
    torch.manual_seed(9)
    batch = build_batch()
    batch["hidden_states"] = torch.randn(1, 24, 2 * 64)
    batch["verifier_last_hidden_states"] = torch.randn(1, 24, 64)
    ctx_before = {
        f"layers.{j}.self_attn.context_kv.{s}": model.get_parameter(
            f"layers.{j}.self_attn.context_kv.{s}"
        ).detach().clone()
        for j in range(2)
        for s in CONTEXT_KV_SUFFIXES
    }
    blk_before = {
        f"layers.{j}.self_attn.{s}": model.get_parameter(
            f"layers.{j}.self_attn.{s}"
        ).detach().clone()
        for j in range(2)
        for s in ("k_proj.weight", "v_proj.weight", "k_norm.weight")
    }
    for _ in range(3):
        hidden, logits, targets, aligned_loss_mask, block_idx = (
            model._backbone_forward(max_anchors=8, **batch)  # noqa: SLF001
        )
        from speculators.models.dflash.metrics import compute_metrics

        loss, _ = compute_metrics(
            logits, targets, aligned_loss_mask, model.block_size,
            gamma=4.0, loss_config=resolve_loss_config("ce", "eager"),
            per_position_loss_weight="fixed-exp-decay", dpace_alpha=0.5,
            sample_from_anchor=model.config.sample_from_anchor,
            token_ids=batch["input_ids"][:, block_idx],
            target_temperature=1.0, t2d=None, loss_implementation="eager",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for name, before in ctx_before.items():
        if kv_mode == "split":
            assert torch.equal(
                before, model.get_parameter(name).detach()
            ), f"{name} (context path) moved in split mode"
    ctx_moved = [
        name
        for name, before in ctx_before.items()
        if not torch.equal(before, model.get_parameter(name).detach())
    ]
    if kv_mode == "split_trainable":
        assert ctx_moved, "no context-path tensor changed in split_trainable"
    else:
        assert not ctx_moved
    moved = [
        name
        for name, before in blk_before.items()
        if not torch.equal(before, model.get_parameter(name).detach())
    ]
    assert moved, "no block-path tensor changed in split mode"


def test_random_split_trainable_context_moves(target_dir):
    """random init: the context path has no target copy to pin to -- it
    must be trainable and actually move under optimizer steps, with only
    the verifier-shared tensors frozen."""
    _, path = target_dir
    model = _build_per_layer(path, kv_mode="split_trainable", init_mode="random")
    model.train()

    ctx_names = [
        f"layers.{j}.self_attn.context_kv.{s}"
        for j in range(2)
        for s in CONTEXT_KV_SUFFIXES
    ]
    assert all(model.get_parameter(n).requires_grad for n in ctx_names)
    frozen = [
        n for n, p in model.named_parameters() if not p.requires_grad
    ]
    assert sorted(frozen) == [
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]
    ctx_before = {
        n: model.get_parameter(n).detach().clone() for n in ctx_names
    }

    config = SimpleNamespace(optimizer="adamw", lr=1e-2, weight_decay=0.0)
    optimizer = build_optimizers(model, config)[0]
    torch.manual_seed(9)
    batch = build_batch()
    batch["hidden_states"] = torch.randn(1, 24, 2 * 64)
    batch["verifier_last_hidden_states"] = torch.randn(1, 24, 64)
    from speculators.models.dflash.metrics import compute_metrics

    for _ in range(3):
        hidden, logits, targets, aligned_loss_mask, block_idx = (
            model._backbone_forward(max_anchors=8, **batch)  # noqa: SLF001
        )
        loss, _ = compute_metrics(
            logits, targets, aligned_loss_mask, model.block_size,
            gamma=4.0, loss_config=resolve_loss_config("ce", "eager"),
            per_position_loss_weight="fixed-exp-decay", dpace_alpha=0.5,
            sample_from_anchor=model.config.sample_from_anchor,
            token_ids=batch["input_ids"][:, block_idx],
            target_temperature=1.0, t2d=None, loss_implementation="eager",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    moved = [
        n
        for n, before in ctx_before.items()
        if not torch.equal(before, model.get_parameter(n).detach())
    ]
    assert moved, "no context-path tensor moved in random split_trainable"


# --- 5. learning-rate groups ------------------------------------------------


def _lr_groups(model):
    """(base names, {scale: names}) straight from the config-derived scales."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    scales = model.optimizer_param_lr_scales()
    scaled_names = {n for n, s in scales.items() if s != 1.0}
    base = trainable - scaled_names
    by_scale: dict[float, set[str]] = {}
    for n, s in scales.items():
        if s != 1.0:
            by_scale.setdefault(s, set()).add(n)
    return base, by_scale


def test_lr_groups_frozen_shared(target_dir):
    _, path = target_dir
    model = _build_per_layer(
        path, kv_mode="frozen_shared", lr_scale_copied=0.5, lr_scale_new=0.25
    )
    base, scaled = _lr_groups(model)
    copied_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in COPIED_LAYER_SUFFIXES
        if s not in KV_PATH_SUFFIXES
    }
    assert set(scaled) == {0.5, 0.25}
    assert scaled[0.5] == copied_expected
    # the kv path is frozen (absent from trainable); the draft final norm is
    # new (init_final_norm=False), so it carries the new scale
    assert scaled[0.25] == {"norm.weight"}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(
        f"layers.{j}.{s}" not in trainable
        for j in range(2)
        for s in KV_PATH_SUFFIXES
    )
    # layers + norm are the only trainable tensors and all are scaled
    assert base == set()
    assert trainable == copied_expected | {"norm.weight"}


def test_lr_groups_split(target_dir):
    _, path = target_dir
    model = _build_per_layer(
        path,
        kv_mode="split",
        lr_scale_copied=0.5,
        lr_scale_block_kv=0.2,
        lr_scale_new=0.25,
    )
    base, scaled = _lr_groups(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    block_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in BLOCK_KV_SUFFIXES
    }
    copied_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in COPIED_LAYER_SUFFIXES
        if s not in BLOCK_KV_SUFFIXES
    }
    assert set(scaled) == {0.2, 0.5, 0.25}
    assert scaled[0.2] == block_expected
    assert scaled[0.5] == copied_expected
    assert scaled[0.25] == {"norm.weight"}
    assert base == set()
    assert trainable == block_expected | copied_expected | {"norm.weight"}
    # context path frozen: not trainable, not scaled
    assert all(
        f"layers.{j}.self_attn.context_kv.{s}" not in trainable
        for j in range(2)
        for s in CONTEXT_KV_SUFFIXES
    )


def test_lr_groups_split_trainable(target_dir):
    """split_trainable adds the trainable context copy: the context_kv
    tensors carry lr_scale_context_kv, the block path lr_scale_block_kv,
    and the block path's input_layernorm stays in the copied group."""
    _, path = target_dir
    model = _build_per_layer(
        path,
        kv_mode="split_trainable",
        lr_scale_copied=0.5,
        lr_scale_block_kv=0.2,
        lr_scale_context_kv=0.1,
        lr_scale_new=0.25,
    )
    base, scaled = _lr_groups(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    ctx_expected = {
        f"layers.{j}.self_attn.context_kv.{s}"
        for j in range(2)
        for s in CONTEXT_KV_SUFFIXES
    }
    block_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in BLOCK_KV_SUFFIXES
    }
    # the block path's input_layernorm is copied-group, not block_kv
    copied_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in COPIED_LAYER_SUFFIXES
        if s not in BLOCK_KV_SUFFIXES
    }
    assert set(scaled) == {0.1, 0.2, 0.5, 0.25}
    assert scaled[0.1] == ctx_expected
    assert scaled[0.2] == block_expected
    assert scaled[0.5] == copied_expected
    assert scaled[0.25] == {"norm.weight"}
    assert base == set()
    assert trainable == ctx_expected | block_expected | copied_expected | {
        "norm.weight"
    }


def test_lr_groups_trainable_shared(target_dir):
    _, path = target_dir
    model = _build_per_layer(
        path,
        kv_mode="trainable_shared",
        lr_scale_copied=0.5,
        lr_scale_context_kv=0.1,
        lr_scale_new=0.25,
    )
    base, scaled = _lr_groups(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    kv_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in KV_PATH_SUFFIXES
    }
    copied_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in COPIED_LAYER_SUFFIXES
        if s not in KV_PATH_SUFFIXES
    }
    assert set(scaled) == {0.1, 0.5, 0.25}
    assert scaled[0.1] == kv_expected
    assert scaled[0.5] == copied_expected
    assert scaled[0.25] == {"norm.weight"}
    assert base == set()
    assert trainable == kv_expected | copied_expected | {"norm.weight"}


def test_lr_groups_random_baseline_all_new(target_dir):
    _, path = target_dir
    model = _build_per_layer(
        path, kv_mode="trainable_shared", init_mode="random", lr_scale_new=0.3
    )
    base, scaled = _lr_groups(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    new_expected = trainable  # every trainable parameter is new in random mode
    assert set(scaled) == {0.3}
    assert scaled[0.3] == new_expected
    assert base == set()


def test_lr_groups_random_split_trainable(target_dir):
    """random + split_trainable: everything is new except the context path,
    which keeps its role scale -- one knob, one meaning across init modes."""
    _, path = target_dir
    model = _build_per_layer(
        path,
        kv_mode="split_trainable",
        init_mode="random",
        lr_scale_new=0.3,
        lr_scale_context_kv=0.15,
    )
    base, scaled = _lr_groups(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    ctx_expected = {
        f"layers.{j}.self_attn.context_kv.{s}"
        for j in range(2)
        for s in CONTEXT_KV_SUFFIXES
    }
    new_expected = {
        f"layers.{j}.{s}"
        for j in range(2)
        for s in COPIED_LAYER_SUFFIXES
    } | {"norm.weight"}
    assert set(scaled) == {0.3, 0.15}
    assert scaled[0.15] == ctx_expected
    assert scaled[0.3] == new_expected
    assert base == set()
    assert trainable == ctx_expected | new_expected


@pytest.mark.parametrize("kv_mode", ["split", "split_trainable"])
def test_split_by_lr_scale_partitions_trainable(target_dir, kv_mode):
    """The optimizer-side partition agrees with the config-derived scales:
    every trainable tensor lands in exactly its scale's group, nothing is
    lost or duplicated, and base holds only tensors without a scale."""
    _, path = target_dir
    model = _build_per_layer(
        path,
        kv_mode=kv_mode,
        lr_scale_copied=0.5,
        lr_scale_block_kv=0.2,
        lr_scale_new=0.25,
        **({"lr_scale_context_kv": 0.1} if kv_mode == "split_trainable" else {}),
    )
    scales = model.optimizer_param_lr_scales()
    base, scaled = split_by_lr_scale(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad],
        scales,
    )
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    grouped = [p for params in scaled.values() for p in params]
    assert len(base) + len(grouped) == len(name_by_id) - sum(
        not p.requires_grad for p in model.parameters()
    )
    assert len({id(p) for p in grouped}) == len(grouped)  # no duplicates
    for scale, params in scaled.items():
        for p in params:
            assert scales[name_by_id[id(p)]] == scale
    for p in base:
        assert name_by_id[id(p)] not in scales
    # in the split family every trainable tensor has a scale: base is empty
    assert base == []


def test_muon_split_routes_frozen_out_of_both_optimizers(target_dir):
    """The Muon path also excludes frozen tensors from Muon and AdamW."""
    _, path = target_dir
    model = _build_per_layer(path, kv_mode="frozen_shared")
    muon_params, adamw_params = split_named_params_for_muon(model)
    held = {name for name, _ in [*muon_params, *adamw_params]}
    frozen = {
        f"layers.{j}.{s}" for j in range(2) for s in KV_PATH_SUFFIXES
    }
    assert not (held & frozen)
    assert held == {n for n, p in model.named_parameters() if p.requires_grad}


# --- 6. validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        # frozen context modes need target weights (trainable ones accept
        # random: frozen_shared/split + random -> error)
        (
            dict(context_mode="per_layer", kv_mode="frozen_shared",
                 init_mode="random"),
            "requires a target-derived",
        ),
        (
            dict(context_mode="per_layer", kv_mode="split", init_mode="random"),
            "requires a target-derived",
        ),
        # per-layer knobs rejected in fused mode (by value)
        (
            dict(context_mode="fused", kv_mode="split"),
            "only apply with",
        ),
        (
            dict(context_mode="fused", kv_mode="split_trainable"),
            "only apply with",
        ),
        (
            dict(context_mode="fused", lr_scale_new=0.5),
            "only apply with",
        ),
        (
            dict(context_mode="fused", lr_scale_context_kv=0.5),
            "only apply with",
        ),
        (
            dict(context_mode="fused", lr_scale_block_kv=0.5),
            "only apply with",
        ),
        # scale/kv-mode consistency
        (
            dict(context_mode="per_layer", kv_mode="frozen_shared",
                 lr_scale_context_kv=0.5),
            "lr_scale_context_kv",
        ),
        (
            dict(context_mode="per_layer", kv_mode="split",
                 lr_scale_context_kv=0.5),
            "lr_scale_context_kv",
        ),
        (
            dict(context_mode="per_layer", kv_mode="trainable_shared",
                 lr_scale_block_kv=0.5),
            "lr_scale_block_kv",
        ),
        # no fusion to initialise in per_layer mode
        (
            dict(context_mode="per_layer", fusion_init="passthrough_deepest"),
            "passthrough_deepest",
        ),
        # pairing: taps must equal sources (tuple tap s_j == input of s_j)
        (
            dict(context_mode="per_layer", init_mode="target_explicit",
                 init_source_layers=[2, 3],
                 aux_hidden_state_layer_ids=[1, 3]),
            "per_layer context requires aux_hidden_state_layer_ids",
        ),
        # random needs one tap per draft layer
        (
            dict(context_mode="per_layer", init_mode="random",
                 kv_mode="trainable_shared",
                 aux_hidden_state_layer_ids=[1, 3, 2]),
            "one tap per draft layer",
        ),
        # copied-LR alias disagreement
        (
            dict(copied_lr_scale=0.2, lr_scale_copied=0.1),
            "disagree",
        ),
    ],
)
def test_config_validation(kwargs, needle):
    with pytest.raises(ValidationError, match=needle):
        DNestSpeculatorConfig(**kwargs)


def test_pairing_error_names_the_required_ids(target_dir):
    """The pairing error tells the user exactly which ids to pass."""
    _, path = target_dir
    torch.manual_seed(123)
    with pytest.raises(ValueError, match=r'--target-layer-ids\s+"2 3"'):
        DNestDraftModel.from_training_args(
            verifier_config=Qwen3Config(**DRAFT_KWARGS),
            verifier_name_or_path=str(path),
            draft_vocab_size=256,
            block_size=4,
            target_layer_ids=[1, 3],
            mask_token_id=255,
            draft_attn_impl="eager",
            init_mode="target_explicit",
            init_source_layers=[2, 3],
            context_mode="per_layer",
            kv_mode="split",
        )


def test_target_last_requires_matching_taps(target_dir):
    """per_layer + target_last derives taps [2, 3] for a 4-layer target with
    2 draft layers; the configured taps must match."""
    _, path = target_dir
    # hermetic equivalent with the tiny 4-layer target: sources [2, 3]
    torch.manual_seed(123)
    with pytest.raises(ValueError, match=r'--target-layer-ids\s+"2 3"'):
        DNestDraftModel.from_training_args(
            verifier_config=Qwen3Config(**DRAFT_KWARGS),
            verifier_name_or_path=str(path),
            draft_vocab_size=256,
            block_size=4,
            target_layer_ids=[1, 3],  # wrong: inputs of layers 2,3 are taps 2,3
            mask_token_id=255,
            draft_attn_impl="eager",
            init_mode="target_last",
            context_mode="per_layer",
            kv_mode="trainable_shared",
        )
    torch.manual_seed(123)
    model = DNestDraftModel.from_training_args(
        verifier_config=Qwen3Config(**DRAFT_KWARGS),
        verifier_name_or_path=str(path),
        draft_vocab_size=256,
        block_size=4,
        target_layer_ids=[2, 3],
        mask_token_id=255,
        draft_attn_impl="eager",
        init_mode="target_last",
        context_mode="per_layer",
        kv_mode="trainable_shared",
    )
    assert model.config.init_source_layers == [2, 3]
    assert model.config.aux_hidden_state_layer_ids == [2, 3]


def test_copied_lr_alias_lockstep():
    config = DNestSpeculatorConfig(copied_lr_scale=0.1)
    assert config.lr_scale_copied == 0.1
    config = DNestSpeculatorConfig(lr_scale_copied=0.2)
    assert config.copied_lr_scale == 0.2
    config = DNestSpeculatorConfig()
    assert config.copied_lr_scale == 1.0 == config.lr_scale_copied


def test_tap_layers_manifest(tmp_path):
    hs = tmp_path / "hidden_states"
    hs.mkdir()
    data_path = tmp_path / "dataset"

    # no manifest: accepted with a warning (older caches)
    check_tap_layers_manifest(hs, data_path, [1, 3])

    manifest = hs / "tap_layers.json"
    manifest.write_text(json.dumps({"target_layer_ids": [1, 3]}))
    check_tap_layers_manifest(hs, data_path, [1, 3])  # match: ok
    with pytest.raises(ValueError, match="extracted from target layers"):
        check_tap_layers_manifest(hs, data_path, [2, 3])
    # implicit path resolution: hidden_states under data_path
    nested = data_path / "hidden_states"
    nested.mkdir(parents=True)
    (nested / "tap_layers.json").write_text(
        json.dumps({"target_layer_ids": [1, 3]})
    )
    check_tap_layers_manifest(None, data_path, [1, 3])
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        check_tap_layers_manifest(None, data_path, [2, 3])


# --- 7. fused regression (golden) -------------------------------------------


def test_fused_regression_golden(tmp_path):
    """context_mode="fused" reproduces the pre-per-layer dnest bit-for-bit.

    The golden was captured on unmodified main (before the per-layer work)
    with the same construction and batch; see
    generate_dnest_fused_golden.py.
    """
    import pathlib

    golden_path = (
        pathlib.Path(__file__).parent / "golden" / "dnest_fused_golden.pt"
    )
    golden = torch.load(golden_path, weights_only=False)

    target_path = tmp_path / "target"
    build_target(target_path)
    model = build_golden_model(target_path)
    batch = build_batch()
    # the golden inputs pin our batch construction
    for key, value in golden["batch"].items():
        assert torch.equal(value, batch[key]), key

    out = run_golden_model(model, batch)
    for key in ("hidden", "logits", "targets", "aligned_loss_mask", "loss"):
        assert torch.equal(golden[key].detach(), out[key].detach()), key
    for key, value in golden["metrics"].items():
        assert torch.equal(value.detach(), out["metrics"][key].detach()), key


# --- smoke: every kv_mode trains and every forward runs ---------------------


@pytest.mark.parametrize(
    "kv_mode,init_mode",
    [
        ("frozen_shared", "target_after_extracted"),
        ("split", "target_after_extracted"),
        ("split_trainable", "target_after_extracted"),
        ("split_trainable", "random"),
        ("trainable_shared", "target_after_extracted"),
        ("trainable_shared", "random"),
    ],
)
def test_smoke_loss_decreases(target_dir, kv_mode, init_mode):
    """A few hundred-iteration-scale steps on the tiny model: the loss
    decreases for every arm (the hermetic stand-in for the training
    smokes)."""
    _, path = target_dir
    model = _build_per_layer(path, kv_mode=kv_mode, init_mode=init_mode)
    model.train()
    config = SimpleNamespace(optimizer="adamw", lr=5e-4, weight_decay=0.0)
    optimizer = build_optimizers(model, config)[0]
    loss_config = resolve_loss_config("ce", "eager")
    torch.manual_seed(9)
    batch = build_batch()
    batch["hidden_states"] = torch.randn(1, 24, 2 * 64)
    batch["verifier_last_hidden_states"] = torch.randn(1, 24, 64)

    from speculators.models.dflash.metrics import compute_metrics

    losses = []
    for _ in range(30):
        hidden, logits, targets, aligned_loss_mask, block_idx = (
            model._backbone_forward(max_anchors=8, **batch)  # noqa: SLF001
        )
        loss, _ = compute_metrics(
            logits, targets, aligned_loss_mask, model.block_size,
            gamma=4.0, loss_config=loss_config,
            per_position_loss_weight="fixed-exp-decay", dpace_alpha=0.5,
            sample_from_anchor=model.config.sample_from_anchor,
            token_ids=batch["input_ids"][:, block_idx],
            target_temperature=1.0, t2d=None, loss_implementation="eager",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], (losses[0], losses[-1])
