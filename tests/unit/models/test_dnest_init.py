"""Unit tests for the dnest target-derived initialisation.

Maps to the dnest test plan:
1. copy correctness per init mode (+ init_source_layers in the saved config),
2. functional equivalence of a copied layer vs its source target layer,
3. no re-copy when loading a saved checkpoint,
4. shape mismatch raises and copies nothing,
5. random mode goes through the same code path as DFlash (matched weights,
   parameter set, and config),
6. passthrough fusion initialisation,
7. optimizer learning-rate groups under copied_lr_scale.
"""

import json

import pytest
import torch
from pydantic import ValidationError
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)

from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashDraftModel
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dnest import DNestDraftModel, DNestSpeculatorConfig
from speculators.models.dnest.weight_copy import (
    COPIED_LAYER_SUFFIXES,
    check_layer_compatibility,
    copy_target_decoder_layers,
    init_fusion_passthrough_deepest,
    resolve_init_source_layers,
)
from speculators.train.optimizers import build_optimizers
from speculators.utils.loading import load_model_layers

# --- tiny shared target: 4 layers, small dims, tied head, sharded on save ---

TARGET_KWARGS = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    attention_bias=False,
    tie_word_embeddings=True,
    layer_types=["full_attention"] * 4,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
)


@pytest.fixture(scope="module")
def target_dir(tmp_path_factory):
    """A tiny random Qwen3-4B stand-in, saved as a sharded safetensors checkpoint."""
    torch.manual_seed(0)
    target = Qwen3ForCausalLM(Qwen3Config(**TARGET_KWARGS))
    target.eval()
    path = tmp_path_factory.mktemp("dnest_target")
    target.save_pretrained(path, max_shard_size="200kB")
    return target, path


def _draft_config(num_layers=2, **overrides):
    kwargs = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        sliding_window=None,
        use_sliding_window=False,
        layer_types=["full_attention"] * num_layers,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def _build_dnest(target_path, *, init_mode=None, seed=123, num_layers=2,
                 target_layer_ids=(1, 3), **extra):
    torch.manual_seed(seed)
    kwargs = dict(
        verifier_name_or_path=str(target_path),
        draft_vocab_size=256,
        block_size=4,
        target_layer_ids=list(target_layer_ids),
        mask_token_id=255,
    )
    if init_mode is not None:
        kwargs["init_mode"] = init_mode
    kwargs.update(extra)
    return DNestDraftModel.from_training_args(
        verifier_config=_draft_config(num_layers=num_layers), **kwargs
    )


# --- source resolution ------------------------------------------------------


def test_resolve_init_source_layers_modes():
    """after_extracted pairs each draft layer with the consumer of its tap."""
    assert (
        resolve_init_source_layers(
            "target_after_extracted", [1, 3], 2, 4, None
        )
        == [1, 3]
    )
    assert resolve_init_source_layers("target_last", [1, 3], 2, 4, None) == [2, 3]
    assert (
        resolve_init_source_layers("target_explicit", [1, 3], 2, 4, [0, 2])
        == [0, 2]
    )
    assert resolve_init_source_layers("random", [1, 3], 2, 4, None) is None


@pytest.mark.parametrize(
    "mode,aux,n_draft,explicit,needle",
    [
        ("target_after_extracted", [1], 2, None, "number of draft layers"),
        ("target_after_extracted", [1, 4], 2, None, "final"),
        ("target_explicit", [1, 3], 2, None, "requires init_source_layers"),
        ("target_explicit", [1, 3], 2, [0, 4], "not valid target layers"),
        ("target_explicit", [1, 3], 2, [1, 1], "distinct"),
        ("target_explicit", [1, 3], 2, [0, 1, 2], "must equal"),
        ("target_last", [1, 3], 5, None, "only has"),
    ],
)
def test_resolve_init_source_layers_errors(mode, aux, n_draft, explicit, needle):
    with pytest.raises(ValueError, match=needle):
        resolve_init_source_layers(mode, aux, n_draft, 4, explicit)


# --- test 1: copy correctness per mode --------------------------------------


def _assert_layer_copied(model, target_path, draft_idx, source_idx):
    names = [f"model.layers.{source_idx}.{s}" for s in COPIED_LAYER_SUFFIXES]
    ref = load_model_layers(names, str(target_path))
    for suffix in COPIED_LAYER_SUFFIXES:
        got = model.get_parameter(f"layers.{draft_idx}.{suffix}").detach()
        want = ref[f"model.layers.{source_idx}.{suffix}"].to(got.dtype)
        assert torch.equal(got, want), f"layers.{draft_idx}.{suffix} mismatch"


@pytest.mark.parametrize(
    "init_mode,extra,expected_sources",
    [
        ("target_after_extracted", {}, [1, 3]),
        ("target_last", {}, [2, 3]),
        ("target_explicit", {"init_source_layers": [0, 2]}, [0, 2]),
    ],
)
def test_copy_correctness_each_mode(
    target_dir, init_mode, extra, expected_sources
):
    _, path = target_dir
    model = _build_dnest(path, init_mode=init_mode, **extra)
    for draft_idx, source_idx in enumerate(expected_sources):
        _assert_layer_copied(model, path, draft_idx, source_idx)
    assert model.config.init_source_layers == expected_sources


def test_init_final_norm_option(target_dir):
    _, path = target_dir
    model = _build_dnest(path, init_mode="target_after_extracted",
                         init_final_norm=True)
    ref = load_model_layers(["model.norm.weight"], str(path))
    got = model.get_parameter("norm.weight").detach()
    assert torch.equal(got, ref["model.norm.weight"].to(got.dtype))


def test_saved_config_records_source_layers(target_dir, tmp_path):
    """init_source_layers is written to the saved config; null for random."""
    _, path = target_dir
    model = _build_dnest(path, init_mode="target_after_extracted")
    model.save_pretrained(tmp_path / "copied")
    saved = json.loads((tmp_path / "copied" / "config.json").read_text())
    assert saved["init_source_layers"] == [1, 3]
    assert saved["init_mode"] == "target_after_extracted"

    random_model = _build_dnest(path, init_mode="random", seed=7)
    random_model.save_pretrained(tmp_path / "random")
    saved = json.loads((tmp_path / "random" / "config.json").read_text())
    assert saved["init_source_layers"] is None
    assert saved["init_mode"] == "random"


# --- test 2: functional equivalence -----------------------------------------


def test_copied_layer_matches_target_layer(target_dir):
    """A copied draft layer reproduces its source target layer bit-for-bit
    (same input states, positions, causal mask; no target-feature injection).

    With a zero-length target_hidden the DFlash attention's k/v reduce to the
    plain Qwen3 path, so this catches missed norms or RoPE differences that a
    weight comparison cannot."""
    target, path = target_dir
    config = _draft_config(num_layers=4)
    config._attn_implementation = "eager"

    draft_layer = Qwen3DFlashDecoderLayer(config, layer_idx=0)
    draft_layer.load_state_dict(target.model.layers[1].state_dict())
    draft_layer.eval()
    target_layer = target.model.layers[1]
    target_layer.eval()

    torch.manual_seed(11)
    hidden = torch.randn(1, 8, config.hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)
    rotary = Qwen3RotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min)
    causal_mask = causal_mask.triu(diagonal=1)
    cache_position = torch.arange(8)

    with torch.no_grad():
        target_out = target_layer(
            hidden_states=hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
        draft_out = draft_layer(
            hidden_states=hidden,
            target_hidden=torch.zeros(1, 0, config.hidden_size),
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    assert torch.allclose(target_out, draft_out, atol=1e-5, rtol=1e-5), (
        (target_out - draft_out).abs().max()
    )


# --- test 3: no re-copy on load ---------------------------------------------


def test_no_recopy_on_load(target_dir, tmp_path):
    _, path = target_dir
    model = _build_dnest(path, init_mode="target_after_extracted")
    save_dir = tmp_path / "dnest_ckpt"
    model.save_pretrained(save_dir)

    from safetensors.torch import load_file, save_file

    shard = save_dir / "model.safetensors"
    state = load_file(str(shard))
    state["layers.0.self_attn.q_proj.weight"] += 1.0
    save_file(state, str(shard))

    reloaded = SpeculatorModel.from_pretrained(str(save_dir))
    assert isinstance(reloaded, DNestDraftModel)
    got = reloaded.get_parameter("layers.0.self_attn.q_proj.weight").detach()
    want = (
        model.get_parameter("layers.0.self_attn.q_proj.weight").detach() + 1.0
    )
    assert torch.equal(got, want), "loading re-copied or lost checkpoint values"


# --- test 4: shape mismatch raises, copies nothing ---------------------------


def test_check_layer_compatibility_raises(target_dir):
    _, path = target_dir
    from speculators.models.utils import get_verifier_config

    target_config = get_verifier_config(str(path))
    mismatched = _draft_config(intermediate_size=192)
    with pytest.raises(ValueError, match="intermediate size"):
        check_layer_compatibility(mismatched, target_config, [0, 1])


def test_copy_target_layers_shape_mismatch_copies_nothing(target_dir, tmp_path):
    """A target whose tensors are shaped differently is refused before any
    parameter is touched."""
    _, path = target_dir
    torch.manual_seed(123)
    model = _build_dnest(path, init_mode="random")
    before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith("layers.")
    }

    # A mismatched target: same dims except intermediate_size.
    kwargs = dict(TARGET_KWARGS)
    kwargs["intermediate_size"] = 192
    torch.manual_seed(0)
    mismatched = Qwen3ForCausalLM(Qwen3Config(**kwargs))
    mismatched_dir = tmp_path / "mismatched_target"
    mismatched.save_pretrained(mismatched_dir)

    with pytest.raises(ValueError, match="No weights were copied"):
        copy_target_decoder_layers(model, str(mismatched_dir), [0, 1])
    for name, param in model.named_parameters():
        if name.startswith("layers."):
            assert torch.equal(before[name], param.detach()), name


def test_missing_target_tensors_refuse_copy(target_dir, tmp_path):
    """A target lacking any of the copied tensors (e.g. no q/k-norm) is
    refused entirely."""
    _, path = target_dir
    torch.manual_seed(123)
    model = _build_dnest(path, init_mode="random")

    from safetensors.torch import load_file, save_file

    poor_dir = tmp_path / "no_qk_norm_target"
    poor_dir.mkdir()
    index = json.loads((path / "model.safetensors.index.json").read_text())
    drop = [
        k
        for k in index["weight_map"]
        if "q_norm" in k or "k_norm" in k
    ]
    for key in drop:
        index["weight_map"].pop(key)
    shards = {}
    for key, shard in index["weight_map"].items():
        shards.setdefault(shard, {})[key] = load_file(str(path / shard))[key]
    for shard, tensors in shards.items():
        save_file(tensors, str(poor_dir / shard))
    (poor_dir / "model.safetensors.index.json").write_text(
        json.dumps(index)
    )
    (poor_dir / "config.json").write_text((path / "config.json").read_text())

    with pytest.raises(ValueError, match="absent"):
        copy_target_decoder_layers(model, str(poor_dir), [0, 1])


# --- test 5: random mode == DFlash ------------------------------------------


def test_random_mode_matches_dflash(target_dir):
    """init_mode=random trains through exactly the DFlash path: same weights
    under the same seed, same parameter set, config equal modulo the dnest
    fields."""
    _, path = target_dir
    kwargs = dict(
        verifier_name_or_path=str(path),
        draft_vocab_size=256,
        block_size=4,
        target_layer_ids=[1, 3],
        mask_token_id=255,
    )

    torch.manual_seed(42)
    dflash = DFlashDraftModel.from_training_args(
        verifier_config=_draft_config(), **kwargs
    )
    torch.manual_seed(42)
    dnest = _build_dnest(path, init_mode="random", seed=42)

    dflash_state = dict(dflash.named_parameters())
    dnest_state = dict(dnest.named_parameters())
    assert set(dflash_state) == set(dnest_state)
    for name, param in dflash_state.items():
        assert torch.equal(param.detach(), dnest_state[name].detach()), name

    assert dnest.config.init_mode == "random"
    assert dnest.config.init_source_layers is None
    assert dnest.config.copied_lr_scale == 1.0
    assert dnest.config.fusion_init == "random"
    assert dnest.config.init_final_norm is False
    # everything except the dnest-added fields and identity matches dflash
    dflash_dict = dflash.config.to_dict()
    dnest_dict = dnest.config.to_dict()
    dnest_only = {
        "speculators_model_type",
        "architectures",
        "speculators_config",  # identical except algorithm: "dnest" vs "dflash"
        "init_mode",
        "init_source_layers",
        "fusion_init",
        "copied_lr_scale",
        "init_final_norm",
    }
    for key in set(dnest_dict) - dnest_only:
        assert dnest_dict[key] == dflash_dict.get(key), key
    assert (
        dnest_dict["speculators_config"]["algorithm"] == "dnest"
        and dflash_dict["speculators_config"]["algorithm"] == "dflash"
    )


# --- test 6: passthrough fusion ---------------------------------------------


def test_passthrough_fusion(target_dir):
    _, path = target_dir
    model = _build_dnest(
        path, init_mode="target_after_extracted", fusion_init="passthrough_deepest"
    )
    hidden = model.config.transformer_layer_config.hidden_size
    torch.manual_seed(3)
    taps = [torch.randn(1, 5, hidden) for _ in range(2)]
    fused = model.fc(torch.cat(taps, dim=-1))

    deepest = taps[1]  # aux ids [1, 3]: the deepest tap is the last slice
    assert torch.equal(fused, deepest)
    # hidden_norm then divides by the per-token RMS: the documented closest
    # equivalent to an exact passthrough.
    normed = model.hidden_norm(fused)
    rms = deepest.norm(dim=-1, keepdim=True) / (hidden**0.5)
    assert torch.allclose(normed, deepest / rms, atol=1e-5)
    assert torch.equal(
        model.hidden_norm.weight.detach(),
        torch.ones_like(model.hidden_norm.weight),
    )


def test_passthrough_fusion_unsorted_aux_ids():
    """The deepest tap is located by value, not assumed last."""
    fc = torch.nn.Linear(3 * 8, 8, bias=False)
    slice_idx = init_fusion_passthrough_deepest(fc, [5, 20, 9])
    assert slice_idx == 1
    x = torch.randn(2, 3 * 8)
    out = fc(x)
    assert torch.equal(out, x[..., 8:16])


# --- test 7: learning-rate groups -------------------------------------------


def test_optimizer_lr_groups(target_dir):
    from types import SimpleNamespace

    _, path = target_dir
    model = _build_dnest(
        path, init_mode="target_after_extracted", copied_lr_scale=0.1
    )
    scales = model.optimizer_param_lr_scales()
    assert len(scales) == 2 * len(COPIED_LAYER_SUFFIXES)
    assert set(scales.values()) == {0.1}
    assert "fc.weight" not in scales and "hidden_norm.weight" not in scales

    config = SimpleNamespace(
        optimizer="adamw", lr=1e-3, weight_decay=0.0
    )
    optimizer = build_optimizers(model, config)[0]
    lrs = {round(group["lr"], 10) for group in optimizer.param_groups}
    assert lrs == {1e-3, 1e-4}
    scaled = [g for g in optimizer.param_groups if abs(g["lr"] - 1e-4) < 1e-12]
    assert len(scaled[0]["params"]) == 2 * len(COPIED_LAYER_SUFFIXES)

    # scale 1.0 and random mode: a single group, DFlash-identical
    plain = _build_dnest(path, init_mode="random")
    assert plain.optimizer_param_lr_scales() == {}
    single = build_optimizers(plain, config)[0]
    assert len(single.param_groups) == 1
    assert single.param_groups[0]["lr"] == 1e-3


# --- config defaults ---------------------------------------------------------


def test_config_defaults_and_explicit_validation():
    config = DNestSpeculatorConfig()
    assert config.init_mode == "target_after_extracted"
    assert config.init_source_layers is None
    assert config.fusion_init == "random"
    assert config.copied_lr_scale == 1.0
    assert config.init_final_norm is False
    with pytest.raises(ValidationError):
        DNestSpeculatorConfig(init_mode="target_explicit")
    with pytest.raises(ValidationError):
        DNestSpeculatorConfig(copied_lr_scale=-0.5)
