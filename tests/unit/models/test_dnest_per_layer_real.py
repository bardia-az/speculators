"""Real-model tests for dnest per-layer target context (Qwen3-0.6B).

Counterpart to ``test_dnest_per_layer.py`` (hermetic, tiny 4-layer target):
here the invariant is checked against the *actual* Qwen3-0.6B checkpoint --
the tap indexing on real transformer modules, and the per-layer context
keys/values reproducing the real target's own KV cache through weights
copied from the real safetensors file.

Requires the model in the local HF cache (``HF_HOME``); downloads are never
attempted (``local_files_only``). Skipped entirely when absent.
"""

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.utils import cached_file

from speculators.models.dnest import DNestDraftModel

TARGET_ID = "Qwen/Qwen3-0.6B"
# Production-like spacing across the 28 target layers: draft layer j copies
# target layer SOURCES[j] and conditions on tuple tap SOURCES[j].
SOURCES = [2, 10, 18, 26]

try:
    _TARGET_PATH = str(
        cached_file(TARGET_ID, "config.json", local_files_only=True)
    ).rsplit("/", 1)[0]
except Exception:  # noqa: BLE001 - any cache/conn error means: skip
    _TARGET_PATH = None

pytestmark = pytest.mark.skipif(
    _TARGET_PATH is None,
    reason=f"{TARGET_ID} not found in the local HF cache (HF_HOME)",
)


def _load_target():
    torch.manual_seed(0)
    target = Qwen3ForCausalLM.from_pretrained(
        _TARGET_PATH, dtype=torch.float32
    )
    target.eval()
    return target


def _token_ids(seq_len: int = 64):
    tokenizer = AutoTokenizer.from_pretrained(_TARGET_PATH)
    return tokenizer(
        "The theory of speculative decoding rests on a simple observation:",
        return_tensors="pt",
    ).input_ids[:, :seq_len]


@pytest.fixture(scope="module")
def target_model():
    return _load_target()


@pytest.fixture(scope="module")
def target_forward(target_model):
    """One real fp32 forward with hidden states and the KV cache captured."""
    ids = _token_ids()
    with torch.no_grad():
        out = target_model(ids, use_cache=True, output_hidden_states=True)
    return ids, out


def _build_draft(kv_mode: str) -> DNestDraftModel:
    torch.manual_seed(123)
    target_config = Qwen3Config.from_pretrained(_TARGET_PATH)
    # Qwen3 has no mask token; the mask embedding is irrelevant to the K/V
    # invariant under test, so the eos id stands in.
    mask_token_id = AutoTokenizer.from_pretrained(_TARGET_PATH).eos_token_id
    draft_config = Qwen3Config(
        vocab_size=target_config.vocab_size,  # full vocab: embed is copied
        hidden_size=target_config.hidden_size,
        intermediate_size=target_config.intermediate_size,
        num_hidden_layers=len(SOURCES),
        num_attention_heads=target_config.num_attention_heads,
        num_key_value_heads=target_config.num_key_value_heads,
        head_dim=target_config.head_dim,
        rms_norm_eps=target_config.rms_norm_eps,
        rope_parameters=target_config.rope_parameters,
        max_position_embeddings=target_config.max_position_embeddings,
        tie_word_embeddings=False,
        layer_types=["full_attention"] * len(SOURCES),
    )
    return DNestDraftModel.from_training_args(
        verifier_config=draft_config,
        verifier_name_or_path=_TARGET_PATH,
        draft_vocab_size=target_config.vocab_size,
        block_size=4,
        target_layer_ids=list(SOURCES),
        mask_token_id=mask_token_id,
        draft_attn_impl="eager",
        init_mode="target_after_extracted",
        context_mode="per_layer",
        kv_mode=kv_mode,
    )


def test_real_tap_is_the_input_of_its_consumer_layer(target_model):
    """On the real 0.6B: tuple tap s equals the actual input of layer s."""
    for s in SOURCES:
        captured = {}

        def hook(module, args, kwargs, s=s, captured=captured):
            captured["input"] = (
                kwargs.get("hidden_states")
                if kwargs.get("hidden_states") is not None
                else args[0]
            )
            return None

        handle = target_model.model.layers[s].register_forward_pre_hook(
            hook, with_kwargs=True
        )
        try:
            ids = _token_ids()
            with torch.no_grad():
                out = target_model(ids, use_cache=True, output_hidden_states=True)
        finally:
            handle.remove()
        assert torch.equal(out.hidden_states[s], captured["input"]), (
            f"hidden_states[{s}] is not the input of decoder layer {s} "
            "(real Qwen3-0.6B)"
        )


@pytest.mark.parametrize("kv_mode", ["frozen_shared", "split", "trainable_shared"])
def test_real_context_kv_equals_target_cache(kv_mode, target_forward):
    """Per-layer context K/V through real copied weights == the real target's
    own cache at layer s_j (post k_norm, post RoPE for keys)."""
    from speculators.models.dflash.model_definitions import (
        apply_rotary_pos_emb,
    )

    ids, out = target_forward
    model = _build_draft(kv_mode)
    model.eval()
    seq_len = ids.shape[1]

    for j, s in enumerate(SOURCES):
        layer = model.layers[j]
        ctx_kv = getattr(layer.self_attn, "context_kv", None)
        norm = ctx_kv.input_layernorm if ctx_kv is not None else layer.input_layernorm
        k_proj = ctx_kv.k_proj if ctx_kv is not None else layer.self_attn.k_proj
        v_proj = ctx_kv.v_proj if ctx_kv is not None else layer.self_attn.v_proj
        k_norm = ctx_kv.k_norm if ctx_kv is not None else layer.self_attn.k_norm

        h_j = out.hidden_states[s]  # the extracted tap = input of layer s
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
        torch.testing.assert_close(k_rope, tgt_k, rtol=0, atol=1e-4)
        torch.testing.assert_close(v, tgt_v, rtol=0, atol=1e-4)


def test_real_context_kv_through_full_forward_split(target_forward):
    """End-to-end: the full training forward routes each real h_j through
    its layer's frozen context path (split mode, real weights)."""
    from speculators.models.dflash.model_definitions import (
        apply_rotary_pos_emb,
    )

    ids, out = target_forward
    seq_len = ids.shape[1]
    concat = torch.cat([out.hidden_states[s] for s in SOURCES], dim=-1)

    model = _build_draft("split")
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
        # recorded[j]: [bsz, ctx+q, H_kv, D] -> [H_kv, L, D], context part
        # pre-RoPE; apply the draft's own RoPE at the base positions
        k_normed = recorded[j][0, :seq_len].permute(1, 0, 2)
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = model.rotary_emb(recorded[j], pos)
        _, k_rope = apply_rotary_pos_emb(torch.zeros_like(k_normed), k_normed, cos, sin)
        tgt_k = out.past_key_values.layers[s].keys
        torch.testing.assert_close(k_rope, tgt_k, rtol=0, atol=1e-4)
