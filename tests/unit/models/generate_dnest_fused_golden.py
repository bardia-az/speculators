"""Regenerate the fused-mode dnest regression golden.

Run from the speculators fork root against the code version that should be
pinned (originally: unmodified main, before the per-layer-context work):

    uv run python tests/unit/models/generate_dnest_fused_golden.py

Writes ``tests/unit/models/golden/dnest_fused_golden.pt``. The companion test
``test_dnest_per_layer.py::test_fused_regression_golden`` rebuilds the exact
same model and batch and compares bitwise. Bitwise equality is only guaranteed
on the platform that produced the golden (BLAS kernels differ across
architectures) — regenerate there if the test runs elsewhere.
"""

from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.models.dnest import DNestDraftModel

# Mirrors the fixtures in test_dnest_init.py so the models (and thus the
# weights drawn for a given seed) are identical between generator and test.
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

DRAFT_KWARGS = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
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
    layer_types=["full_attention"] * 2,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
)

SEQ_LEN = 24


def build_target(tmpdir: Path):
    torch.manual_seed(0)
    target = Qwen3ForCausalLM(Qwen3Config(**TARGET_KWARGS))
    target.eval()
    target.save_pretrained(tmpdir, max_shard_size="200kB")
    return tmpdir


def build_model(target_path: str) -> DNestDraftModel:
    torch.manual_seed(123)
    model = DNestDraftModel.from_training_args(
        verifier_config=Qwen3Config(**DRAFT_KWARGS),
        verifier_name_or_path=str(target_path),
        draft_vocab_size=256,
        block_size=4,
        target_layer_ids=[1, 3],
        mask_token_id=255,
        draft_attn_impl="eager",
    )
    model.eval()
    return model


def build_batch() -> dict:
    g = torch.Generator().manual_seed(7)
    seq = SEQ_LEN
    hidden_states = torch.randn(1, seq, 2 * 64, generator=g)
    verifier_last = torch.randn(1, seq, 64, generator=g)
    input_ids = torch.randint(0, 255, (1, seq), generator=g)
    loss_mask = torch.zeros(1, seq, dtype=torch.float32)
    loss_mask[0, 4:12] = 1.0
    loss_mask[0, 16:24] = 1.0
    document_ids = torch.zeros(1, seq, dtype=torch.long)
    document_ids[0, 12:] = 1
    return dict(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last,
        document_ids=document_ids,
    )


def run_model(model: DNestDraftModel, batch: dict) -> dict:
    # _backbone_forward + compute_metrics instead of the compiled forward
    # wrapper, so the pinned path is eager on every platform. The loss config
    # matches the training arms (plain CE, eager implementation).
    from speculators.losses import resolve_loss_config
    from speculators.models.dflash.metrics import compute_metrics

    hidden, logits, targets, aligned_loss_mask, block_indices = (
        model._backbone_forward(  # noqa: SLF001
            max_anchors=8,
            **batch,
        )
    )
    loss, metrics = compute_metrics(
        logits,
        targets,
        aligned_loss_mask,
        model.block_size,
        gamma=4.0,
        loss_config=resolve_loss_config("ce", "eager"),
        per_position_loss_weight="fixed-exp-decay",
        dpace_alpha=0.5,
        sample_from_anchor=model.config.sample_from_anchor,
        token_ids=input_ids_of(batch, block_indices),
        target_temperature=1.0,
        t2d=None,
        loss_implementation="eager",
    )
    return dict(
        hidden=hidden,
        logits=logits,
        targets=targets,
        aligned_loss_mask=aligned_loss_mask,
        block_indices=block_indices,
        loss=loss,
        loss_value=float(loss),
        metrics=metrics,
    )


def input_ids_of(batch: dict, block_indices: torch.Tensor) -> torch.Tensor:
    return batch["input_ids"][:, block_indices]


def main() -> None:
    here = Path(__file__).parent
    tmpdir = here / "golden" / "_golden_target"
    if not (tmpdir / "config.json").exists():
        build_target(tmpdir)
    model = build_model(str(tmpdir))
    batch = build_batch()
    out = run_model(model, batch)
    golden = here / "golden" / "dnest_fused_golden.pt"
    torch.save(
        dict(
            hidden=out["hidden"],
            logits=out["logits"],
            targets=out["targets"],
            aligned_loss_mask=out["aligned_loss_mask"],
            loss=out["loss"],
            metrics=out["metrics"],
            # inputs too, so the test can verify its own batch construction
            batch=batch,
        ),
        golden,
    )
    print(f"wrote {golden} loss={out['loss_value']:.6f}")


if __name__ == "__main__":
    main()
