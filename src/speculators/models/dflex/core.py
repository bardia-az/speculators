import logging
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflex.config import DFlexSpeculatorConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DFlexDraftModel",
]


@SpeculatorModel.register("dflex")
class DFlexDraftModel(DFlashDraftModel):
    """DFlash with next-token (anchor-inclusive) output indexing.

    Identical to DFlash -- same backbone and number of draft layers,
    bidirectional attention within blocks, target-context visibility strictly
    before the anchor, anchor-offset position ids, mask token, shared frozen
    embedding and LM head, target-feature injection -- except for what each
    block position predicts. With the anchor at sequence position ``a`` and
    ``B = block_size`` input positions ``[anchor, MASK x (B-1)]``:

    - DFlash: the output at position ``k`` (``k = 1..B-1``) predicts the
      token at sequence position ``a + k``; the anchor output is unused
      (``B - 1`` predictions per block).
    - dflex: the output at position ``i`` (``i = 0..B-1``) predicts the
      token at sequence position ``a + i + 1``; the anchor's own output is
      the first draft token (``B`` predictions per block).

    A dflex block therefore drafts ``B`` tokens from ``B`` positions, while a
    DFlash block drafts ``B - 1``: DFlash with ``--block-size 16`` and dflex
    with ``--block-size 15`` both draft 15 tokens. Compare the two at equal
    draft-token counts.
    """

    config_class: ClassVar[type[DFlexSpeculatorConfig]] = DFlexSpeculatorConfig  # type: ignore[misc]

    def _loss_mask_gather_indices(
        self, anchored_block_indices: torch.Tensor
    ) -> torch.Tensor:
        """Loss-mask positions for dflex slots: the *label* positions.

        Slot ``i`` predicts the token at ``a + i + 1``, so a slot's loss-mask
        entry is ``loss_mask[a + i + 1]`` -- one position past the block
        positions DFlash gathers. ``select_anchors`` only yields anchors with
        ``a <= T - B - 1``, so the final label position ``a + B <= T - 1`` is
        always in bounds.
        """
        return anchored_block_indices + 1

    @staticmethod
    def _apply_serving_causal_hint(config: DFlexSpeculatorConfig) -> None:
        """Stamp the draft transformer config with vLLM's serving causality.

        vLLM resolves per-layer draft causality from
        ``transformer_layer_config.is_causal`` first (``_dflash_layer_causal``
        in its qwen3_dflash runtime), so stamping it at training time makes
        serving follow the training mode: ``bidirectional`` -> ``False``,
        ``causal`` -> ``True``. ``hybrid`` -- a per-(query, key) in-block
        pattern -- cannot ride a single boolean, so it is deliberately left
        unstamped and warned about: hybrid checkpoints are refused by the
        repo's serving script and evaluated by the offline crosscheck.
        """
        mode = config.attention_mode
        if mode == "hybrid":
            logger.warning(
                "dflex attention_mode='hybrid' cannot be served by vLLM: its "
                "Attention interface takes a single causal flag, not a "
                "per-(query, key) in-block pattern. The serving-causality "
                "hint is left unstamped; scripts/serve_dflash.sh refuses "
                "hybrid checkpoints and scripts/acceptance_crosscheck.py "
                "evaluates them offline."
            )
            tl_config = config.transformer_layer_config
            if "is_causal" in tl_config.__dict__:  # noqa: SLF001
                # Never inherit a stale hint on an explicit hybrid (e.g. a
                # finetune run re-selecting the mode via the CLI).
                del tl_config.__dict__["is_causal"]  # noqa: SLF001
            return
        config.transformer_layer_config.is_causal = mode == "causal"

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlexDraftModel":
        """Create a dflex model from training arguments (mirrors DFlash)."""
        if kwargs.get("sample_from_anchor") is False:
            raise ValueError(
                "dflex is DFlash with sample_from_anchor=True indexing (the "
                "anchor output is the first draft token). For the DFlash "
                "bonus-anchor layout use --speculator-type dflash."
            )
        kwargs["sample_from_anchor"] = True
        config = DFlexSpeculatorConfig(
            **cls._build_base_config_kwargs("dflex", verifier_config, **kwargs)
        )
        cls._apply_serving_causal_hint(config)
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        logger.warning(
            "dflex checkpoints need a serving stack that supports "
            "sample_from_anchor=True for the DFlash runtime (e.g. vLLM with "
            "the 'dflex' method); stock vLLM 'dflash' serving rejects them."
        )
        return model
