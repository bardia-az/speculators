import logging
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import kl_div_loss
from speculators.model import SpeculatorModel
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflex.config import DFlexSpeculatorConfig
from speculators.models.dflex.prefix import sample_prefix_layout
from speculators.models.dflex.prefix_metrics import prefix_metrics

PREFIX_FIELDS = (
    "num_blind_anchors",
    "num_full_anchors",
    "num_partial_anchors",
    "prefix_length_dist",
    "prefix_length_ratio",
    "lambda_ar",
    "lambda_prefix",
)

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
    def prefix_settings(args):
        settings = {k: args[k] for k in PREFIX_FIELDS if args.get(k) is not None}
        if "num_blind_anchors" not in settings and args.get("max_anchors") is not None:
            settings["num_blind_anchors"] = args["max_anchors"]
        return settings

    def _prefix_forward(
        self,
        layout,
        hidden_states,
        input_ids,
        verifier_last_hidden_states,
        document_ids,
        position_ids,
    ):
        width = input_ids.shape[1]
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(width, device=device)[None, :]
        positions = position_ids[0, layout.anchors[layout.block]] + layout.offset
        positions = torch.cat((position_ids, positions[None, :]), dim=1)

        def mask(window):
            predicate, rows, cols = create_anchor_block_mask_mod(
                document_ids[0],
                width,
                layout.anchors,
                self.block_size,
                sliding_window=window,
                attention_mode=self.attention_mode,
                prefix_layout=layout,
            )
            return self._create_mask_fn(
                predicate, B=None, H=None, Q_LEN=rows, KV_LEN=cols, device=device
            )

        _, logits, teacher = self._decode_training_rows(
            hidden_states,
            verifier_last_hidden_states,
            self.embed_tokens(layout.input_ids(input_ids, self.mask_token_id)),
            positions,
            layout.indices,
            mask(None) if self.uses_full_attn else None,
            mask(self.sliding_window) if self.uses_sliding_window_attn else None,
        )
        return logits, teacher

    def forward(
        self,
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids=None,
        loss_config=None,
        gamma=4.0,
        max_anchors=None,
        per_position_loss_weight="fixed-exp-decay",
        dpace_alpha=0.5,
        target_temperature=1.0,
        loss_implementation="fused",
        prefix_seed=42,
        prefix_step=0,
        **kwargs,
    ):
        enabled = self.config.uses_prefix_blocks
        if enabled and loss_config and "tau_rb" in loss_config:
            raise ValueError("tau_rb cannot be combined with DFlex prefix blocks")
        if enabled and getattr(self, "t2d", None) is not None:
            raise ValueError(
                "DFlex prefix acceptance metrics require full verifier vocabulary"
            )
        stats = {}
        _, blind, metrics = super().forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            loss_config,
            gamma,
            self.config.num_blind_anchors if max_anchors is None else max_anchors,
            per_position_loss_weight,
            dpace_alpha,
            target_temperature,
            loss_implementation,
            blind_stats=stats,
            **kwargs,
        )
        metrics.update(stats)
        metrics["blind_loss_sum"] = metrics["loss_sum"]
        metrics["blind_loss_total"] = metrics["loss_total"]
        if not enabled:
            metrics["total_loss_sum"] = metrics["loss_sum"]
            metrics["total_loss_total"] = metrics["loss_total"]
            return None, blind, metrics

        # Step is restored by the trainer on resume. Validation uses batch index.
        # Draws happen on CPU with a dedicated generator. Protect the blind RNG
        # stream as well if the transformer config enables attention dropout.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = (
            prefix_seed + 1_000_003 * prefix_step + 97 * rank + int(self.training)
        ) % (2**63)
        layout = sample_prefix_layout(self.config, loss_mask, document_ids, seed)
        logits = targets = None
        if layout.offset.numel():
            device = input_ids.device
            devices = [] if device.type == "cpu" else [device]
            with torch.random.fork_rng(devices=devices, device_type=device.type):
                logits, targets = self._prefix_forward(
                    layout,
                    hidden_states,
                    input_ids,
                    verifier_last_hidden_states,
                    document_ids,
                    position_ids,
                )
        valid = loss_mask[:, self._loss_mask_gather_indices(layout.indices)]
        prefix, stats = prefix_metrics(
            self.config,
            layout,
            logits,
            targets,
            valid,
            loss_config or {"kl_div": (kl_div_loss, 1.0)},
            gamma,
            self.training,
        )
        total = blind + self.config.lambda_prefix * prefix
        metrics.update(stats)
        metrics["total_loss_sum"] = total.detach()
        metrics["total_loss_total"] = metrics["loss_total"]
        return None, total, metrics

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
            **cls._build_base_config_kwargs("dflex", verifier_config, **kwargs),
            **cls.prefix_settings(kwargs),
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
