import logging
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dnest.config import DNestSpeculatorConfig
from speculators.models.dnest.weight_copy import (
    COPIED_LAYER_SUFFIXES,
    check_layer_compatibility,
    copy_target_decoder_layers,
    init_fusion_passthrough_deepest,
    resolve_init_source_layers,
)
from speculators.models.utils import get_verifier_config

logger = logging.getLogger(__name__)

__all__ = [
    "DNestDraftModel",
]


@SpeculatorModel.register("dnest")
class DNestDraftModel(DFlashDraftModel):
    """DFlash drafter whose decoder layers can be initialised from target layers.

    Architecturally identical to DFlash: same anchored-block indexing (block
    position k predicts token anchor + k), fused target-feature injection,
    attention, losses, and frozen shared embedding / LM head. The only
    addition is target-derived initialisation, configured by
    ``DNestSpeculatorConfig.init_mode`` (see there for the modes and the
    aux-id notation note).

    Target weights are copied exactly once, in :meth:`from_training_args`
    (the fresh-training path); loading a saved dnest checkpoint -- for
    resuming, evaluation, or serving -- goes through ``from_pretrained``
    and never re-copies.
    """

    config_class: ClassVar[type[DNestSpeculatorConfig]] = DNestSpeculatorConfig  # type: ignore[misc]

    def __init__(self, config: DNestSpeculatorConfig) -> None:
        super().__init__(config=config)
        # Informational record of the last fresh-init copy; the optimizer LR
        # groups never read this -- optimizer_param_lr_scales() derives them
        # from the config so a resumed run rebuilds identical groups.
        self._copied_param_names: set[str] = set()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DNestDraftModel":
        """Create a dnest model from training arguments and run the copy.

        Accepts every DFlash training argument plus the dnest knobs
        (``init_mode``, ``init_source_layers``, ``fusion_init``,
        ``copied_lr_scale``, ``init_final_norm``), which are stored on the
        saved config for reproducibility.
        """
        init_mode = kwargs.pop("init_mode", None) or "target_after_extracted"
        explicit_sources = kwargs.pop("init_source_layers", None)
        fusion_init = kwargs.pop("fusion_init", None) or "random"
        copied_lr_scale = kwargs.pop("copied_lr_scale", None)
        copied_lr_scale = 1.0 if copied_lr_scale is None else float(copied_lr_scale)
        init_final_norm = bool(kwargs.pop("init_final_norm", False) or False)

        base_config_kwargs = cls._build_base_config_kwargs(
            "dnest", verifier_config, **kwargs
        )
        aux_layer_ids = base_config_kwargs["aux_hidden_state_layer_ids"]
        num_draft_layers = verifier_config.num_hidden_layers

        if init_mode != "random":
            target_config = get_verifier_config(
                kwargs["verifier_name_or_path"],
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
            num_target_layers = target_config.num_hidden_layers
            source_layers = resolve_init_source_layers(
                init_mode,
                aux_layer_ids,
                num_draft_layers,
                num_target_layers,
                explicit_sources,
            )
            check_layer_compatibility(
                verifier_config, target_config, source_layers
            )
        else:
            source_layers = None

        config = DNestSpeculatorConfig(
            init_mode=init_mode,
            init_source_layers=source_layers,
            fusion_init=fusion_init,
            copied_lr_scale=copied_lr_scale,
            init_final_norm=init_final_norm,
            **base_config_kwargs,
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        if source_layers is not None:
            model._copied_param_names = set(
                copy_target_decoder_layers(
                    model,
                    kwargs["verifier_name_or_path"],
                    source_layers,
                    init_final_norm=init_final_norm,
                )
            )
        if fusion_init == "passthrough_deepest":
            init_fusion_passthrough_deepest(model.fc, aux_layer_ids)
        return model

    def optimizer_param_lr_scales(self) -> dict[str, float]:
        """Per-parameter learning-rate multipliers for optimizer grouping.

        Parameters initialised from the target get ``copied_lr_scale``;
        everything else (fusion fc/hidden_norm, any non-copied module) keeps
        the base learning rate. Derived purely from the saved config -- not
        from runtime state -- so a run resumed from a checkpoint rebuilds
        bit-identical optimizer groups and its saved optimizer state loads
        cleanly.
        """
        config: DNestSpeculatorConfig = self.config  # type: ignore[assignment]
        if config.init_mode == "random" or config.copied_lr_scale == 1.0:
            return {}
        scales: dict[str, float] = {}
        num_draft_layers = config.transformer_layer_config.num_hidden_layers
        for draft_idx in range(num_draft_layers):
            for suffix in COPIED_LAYER_SUFFIXES:
                scales[f"layers.{draft_idx}.{suffix}"] = config.copied_lr_scale
        if config.init_final_norm:
            scales["norm.weight"] = config.copied_lr_scale
        return scales
