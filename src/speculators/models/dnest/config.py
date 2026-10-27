from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DNestSpeculatorConfig",
]


@SpeculatorModelConfig.register("dnest")
class DNestSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash speculator whose decoder layers can be initialised from target layers.

    dnest is architecturally identical to DFlash -- same anchored-block
    indexing, fused target-feature injection, attention, and losses -- so it
    trains and serves through the same code paths. The only addition is
    optional initialisation of the draft decoder layers (and, optionally, the
    draft final norm) from chosen layers of the target model, so training
    starts from the target's computation instead of random weights.

    Initialisation happens exactly once, when the drafter is created for
    training via ``from_training_args``; loading a saved dnest checkpoint
    never re-copies target weights.

    Notation note for ``target_after_extracted``: ``aux_hidden_state_layer_ids``
    are indices into the verifier's ``hidden_states`` tuple (slot 0 = embedding
    output, slot ``i`` = output of 0-based decoder layer ``i - 1``), so the
    target layer that *consumes* the tap with id ``l`` is 0-based decoder
    layer ``l`` -- numerically the aux id itself. ``init_source_layers`` is
    always expressed in 0-based decoder layer numbers (the numbering the
    target checkpoint uses for ``model.layers.{i}``).
    """

    speculators_model_type: Literal["dnest"] = "dnest"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DNestDraftModel"],
        description="Model architectures that can load these weights",
    )

    init_mode: Literal[
        "random", "target_after_extracted", "target_last", "target_explicit"
    ] = Field(
        default="target_after_extracted",
        description=(
            "How the draft decoder layers are initialised on a fresh training "
            "run. 'random': current DFlash initialisation (baseline arm). "
            "'target_after_extracted' (default for dnest): draft layer j is "
            "initialised from the target layer that consumes the j-th "
            "extracted hidden state. 'target_last': the target's last N "
            "layers in the target's order. 'target_explicit': from "
            "init_source_layers. Never applied when loading a checkpoint."
        ),
    )
    init_source_layers: list[int] | None = Field(
        default=None,
        description=(
            "The resolved source layer indices (0-based target decoder layer "
            "numbers) the draft layers were initialised from. Always written "
            "to the saved config for reproducibility; null when init_mode is "
            "'random'."
        ),
    )
    fusion_init: Literal["random", "passthrough_deepest"] = Field(
        default="random",
        description=(
            "Initialisation of the target-feature fusion projection. "
            "'random' (default): as DFlash. 'passthrough_deepest': the "
            "projection initially returns exactly the deepest extracted "
            "hidden state (identity on that slice, zeros elsewhere); the "
            "following hidden_norm then rescales it to unit RMS per token, "
            "which is the closest possible equivalent -- an exact passthrough "
            "cannot be expressed through a fixed-weight RMSNorm."
        ),
    )
    copied_lr_scale: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Learning-rate multiplier applied to parameters initialised from "
            "the target (optimizer parameter group). 1.0 (default) trains "
            "everything at the base learning rate."
        ),
    )
    init_final_norm: bool = Field(
        default=False,
        description=(
            "Also initialise the draft's final norm (before the LM head) from "
            "the target's final norm. Default off: DFlash initialises this "
            "norm randomly (its separately-named frozen 'verifier_norm', "
            "which normalises verifier hidden states when building training "
            "targets, is a different module and is always loaded from the "
            "target)."
        ),
    )

    @model_validator(mode="after")
    def _validate_explicit_sources(self) -> "DNestSpeculatorConfig":
        if self.init_mode == "target_explicit" and not self.init_source_layers:
            raise ValueError(
                "init_mode='target_explicit' requires init_source_layers to "
                "be a non-empty list of target layer indices."
            )
        return self
