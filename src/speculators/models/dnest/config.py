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
    indexing, same attention, same losses -- so it trains and serves through
    the same code paths. The additions are (a) optional initialisation of the
    draft decoder layers (and, optionally, the draft final norm) from chosen
    layers of the target model, and (b) an optional per-layer target context
    mode (``context_mode="per_layer"``) in which every draft layer conditions
    on its own extracted hidden state through its source layer's key/value
    path instead of DFlash's fused feature.

    Initialisation and copying happen exactly once, when the drafter is
    created for training via ``from_training_args``; loading a saved dnest
    checkpoint never re-copies target weights.

    Notation note for ``target_after_extracted``: ``aux_hidden_state_layer_ids``
    are indices into the verifier's ``hidden_states`` tuple (slot 0 = embedding
    output, slot ``i`` = output of 0-based decoder layer ``i - 1``), so the
    target layer that *consumes* the tap with id ``l`` is 0-based decoder
    layer ``l`` -- numerically the aux id itself. ``init_source_layers`` is
    always expressed in 0-based decoder layer numbers (the numbering the
    target checkpoint uses for ``model.layers.{i}``).

    Per-layer pairing: draft layer ``j`` (a copy of target layer ``s_j``)
    consumes the hidden state at ``s_j``'s input, which is tuple tap
    ``s_j``. In per_layer mode the configured ``aux_hidden_state_layer_ids``
    must therefore equal ``init_source_layers`` element-wise (they coincide
    automatically for ``target_after_extracted``).
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
            "cannot be expressed through a fixed-weight RMSNorm. Fused "
            "context mode only: per_layer has no fusion projection."
        ),
    )
    copied_lr_scale: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Learning-rate multiplier applied to parameters initialised from "
            "the target (optimizer parameter group). Backward-compatible "
            "alias of lr_scale_copied; the two are kept equal and both are "
            "written to saved configs. 1.0 (default) trains everything at "
            "the base learning rate."
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
    context_mode: Literal["fused", "per_layer"] = Field(
        default="fused",
        description=(
            "How the draft layers condition on the target. 'fused' (default): "
            "DFlash's fused feature -- the extracted hidden states are "
            "concatenated, projected by the fusion layer, normalised by "
            "hidden_norm, and every layer turns that same feature into its "
            "context keys/values. 'per_layer': each draft layer j receives "
            "its own extracted hidden state (the input of its source layer "
            "s_j, tuple tap s_j) and computes context keys/values through "
            "its key/value path exactly the way the source layer would "
            "(input_layernorm -> k/v_proj -> k_norm -> RoPE). See kv_mode "
            "for who owns that path."
        ),
    )
    kv_mode: Literal[
        "frozen_shared", "split", "split_trainable", "trainable_shared"
    ] = Field(
        default="frozen_shared",
        description=(
            "Ownership/trainability of the per-layer key/value path "
            "(per_layer context mode only). 'frozen_shared': one path per "
            "layer (the layer's input_layernorm + k_proj + v_proj + k_norm), "
            "frozen, used for both context and block -- context keys/values "
            "are exactly the target's. 'split': two paths; the context path "
            "(self_attn.context_kv) is a frozen copy, the block path is "
            "trainable. 'split_trainable': like 'split', but the context "
            "copy is trainable too (lr_scale_context_kv), so context "
            "keys/values may drift from the target's; random init is also "
            "accepted (the context path is then randomly initialised). "
            "'trainable_shared': one shared, trainable path. The frozen "
            "modes (frozen_shared, split) require a target-derived "
            "init_mode."
        ),
    )
    lr_scale_new: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Learning-rate multiplier for newly created (non-copied, "
            "trainable) parameters. In per_layer mode with a target init "
            "this is typically only the draft final norm; with init_mode "
            "'random' it covers layer weights outside the dedicated key/value "
            "groups. Frozen parameters are "
            "excluded from the optimizer entirely (requires_grad=False), "
            "never 'frozen' with a zero scale."
        ),
    )
    lr_scale_copied: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Learning-rate multiplier for copied weights outside the "
            "key/value paths (q_proj, o_proj, q_norm, post_attention_layernorm, "
            "MLP, and in 'split'/'split_trainable' the block path's "
            "input_layernorm). "
            "copied_lr_scale is the backward-compatible alias; both names "
            "are kept equal on saved configs. Also the active field in "
            "fused context mode (where the whole layer copy is scaled, as "
            "before)."
        ),
    )
    lr_scale_context_kv: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Learning-rate multiplier for the trainable context key/value "
            "path (input_layernorm + k_proj + v_proj + k_norm): the shared "
            "path in kv_mode 'trainable_shared', the context_kv copy in "
            "'split_trainable'."
        ),
    )
    lr_scale_block_kv: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Learning-rate multiplier for the block key/value path "
            "(k_proj + v_proj + k_norm), kv_mode 'split'/'split_trainable' "
            "only."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_copied_lr_alias(cls, data):
        """Keep copied_lr_scale (old name) and lr_scale_copied (new name) in
        lockstep: both are written to saved configs, either is accepted on
        load, and disagreement is an error rather than a silent winner."""
        if not isinstance(data, dict):
            return data
        new = data.get("lr_scale_copied")
        old = data.get("copied_lr_scale")
        if new is not None and old is not None and float(new) != float(old):
            raise ValueError(
                f"copied_lr_scale ({old}) and lr_scale_copied ({new}) "
                "disagree; pass only one of the two."
            )
        effective = new if new is not None else old
        if effective is not None:
            data = {
                **data,
                "lr_scale_copied": float(effective),
                "copied_lr_scale": float(effective),
            }
        return data

    @model_validator(mode="after")
    def _validate_explicit_sources(self) -> "DNestSpeculatorConfig":
        if self.init_mode == "target_explicit" and not self.init_source_layers:
            raise ValueError(
                "init_mode='target_explicit' requires init_source_layers to "
                "be a non-empty list of target layer indices."
            )
        return self

    @model_validator(mode="after")
    def _validate_per_layer(self) -> "DNestSpeculatorConfig":
        if self.context_mode != "per_layer":
            # Value-based guard: fused mode keeps DFlash's exact behaviour,
            # so the per_layer knobs must hold their (inert) defaults --
            # anything else is a misconfiguration. Values, not "was the key
            # present", because every key of a serialized config counts as
            # set when it is loaded back.
            misplaced = [
                name
                for name, value in (
                    ("kv_mode", self.kv_mode),
                    ("lr_scale_new", self.lr_scale_new),
                    ("lr_scale_context_kv", self.lr_scale_context_kv),
                    ("lr_scale_block_kv", self.lr_scale_block_kv),
                )
                if (name == "kv_mode" and value != "frozen_shared")
                or (name != "kv_mode" and value != 1.0)
            ]
            if misplaced:
                raise ValueError(
                    f"{', '.join(misplaced)} only apply with "
                    "context_mode='per_layer' (the fused mode is frozen to "
                    "DFlash's exact behaviour)."
                )
            return self

        if (
            self.kv_mode in ("frozen_shared", "split")
            and self.init_mode == "random"
        ):
            raise ValueError(
                f"kv_mode='{self.kv_mode}' freezes a context key/value path "
                "copied from the target, which requires a target-derived "
                "init_mode. For a per-layer ablation without target weights "
                "use kv_mode='trainable_shared' or 'split_trainable' (a "
                "trainable context path may also start from random init)."
            )
        if (
            self.kv_mode not in ("trainable_shared", "split_trainable")
            and self.lr_scale_context_kv != 1.0
        ):
            raise ValueError(
                "lr_scale_context_kv scales the trainable context key/value "
                "path and only applies with kv_mode='trainable_shared' or "
                f"'split_trainable' (got kv_mode='{self.kv_mode}'; the path "
                "is "
                + (
                    "frozen, not optimized"
                    if self.kv_mode == "frozen_shared"
                    else "the frozen context copy"
                )
                + ")."
            )
        if (
            self.kv_mode not in ("split", "split_trainable")
            and self.lr_scale_block_kv != 1.0
        ):
            raise ValueError(
                "lr_scale_block_kv scales the block key/value path and only "
                "applies with kv_mode='split' or 'split_trainable' (got "
                f"kv_mode='{self.kv_mode}')."
            )
        if self.fusion_init == "passthrough_deepest":
            raise ValueError(
                "fusion_init='passthrough_deepest' initialises the fusion "
                "projection, which does not exist in per_layer context mode."
            )

        num_draft_layers = self.transformer_layer_config.num_hidden_layers
        aux_ids = self.aux_hidden_state_layer_ids
        if self.init_mode == "random":
            got = len(aux_ids) if aux_ids is not None else 0
            if got != num_draft_layers:
                raise ValueError(
                    "per_layer context gives draft layer j the j-th "
                    f"extracted hidden state, so init_mode='random' needs "
                    f"exactly one tap per draft layer: {num_draft_layers} "
                    f"taps expected, got {got} ({aux_ids})."
                )
            return self
        if aux_ids is None:
            raise ValueError(
                "per_layer context requires aux_hidden_state_layer_ids "
                "(the extracted taps; training arg --target-layer-ids) to "
                "be set."
            )

        # Pairing is enforced, not trusted: draft layer j must consume the
        # input of its source layer s_j, which is tuple tap s_j.
        if aux_ids != self.init_source_layers:
            raise ValueError(
                "per_layer context requires aux_hidden_state_layer_ids to "
                "equal init_source_layers element-wise: draft layer j "
                f"(a copy of target layer s_j) consumes tuple tap s_j, the "
                f"input of its source layer. Got taps {aux_ids} but sources "
                f"{self.init_source_layers} (init_mode="
                f"'{self.init_mode}'). Pass --target-layer-ids "
                f"\"{' '.join(str(s) for s in self.init_source_layers)}\" "
                "to BOTH the trainer and launch_vllm.py, and regenerate any "
                "hidden states extracted with different ids."
            )
        return self
