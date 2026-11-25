import logging
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dnest.config import DNestSpeculatorConfig
from speculators.models.dnest.model_definitions import (
    CONTEXT_KV_SUFFIXES,
    DNestDecoderLayer,
)
from speculators.models.dnest.weight_copy import (
    BLOCK_KV_SUFFIXES,
    COPIED_LAYER_SUFFIXES,
    KV_PATH_SUFFIXES,
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

    Architecturally identical to DFlash in the default ``fused`` context
    mode: same anchored-block indexing (block position k predicts token
    anchor + k), fused target-feature injection, attention, losses, and
    frozen shared embedding / LM head. The additions over DFlash are
    target-derived initialisation (``init_mode``, see
    :class:`DNestSpeculatorConfig`) and the optional ``per_layer`` context
    mode, where each draft layer j conditions on the extracted hidden state
    at its source layer's input (tuple tap ``s_j``) and computes its context
    keys/values through its key/value path exactly as the source layer
    would -- so with a frozen path the context keys/values are exactly the
    target's own.

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
        self._apply_context_freezing()
        if config.context_mode == "per_layer":
            # per_layer checkpoints have no fusion projection; loading one
            # must not warn about the absent tensors.
            self.__dict__["_keys_to_ignore_on_load_missing"] = [
                *self.__dict__["_keys_to_ignore_on_load_missing"],
                "fc.weight",
                "hidden_norm.weight",
            ]

    # --- per-layer context plumbing (context_mode="per_layer") ------------

    @property
    def _use_fusion(self) -> bool:
        return self.config.context_mode != "per_layer"

    def _make_decoder_layer(
        self, config: DNestSpeculatorConfig, layer_idx: int
    ) -> DNestDecoderLayer:
        if config.context_mode != "per_layer":
            return super()._make_decoder_layer(config, layer_idx)  # type: ignore[return-value]
        return DNestDecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            split=config.kv_mode in ("split", "split_trainable"),
        )

    def _context_features(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """One raw extracted hidden state per draft layer (tap j for layer j).

        The concatenated training tensor's slice j is tap j (the input of
        source layer s_j), exactly what the layer's context key/value path
        expects; the layer applies its own ``input_layernorm``. In fused mode
        the base fusion (``fc`` + ``hidden_norm``) must run unchanged.
        """
        if self.config.context_mode != "per_layer":
            return super()._context_features(hidden_states)
        config = self.config
        hidden_size = config.transformer_layer_config.hidden_size
        num_layers = len(self.layers)
        expected = num_layers * hidden_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"per_layer context gives each draft layer one extracted "
                f"hidden state, so it expects a [{expected} = {num_layers} "
                f"layers x {hidden_size} hidden] feature tensor, got "
                f"{hidden_states.shape[-1]}. Check that the data pipeline's "
                "target_layer_ids matches the model's "
                "aux_hidden_state_layer_ids."
            )
        return [
            hidden_states[:, :, j * hidden_size : (j + 1) * hidden_size]
            for j in range(num_layers)
        ]

    def _apply_context_freezing(self) -> None:
        """Freeze the context key/value path per kv_mode (frozen modes).

        Frozen tensors keep ``requires_grad=False`` and are thereby excluded
        from every optimizer -- they carry no optimizer state. Derived from
        the config, so checkpoint loads (resume, serving) re-freeze
        identically.
        """
        config = self.config
        if config.context_mode != "per_layer":
            return
        if config.kv_mode == "frozen_shared":
            names = [
                f"layers.{j}.{suffix}"
                for j in range(len(self.layers))
                for suffix in KV_PATH_SUFFIXES
            ]
        elif config.kv_mode == "split":
            names = [
                f"layers.{j}.self_attn.context_kv.{suffix}"
                for j in range(len(self.layers))
                for suffix in CONTEXT_KV_SUFFIXES
            ]
        else:  # trainable modes (split_trainable, trainable_shared): nothing frozen
            return
        for name in names:
            self.get_parameter(name).requires_grad_(False)
        logger.info(
            "dnest: froze %d context key/value tensors (kv_mode=%s); they "
            "are excluded from the optimizer.",
            len(names),
            config.kv_mode,
        )

    def load_verifier_weights(self):
        """Reconstruct verifier-shared weights, then re-freeze context paths.

        HF's ``from_pretrained`` resets ``requires_grad=True`` on all
        parameters, so the freeze must be re-applied after any load.
        """
        super().load_verifier_weights()
        self._apply_context_freezing()

    # --- construction -------------------------------------------------------

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
        ``copied_lr_scale``/``lr_scale_copied``, ``init_final_norm``,
        ``context_mode``, ``kv_mode``, ``lr_scale_new``,
        ``lr_scale_context_kv``, ``lr_scale_block_kv``), which are stored on
        the saved config for reproducibility.
        """
        init_mode = kwargs.pop("init_mode", None) or "target_after_extracted"
        explicit_sources = kwargs.pop("init_source_layers", None)
        fusion_init = kwargs.pop("fusion_init", None) or "random"
        copied_lr_scale = kwargs.pop("copied_lr_scale", None)
        lr_scale_copied = kwargs.pop("lr_scale_copied", None)
        init_final_norm = bool(kwargs.pop("init_final_norm", False) or False)

        context_mode = kwargs.pop("context_mode", None) or "fused"
        kv_mode = kwargs.pop("kv_mode", None) or "frozen_shared"
        lr_scale_new = kwargs.pop("lr_scale_new", None)
        lr_scale_context_kv = kwargs.pop("lr_scale_context_kv", None)
        lr_scale_block_kv = kwargs.pop("lr_scale_block_kv", None)

        # copied_lr_scale (old name) and lr_scale_copied (new name) are one
        # knob; the config keeps both in lockstep.
        effective = 1.0
        for value in (copied_lr_scale, lr_scale_copied):
            if value is not None:
                effective = float(value)
        copied_lr_scale = lr_scale_copied = effective

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

        config_kwargs: dict = {
            "init_mode": init_mode,
            "init_source_layers": source_layers,
            "fusion_init": fusion_init,
            "copied_lr_scale": copied_lr_scale,
            "lr_scale_copied": lr_scale_copied,
            "init_final_norm": init_final_norm,
            "context_mode": context_mode,
        }
        if context_mode == "per_layer":
            # per_layer-only fields; the fused mode must stay constructible
            # with exactly DFlash's field set (validators reject these keys
            # there).
            config_kwargs["kv_mode"] = kv_mode
            if lr_scale_new is not None:
                config_kwargs["lr_scale_new"] = float(lr_scale_new)
            if lr_scale_context_kv is not None:
                config_kwargs["lr_scale_context_kv"] = float(lr_scale_context_kv)
            if lr_scale_block_kv is not None:
                config_kwargs["lr_scale_block_kv"] = float(lr_scale_block_kv)

        config = DNestSpeculatorConfig(**config_kwargs, **base_config_kwargs)

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
                    include_context_kv=(
                        context_mode == "per_layer"
                        and kv_mode in ("split", "split_trainable")
                    ),
                )
            )
        if fusion_init == "passthrough_deepest":
            init_fusion_passthrough_deepest(model.fc, aux_layer_ids)
        return model

    # --- optimizer groups ---------------------------------------------------

    def optimizer_param_lr_scales(self) -> dict[str, float]:
        """Per-parameter learning-rate multipliers for optimizer grouping.

        Fused mode: exactly the previous dnest grouping -- copied parameters
        at ``copied_lr_scale`` (== ``lr_scale_copied``), everything else at
        the base learning rate -- so existing checkpoints resume with
        bit-identical optimizer groups.

        per_layer mode groups by role: the key/value path per ``kv_mode``
        (shared path at ``lr_scale_context_kv``; the split block path at
        ``lr_scale_block_kv``; the split_trainable context copy at
        ``lr_scale_context_kv``), the remaining copied weights at
        ``lr_scale_copied``, and non-copied trainable parameters (the draft
        final norm when not target-initialised; every layer parameter in
        random mode) at ``lr_scale_new``. Frozen tensors are absent: they
        are ``requires_grad=False`` and never reach an optimizer.

        Derived purely from the saved config -- not from runtime state -- so
        a run resumed from a checkpoint rebuilds bit-identical optimizer
        groups and its saved optimizer state loads cleanly.
        """
        config: DNestSpeculatorConfig = self.config  # type: ignore[assignment]
        if config.context_mode != "per_layer":
            if config.init_mode == "random" or config.copied_lr_scale == 1.0:
                return {}
            scales: dict[str, float] = {}
            num_draft_layers = config.transformer_layer_config.num_hidden_layers
            for draft_idx in range(num_draft_layers):
                for suffix in COPIED_LAYER_SUFFIXES:
                    scales[f"layers.{draft_idx}.{suffix}"] = (
                        config.copied_lr_scale
                    )
            if config.init_final_norm:
                scales["norm.weight"] = config.copied_lr_scale
            return scales
        return self._per_layer_lr_scales(config)

    def _per_layer_lr_scales(
        self, config: DNestSpeculatorConfig
    ) -> dict[str, float]:
        kv_path = set(KV_PATH_SUFFIXES)
        block_kv = set(BLOCK_KV_SUFFIXES)
        scales: dict[str, float] = {}
        num_draft_layers = config.transformer_layer_config.num_hidden_layers
        for draft_idx in range(num_draft_layers):
            prefix = f"layers.{draft_idx}."
            if config.kv_mode == "split_trainable":
                # The context_kv path is always trainable and always carries
                # lr_scale_context_kv -- copied from the source layer with a
                # target init, randomly initialised with init_mode='random'
                # -- unlike split, where it is frozen and never reaches an
                # optimizer.
                for suffix in CONTEXT_KV_SUFFIXES:
                    scales[f"{prefix}self_attn.context_kv.{suffix}"] = (
                        config.lr_scale_context_kv
                    )
            if config.init_mode == "random":
                # Nothing is copied: every layer parameter is new.
                for suffix in COPIED_LAYER_SUFFIXES:
                    scales[prefix + suffix] = config.lr_scale_new
                continue
            for suffix in COPIED_LAYER_SUFFIXES:
                if config.kv_mode == "frozen_shared" and suffix in kv_path:
                    continue  # frozen: excluded from the optimizer entirely
                if config.kv_mode in ("split", "split_trainable"):
                    scales[prefix + suffix] = (
                        config.lr_scale_block_kv
                        if suffix in block_kv
                        else config.lr_scale_copied
                    )
                elif config.kv_mode == "trainable_shared" and suffix in kv_path:
                    scales[prefix + suffix] = config.lr_scale_context_kv
                else:
                    scales[prefix + suffix] = config.lr_scale_copied
        scales["norm.weight"] = (
            config.lr_scale_copied
            if config.init_final_norm
            else config.lr_scale_new
        )
        return scales
