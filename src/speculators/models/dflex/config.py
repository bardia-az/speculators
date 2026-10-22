from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlexSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflex")
class DFlexSpeculatorConfig(DFlashSpeculatorConfig):
    """Configuration for the dflex speculator.

    dflex is DFlash with next-token (anchor-inclusive) output indexing: the
    draft output at block position ``i`` predicts the token at sequence
    position ``a + i + 1`` (``a`` is the anchor), so the anchor's own output
    is the first draft token and a block of ``block_size`` input positions
    drafts ``block_size`` tokens. DFlash instead discards the anchor output
    and drafts ``block_size - 1`` tokens from positions ``1..block_size-1``.

    Everything else (backbone, attention, target-feature injection, mask
    token, shared frozen embedding and LM head) is unchanged from DFlash, so
    this config only pins ``sample_from_anchor`` to ``True`` and gives the
    checkpoint its own ``speculators_model_type`` so inference cannot mistake
    a dflex checkpoint for a DFlash one.
    """

    speculators_model_type: Literal["dflex"] = "dflex"
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlexSpeculator"],
        description="Model architectures that can load these weights",
    )
    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Pinned to True: dflex samples from the anchor and all mask "
            "positions (block_size speculative tokens). Setting it to False "
            "is the DFlash layout -- train with --speculator-type dflash "
            "instead."
        ),
    )

    @model_validator(mode="after")
    def _validate_sample_from_anchor(self) -> "DFlexSpeculatorConfig":
        if not self.sample_from_anchor:
            raise ValueError(
                "DFlexSpeculatorConfig requires sample_from_anchor=True: "
                "dflex's anchor output is the first draft token. Use "
                "DFlashSpeculatorConfig (--speculator-type dflash) for the "
                "bonus-anchor layout."
            )
        return self
