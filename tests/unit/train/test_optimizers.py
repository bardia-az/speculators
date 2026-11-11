"""Regression tests for optimizer construction through DDP wrappers.

The trainer wraps the model in ``DistributedDataParallel`` *before* calling
``build_optimizers``. DDP neither forwards arbitrary attribute lookups (a
model's ``optimizer_param_lr_scales`` hook is invisible on the wrapper) nor
preserves parameter names (it prefixes ``module.``), which silently dropped
per-parameter LR scaling in every DDP run. These tests pin the unwrap.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from speculators.train.optimizers import build_optimizers


class _ScaledDraft(nn.Module):
    """Tiny stand-in shaped like a dnest drafter: a layer stack whose weights
    carry an LR-scale hook, plus a never-scaled ``fc`` module."""

    def __init__(self):
        super().__init__()
        # 2D matrices (-> Muon) with 1D biases (-> AdamW)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(5)])
        # 1D vectors (-> AdamW)
        self.norms = nn.ModuleList([nn.RMSNorm(8) for _ in range(5)])
        # never scaled: 2D weight (-> Muon) + 1D bias (-> AdamW)
        self.fc = nn.Linear(8, 8)

    def optimizer_param_lr_scales(self):
        scales: dict[str, float] = {}
        for i, _layer in enumerate(self.layers):
            scales[f"layers.{i}.weight"] = 0.1
            scales[f"layers.{i}.bias"] = 0.1
        for i, _norm in enumerate(self.norms):
            scales[f"norms.{i}.weight"] = 0.1
        return scales


@pytest.fixture
def gloo_world1(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


def test_ddp_wrapper_hides_hook_and_renames_params(gloo_world1):
    """Documents why ``build_optimizers`` must unwrap: the wrapper hides the
    model's LR-scale hook and prefixes parameter names."""
    wrapped = DistributedDataParallel(_ScaledDraft())
    assert getattr(wrapped, "optimizer_param_lr_scales", None) is None
    first_name = next(iter(dict(wrapped.named_parameters())))
    assert first_name.startswith("module.")


def test_ddp_wrapper_keeps_adamw_lr_scale_groups(gloo_world1):
    model = _ScaledDraft()
    wrapped = DistributedDataParallel(model)
    config = SimpleNamespace(optimizer="adamw", lr=1e-3, weight_decay=0.0)

    (optimizer,) = build_optimizers(wrapped, config)

    lrs = sorted(round(group["lr"], 10) for group in optimizer.param_groups)
    assert lrs == sorted([1e-3, 1e-4])
    scaled = next(g for g in optimizer.param_groups if g["lr"] < 1e-3)
    assert len(scaled["params"]) == 15  # 10 layer tensors + 5 norms
    # The optimizer must hold the wrapped module's tensors -- those are the
    # same tensors DDP synchronizes gradients for.
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in model.parameters()}


def test_ddp_wrapper_keeps_muon_lr_scale_groups(gloo_world1):
    wrapped = DistributedDataParallel(_ScaledDraft())
    config = SimpleNamespace(
        optimizer="muon",
        lr=1e-3,
        weight_decay=0.0,
        muon_lr=1e-2,
        muon_momentum=0.95,
        muon_weight_decay=0.1,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
    )

    muon, adamw = build_optimizers(wrapped, config)

    # Muon side: matrices only -- fc.weight at base, 5 layer weights at 0.1x.
    muon_lrs = sorted(round(g["lr"], 10) for g in muon.param_groups)
    assert muon_lrs == sorted([1e-2, 1e-3])
    scaled_muon = next(g for g in muon.param_groups if g["lr"] < 1e-2)
    assert len(scaled_muon["params"]) == 5

    # AdamW side: vectors -- fc.bias at base, 5 biases + 5 norms at 0.1x.
    adamw_lrs = sorted(round(g["lr"], 10) for g in adamw.param_groups)
    assert adamw_lrs == sorted([1e-3, 1e-4])
    scaled_adamw = next(g for g in adamw.param_groups if g["lr"] < 1e-3)
    assert len(scaled_adamw["params"]) == 10


def test_bare_model_unaffected_by_unwrap():
    """A model without the hook still builds a single group, wrapped or not."""
    plain = nn.Linear(8, 8)
    config = SimpleNamespace(optimizer="adamw", lr=1e-3, weight_decay=0.0)
    (optimizer,) = build_optimizers(plain, config)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert isinstance(optimizer, torch.optim.AdamW)
