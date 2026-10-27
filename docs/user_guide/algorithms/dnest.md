# dnest

dnest ("DFlash, nestable") is DFlash with **target-derived layer
initialisation**: it is architecturally identical to
[DFlash](dflash.md) — same anchored-block indexing (block position `k`
predicts the token at anchor `+ k`), fused target-feature injection,
attention, losses, and frozen shared embedding / LM head — except that its
draft decoder layers (and, optionally, its final norm) can be initialised
from chosen layers of the target model instead of randomly, so training
starts from the target's computation. It adds no heads and no new modules: a
dnest checkpoint is a DFlash checkpoint whose training began from copied
weights, and it serves through the `dflash` method unchanged.

The initialisation runs exactly once, when the drafter is created for
training (`from_training_args`); loading a saved dnest checkpoint — for
resuming, evaluation, or serving — never re-copies target weights.

## Init modes

`--init-mode` selects the source layers; the resolved list is always saved
on the checkpoint config as `init_source_layers` (0-based target decoder
layer numbers) for reproducibility, except `random`, which saves `null`:

| mode | source layers |
| ---- | ------------- |
| `random` | none — DFlash's initialisation (baseline) |
| `target_after_extracted` *(default)* | draft layer `j` from the target layer that consumes the `j`-th extracted hidden state |
| `target_last` | the target's last `num_layers` layers, in target order |
| `target_explicit` | `--init-source-layers` (must match `--num-layers`) |

**Notation:** `aux_hidden_state_layer_ids` are indices into the verifier's
`hidden_states` tuple (slot 0 = embedding output, slot `i` = output of
0-based decoder layer `i-1`), so the tap with id `l` is consumed by 0-based
decoder layer `l`. For the Qwen3-4B recipe (taps `2 10 18 26 34`),
`target_after_extracted` therefore resolves to sources `2 10 18 26 34` —
numerically the aux ids themselves, *not* `l + 1`. `target_last` resolves
to `31 32 33 34 35`.

Companion knobs (saved on the config):

- `--fusion-init passthrough_deepest` — the fusion projection initially
  returns the deepest extracted hidden state exactly (identity on that
  slice, zeros elsewhere); the following `hidden_norm` RMSNorm rescales it
  to unit RMS per token, the closest equivalent an exact passthrough allows
  through a fixed-weight norm.
- `--copied-lr-scale` — learning-rate multiplier for the copied parameters,
  implemented as a separate optimizer parameter group (default 1.0 = the
  DFlash optimizer, unchanged).
- `--init-final-norm` — also copy the target's final norm into the draft's
  final norm (before the LM head). Off by default: DFlash initialises that
  norm randomly. (The frozen `verifier_norm` used for target logits is a
  different module and is always loaded from the target.)

## What is copied and guarded

Per draft layer: `q/k/v/o` projections, `q_norm`/`k_norm`,
`input_layernorm`, `post_attention_layernorm`, MLP `gate/up/down` — 11
tensors with identical names on both sides (prefix `model.layers.{s}.` →
`layers.{j}.`). Not copied: fusion `fc`/`hidden_norm`, embedding, LM head.

Before anything is copied (all-or-nothing), the draft layer configuration
must match the target's on hidden size, attention/KV head counts, head dim,
MLP size, RMSNorm eps, attention bias, and RoPE settings; every required
tensor must exist with matching shapes; source ids must be valid, distinct,
and match the draft layer count. A draft/target attention-*type* difference
(sliding vs full) only warns — it changes masks, not parameters. Only the
needed tensors are read, from the target's safetensors shards via the index
file, on CPU.

## Serving

A dnest checkpoint's `speculators_model_type: dnest` is registered in
vLLM's speculators config loader and rewritten to `method: dflash` (the
same mechanism as `dflash2`), so it serves through the DFlash runtime with
no other changes.

## Example

```bash
torchrun --standalone -m speculators.train \
    --speculator-type dnest \
    --init-mode target_after_extracted \
    --num-layers 5 --block-size 16 \
    --target-layer-ids 2 10 18 26 34 \
    ...  # all other flags as DFlash
```
