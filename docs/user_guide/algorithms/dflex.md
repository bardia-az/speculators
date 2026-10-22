# dflex

dflex ("DFlash, flexible indexing") is DFlash with **next-token output
indexing**: it is architecturally identical to [DFlash](dflash.md) — same
backbone and number of draft layers, bidirectional attention within blocks,
target-context visibility, anchor-offset position ids, mask token, shared
frozen embedding and LM head, target-feature injection — except for what each
block position predicts. It exists to make DFlash-style block drafting a pure
next-token predictor, so a block spends its last input position on a draft
token instead of a discarded anchor output.

It adds no heads and no new modules: a dflex checkpoint is a DFlash
checkpoint with different output indexing.

## The indexing difference

With the anchor at sequence position `a` and `B = block_size` input positions
`[anchor, MASK x (B-1)]`:

| Model  | Output at block position k predicts | Predictions per block |
| ------ | ----------------------------------- | --------------------- |
| DFlash | token at `a + k` for `k = 1..B-1`; the anchor output (`k = 0`) is unused | `B - 1` |
| dflex  | token at `a + k + 1` for `k = 0..B-1`; the anchor output is the first draft token | `B` |

Everything downstream follows from this:

- **Labels:** `label[k] = token[a + k + 1]` for `k = 0..B-1`, with the loss
  mask gathered at those same sequence positions.
- **Target distributions:** the target logits at sequence position `t`
  predict token `t + 1`, so the target distribution for dflex block position
  `k` comes from target position `a + k`. Equivalently, dflex position `k`
  uses exactly what DFlash used for its slot `k + 1`, and position `B - 1`
  uses one target position further.
- **Anchor sampling:** an anchor needs room for `B` labels (`a + B` within
  the valid, same-document, loss-masked range) — one more than DFlash.
- **Position decay weights:** weight by draft depth (distance from the
  anchor): `w_k = exp(-k / gamma)`, and the anchor's prediction gets weight
  1.
- **Metrics:** per-position accuracy and acceptance proxies use the
  next-token alignment; `position_0_acc` is the anchor's own prediction.

## Block-size semantics

`--block-size B` is the number of **input positions** per block:

- DFlash with `--block-size 16` drafts **15** tokens per block.
- dflex with `--block-size 15` drafts **15** tokens per block, using one
  fewer position.
- dflex with `--block-size 16` drafts **16** tokens per block.

Compare DFlash and dflex at equal numbers of draft tokens (e.g. DFlash b16 vs
dflex b15), or at equal block sizes when you want dflex's extra draft token.

## Training

```bash
python -m speculators.train \
    --speculator-type dflex \
    --block-size 15 \
    ... # all other flags exactly as for DFlash
```

`--speculator-type dflex` implies `sample_from_anchor=True` (passing
`--no-sample-from-anchor` is rejected — that layout is DFlash itself) and
inherits DFlash's default recipe (5 draft layers, cross-entropy loss, D-PACE
position weighting, `block_size 16` unless overridden). Checkpoints record
both `speculators_model_type: "dflex"` and `sample_from_anchor: true`, so an
inference engine cannot mistake the indexing.

## Serving

dflex checkpoints need an inference engine whose DFlash runtime supports
anchor-inclusive sampling. In vLLM, serve them with the dedicated `dflex`
method and `num_speculative_tokens` equal to the block size:

```bash
vllm serve <verifier> --speculative-config \
    '{"method": "dflex", "model": <dflex checkpoint>, "num_speculative_tokens": 15}'
```

Stock vLLM `dflash` serving rejects `sample_from_anchor=True` checkpoints.

## See Also

- [DFlash](dflash.md) -- the base algorithm, bonus-anchor indexing
- [DSpark](dspark.md) -- anchor-inclusive indexing plus Markov and confidence
  heads (dflex shares the indexing, not the heads)
