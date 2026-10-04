# Train and distill

Use `pairjudge[train]` with Python 3.10+. Only one CPU or one CUDA device is
qualified. Start with a small local dataset and a fresh output directory.

```bash
python -m pairjudge.training --cfg examples/configs/quickstart.yaml
```

CSV uses Arena JSON-encoded lists; JSONL/parquet use actual lists. All three
winner columns are required. Hard labels are strict one-hot; soft labels must
be finite, nonnegative and sum to one. Default loaders do not relabel human
preferences or remove rows. Nulls/misaligned rounds are rejected. Public
experiment exclusions are separately audited in its frozen manifest.

Provide an exact 40-character `model_revision` for Hub backbones. Configure
`max_length`, `packer` (all other `PackerConfig` fields), and `label_order`.
`device: cpu` requires `dtype: float32` or `auto`; `bf16: true` is an explicit
compatibility alias for a BF16 request and is rejected on CPU. `bf16: false`
no longer means FP16. `dtype: auto` selects CPU FP32 or supported CUDA
BF16/FP16. `gradient_checkpointing` can bound CUDA activations.

Use a separate `validation_path`. Training rejects shared normalized prompts
or unordered response pairs. Without it, connected-group holdout uses
`eval_holdout` and `seed`; this is **validation**, not an independent final
test. Keep test and pseudo-label pools separate before training.

The explicit PyTorch loop sums CE or KL over each microbatch and divides by
the actual example count in the whole accumulation group. Its last smaller
group is normalized by its own count. AdamW, cosine schedule, warmup and
gradient norm clipping are explicit; FP16 overflows/nonfinite losses fail
without exporting success. No implicit distributed Trainer contract is used.
Validation runs at each completed/partial epoch; its lowest log-loss selects
the export. LoRA saves the trained classification head alongside adapters;
validation happens before merge changes the model object.
Merged exports move to CPU FP32 before applying adapter deltas. Qualification
must compare the reloaded adapter and merged model: BF16 merging changed the
public pilot's probabilities materially, whereas FP32 merging passed the
declared tolerance. This increases public weight storage to about 2 GB.

## Two phases

First train a hard-label teacher. Then label a separate, authorized pool:

```bash
python -m pairjudge.pseudo_label --model output/judge-quickstart/merged --data pool.parquet --out pool_pl.parquet --swap-debias
```

The output keeps full canonical probabilities, with teacher revision/artifact
hash, packer, precision and inference mode in parquet metadata and a JSON
sidecar. These are teacher labels, not independent human annotation.
For the student set `label_mode: soft`, retain human one-hot rows as valid soft
distributions, and add pseudo-labeled training rows only. Keep validation/test
human-only when reporting human preference quality. Record pool and teacher
provenance in the training config. `tests/test_user_paths.py` actually executes
teacher → pseudo-label → soft student using tiny offline models; it establishes
the pipeline, not a distillation quality improvement.

The public model experiment is reproducible via `examples/public_judge.py`:
freeze once, inspect manifest, reconstruct private text files from exact
dataset revision and recorded IDs, train using the published seed configs,
select on validation, then evaluate final test once. The frozen report contains
no conversation text. See [evaluation](evaluation.md).
