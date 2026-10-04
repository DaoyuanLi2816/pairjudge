# 0.3.0 implementation and delivery record

Baseline: `06f77034918e7c891e1d6a1f763737750bb50e1b`, package 0.2.0.
The initial checkout was clean and synchronized; GitHub permissions include
admin, with no open PRs/issues or repository rulesets. The latest published
GitHub Release is v0.2.0. Hugging Face authenticated identity is `DaoyuanLi`;
no existing PairJudge model was found in that namespace.

## Investigation

| Lead | Current evidence and disposition |
| --- | --- |
| First-use model | Confirmed: README requires a local placeholder model; deliver an evaluated public checkpoint. |
| Random classification head | Confirmed in source; reproduce with a local causal backbone, then reject unmarked/missing heads during inference. |
| Packer/category persistence | Confirmed: only train_config is saved; loading defaults to a new packer and canonical class order. Add a versioned artifact contract. |
| CPU precision | Confirmed model builder chooses BF16/FP16 only; provide explicit CPU FP32 and validated single-device CUDA behavior. |
| Adapter and merge | PEFT SEQ_CLS normally saves classifier modules; test actual updated tensors and reloading. Evaluate before mutating the trainer reference. |
| Packing | Reproduce strings-as-rounds, zero ratios and framing-only output. Preserve the original algorithm as explicit competition_v1; use diagnosed balanced_v2 for new artifacts. |
| Data split | Random row splits do not audit duplicate prompts/reversed pairs. Freeze grouped splits before the pilot and keep test out of selection/pseudo-labeling. |
| Historical evaluation | Existing script stores aggregate-only metrics and unpinned random split; retain as historical diagnostics. |
| CI/release | Existing read-only test/build jobs and isolated OIDC upload are present. Add real artifact round trips and upload the qualified candidate without rebuilding. |

## Initial resource budget

Native Windows RTX 4080, 16,376 MiB total, 5,442 MiB free at initial inspection;
driver 596.49. Disk has approximately 479 GiB free. Do not interfere with other
GPU work. Start with a local tiny-model smoke, then a bounded real-data pilot.
Investigate cached, ungated Qwen2.5-0.5B-Instruct and Arena human preferences;
record exact revisions and official licenses before training/distribution.

## Validation, model and publication

Baseline tests: 71 passed. Final source qualification: 83 passed in each of
the minimum and current model stacks; coverage 79% on the minimum stack.
Python 3.9 core without Torch: 61 passed. Real CPU tests include hard/soft
training, unequal accumulation gradients, nondefault packer and class order,
changed heads, adapter/full exports, fresh-process loading and actual CLI/HTTP
paths. The human → pseudo-label → soft student smoke passed; it proves the
pipeline, not a new distillation gain. Ruff checks and format checks passed.

Actual GPU pilot: 128 human rows, 8 updates; then two fixed 8,000-row one-epoch
LoRA runs, 500 updates each, BF16, RTX 4080. Validation-selected seed 43 is
exported in FP32. Confirmed additional causes: base padding configuration
was absent when reloading adapters, and BF16 adapter merging materially
changed probabilities. Restore padding and merge on CPU FP32; the two actual
64-row merge comparisons passed 3e-5 maximum absolute probability tolerance.
Current Transformers 5.18.0 CPU inference differs from the minimum-stack
three-example reference by at most 1.073e-6 (allowed 2e-4).

The frozen 1,000-row / 889-group test gives seed 43 single/dual log-loss
1.066808 / 1.058552 and accuracy 41.6% / 45.2%. Both beat uniform and train-only
prior log-loss. Paired group-bootstrap intervals, seed 42 replication,
per-row diagnostics, measured costs and unfavorable failure cases are in
[evaluation](evaluation.md) and [reports](reports/v0.3.0/seed43/evaluation.json).
Tie recall, seed variability, 512-token truncation and public-data contamination
limits are retained. A cancelled slow CPU validation diagnostic supplies no
qualification evidence; no cancelled run is reported as completed.

Public model: `DaoyuanLi/pairjudge-qwen2.5-0.5b-v0.3`; Apache-2.0 weights/data
provenance, separate MIT code. Manifest SHA256
`d44941737da183539449e5a0fd46681d55e0f0c8422573722fdb21b09997e012`;
weights SHA256
`f3cca2bbde8fd089677be597497e5d5a890acde935390d98ac5e5ff48cff495c`.
The exact public revision is pinned in `pairjudge.catalog`; distribution-byte
and public-download receipts are added after actual verification, without
rebuilding the qualified release files. The existing OIDC publisher workflow
filename/binding and platform attestations are preserved.

Attribution uses Daoyuan Li's existing repository identity. No extra author or
coauthor is added. Original competition files and golden reference are
unchanged; Qwen/Alibaba Cloud and Arena notices remain in the model repository.
