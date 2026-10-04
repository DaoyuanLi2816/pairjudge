# Example judge: quality, costs and limits

This report evaluates the trained 0.5B example, independently of the historical competition ensemble. Numbers below are generated from auditable predictions on the same 1,000 human-labeled test rows (889 connected groups). No calibration or pseudo-labels were used.

## Frozen data and selection

Base: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), commit `7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0. Data: [Arena human preferences](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k), commit `18c298340948c0e7f7727399fd459cca6ce0ca6f`, Apache-2.0. Of 57,477 source rows, 490 invalid/empty/identical/duplicate rows were excluded without relabeling; 56,987 unique usable rows remained.

Normalize NFKC, whitespace and case; connect rows sharing a normalized prompt or unordered response pair, and remove reversed conversation duplicates. Frozen samples: train 8,000 / 7,209 groups; validation 769 / 686; test 1,000 / 889. Audited prompt, pair, conversation and component overlaps are zero across these splits. This is a holdout for this delivery, not proof against public-data/pretraining contamination.

[Split IDs and hashes](reports/v0.3.0/split-manifest.json) · [Frozen selection](reports/v0.3.0/selection.json) · [Export correction](reports/v0.3.0/export-protocol.json)

Two predeclared one-epoch seeds (42/43), 500 updates each, LoRA r=16/alpha=32, lr=1e-4, microbatch 2, accumulation 8, BF16 training, checkpointing, length 512. Actual deployed FP32 validation log-loss was 1.096389 / 1.082457; seed 43 was frozen before opening test results. No test-based reselection.

BF16 merging caused a maximum probability change of 0.09745 and 14/64 verdict differences. Exporting the same adapters by merging in CPU FP32 reduced maximum differences to 1.36e-5 / 1.07e-5 (64 validation rows, declared tolerance 3e-5). The corrected public weights and default inference are FP32. [Negative diagnostic](reports/v0.3.0/merge-diagnostic-seed42.json).

## Same-cohort test results

| Mode | Log-loss | Accuracy | Macro F1 | Brier (sum of 3 squared errors) |
| --- | ---: | ---: | ---: | ---: |
| uniform | 1.098612 | 34.1% | 0.1695 | 0.6667 |
| train_prior | 1.095393 | 36.4% | 0.1779 | 0.6646 |
| single | 1.066808 | 41.6% | 0.3837 | 0.6447 |
| swap_average | 1.058552 | 45.2% | 0.4208 | 0.6389 |

Swap minus single: log-loss -0.008256, paired 95% interval [-0.015395, -0.001596]; accuracy 3.6%, interval [1.03%, 6.25%]. These are 2,000 paired connected-group bootstrap replicates, measuring test-sampling uncertainty, not seed variability.

Single-pass aligned verdict flips: 47.1%; mean aligned absolute probability difference: 0.059353. Cached two-pass averaging has maximum exchanged-column discrepancy 0.0; fresh forward/reload checks separately allow explicit numerical tolerances. This removes presentation-order differences under deterministic inference; it does not establish global absence of bias or calibration.

| Class (human support) | Single precision / recall / F1 | Swap precision / recall / F1 |
| --- | --- | --- |
| a_wins (341) | 0.382 / 0.628 / 0.475 | 0.429 / 0.578 / 0.492 |
| b_wins (364) | 0.459 / 0.429 / 0.443 | 0.470 / 0.555 / 0.509 |
| tie (295) | 0.460 / 0.156 / 0.233 | 0.477 / 0.180 / 0.261 |

Tie recall is only about 18% in the averaged path. Arena tie includes both ties and both-bad judgments. A numerically equal maximum is `ambiguous`, not the learned tie class. Uniform accuracy uses the mechanical first-column argmax convention and is not a learned A preference.

## Slices

| Slice | Rows | Single log-loss / accuracy | Swap log-loss / accuracy |
| --- | ---: | --- | --- |
| truncated | 547 | 1.08047 / 39.7% | 1.07170 / 43.0% |
| untruncated | 453 | 1.05032 / 43.9% | 1.04268 / 47.9% |
| multi_turn | 115 | 1.11235 / 33.0% | 1.08718 / 36.5% |
| label_0 | 341 | 0.97945 / 62.8% | 0.99628 / 57.8% |
| label_1 | 364 | 1.03074 / 42.9% | 0.99205 / 55.5% |
| label_2 | 295 | 1.21230 / 15.6% | 1.21259 / 18.0% |
| packed_under256 | 210 | 1.07503 / 38.6% | 1.06504 / 46.2% |
| packed_256plus | 790 | 1.06462 / 42.4% | 1.05683 / 44.9% |

Slices are exploratory, overlap, and have no multiple-comparison correction. Length uses the packed original-order input token count, including framing; truncation uses actual packing diagnostics. The 512-token budget cuts 547/1,000 rows. No packing-quality or distillation gain is inferred from this evaluation.

## Cost on the actual user path

Native Windows 11, NVIDIA RTX 4080 (16 GiB), Torch 2.6.0+cu124 / Transformers 4.57.6, FP32 inference, microbatch 4. GPU was shared with existing work. End-to-end timings include CPU packing/padding and synchronized scoring, exclude model loading/download and warmup. Each measurement has one warmup and three repetitions; medians below. A pair uses the first test row; batch uses the first 64 fixed rows.

| Mode | One pair | Batch 64 | Throughput | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: | ---: |
| single | 48.0 ms | 2.409 s | 26.56 pairs/s | 2.023 GiB |
| swap_average | 99.4 ms | 4.833 s | 13.24 pairs/s | 2.023 GiB |

Model load from local disk: 5.00 s. Both 1,000-row passes: 75.16 s, excluding packing and loading. Selected training took 20.45 minutes and peak 1.282 GiB allocated. Downloaded model weights are 1,976,174,312 bytes; tokenizer adds about 16 MB.

Separate CUDA-event forward intervals exclude CPU packing and host-to-device transfer. They use original row order, while the deployed end-to-end path sorts by length before batching. They are different runs under contention, not a subtraction-based decomposition of wall time; intervals may exceed the sorted end-to-end measurement. Raw repetitions are retained in the report. Dual inference makes two scores; other deployments need their own wall-time measurements.

## Failure analysis

These are high-loss disagreements with the recorded human labels, selected by loss rather than favorable anecdotes. Source text was inspected by dataset ID; only paraphrased explanations are published. Human preference disagreement does not itself prove which answer is factually correct.

| ID | Human / averaged verdict | Inspection |
| --- | --- | --- |
| 2122284002 | B / A | A interprets an environmental slogan; B refuses. The judge favors an explanation while the human label favors refusal. This exposes preference-policy disagreement even without truncation. |
| 4247235353 | A / B | User asks to supply a story premise later. A asks for it; B writes a lengthy story immediately. The judge favors the premature long response. B is truncated. |
| 1448944105 | A / B | A distinguishes a novel from its setting; B elaborates an invented android-city narrative. The judge favors elaboration rather than the factual distinction. Truncation also removes narrative context. |
| 2578998626 | B / A | A gives a long procedure and suggests using a third-party company domain; B provides concise naming suggestions on a user-controlled domain. The judge rewards the longer but problematic procedure. |
| 1391479423 | A / B | A discusses a cross-language greeting pun; B overstates cultural correctness. This multi-turn case is truncated, reducing evidence about follow-up nuance. |

Do not use this small, uncalibrated model as a factuality verifier, safety gate, human-value oracle or high-stakes decision system. Long conversations, subtle humor and domain transfer are particularly uncertain. No external evaluation, calibrated confidence, larger ensemble or new distilled student quality was measured.

## Reproduce and audit

[Selected checkpoint predictions and full report](reports/v0.3.0/seed43/evaluation.json) · [Per-row predictions/diagnostics](reports/v0.3.0/seed43/predictions.jsonl). Reports contain IDs/groups/labels/probabilities, not original conversations.

Install `pairjudge[train]==0.3.0`, reconstruct the pinned source with `python examples/public_judge.py freeze`, and use the published seed configs with fresh output directories. Keep the frozen test out of training, calibration and selection. Training scripts are in the source repository, not required for normal PyPI inference. Costs are local measurements, not a service-level promise.

## Training variability

Seed 42 was evaluated on the identical frozen test after selecting seed 43 on validation. It is a replication diagnostic, not a test-driven replacement.

| Seed | Validation FP32 log-loss | Test single log-loss / accuracy | Test swap log-loss / accuracy |
| --- | ---: | --- | --- |
| 42 | 1.096389 | 1.080323 / 40.2% | 1.071563 / 42.9% |
| 43 | 1.082457 | 1.066808 / 41.6% | 1.058552 / 45.2% |

[Seed 42 report](reports/v0.3.0/seed42/evaluation.json). Two seeds do not estimate a reliable population distribution of training runs. The difference between seeds must not be replaced by bootstrap intervals from one test sample.
