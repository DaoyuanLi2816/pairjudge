# Changelog

## 0.3.0 - 2026-10-04

- Publish a separately evaluated, trained 0.5B example judge with fixed model
  and dataset revisions, legal provenance and anonymous download verification.
- Add versioned safetensors bundles binding trained heads, tokenizer, full
  packer semantics and arbitrary A/B/tie model class order. Reject ordinary
  backbones, incomplete weights and incompatible overrides in inference.
- Default to diagnosed `balanced_v2` content packing; preserve unchanged
  competition tokenization under explicit `competition_v1` golden tests.
- Preserve human labels by default and retain aligned multi-turn/system
  context. Historical cleaning is explicit. See `docs/migration.md`.
- Use a single-device Torch loop with correctly sample-normalized CE/KL
  accumulation and partial final groups; select on grouped validation and
  evaluate before adapter merge. Protect output directories and staged saves.
- Add CPU FP32 and explicit CUDA precision handling, teacher provenance,
  ordered JSONL CLI and a bounded loopback demo with no input uploads/logging.
- Distinguish swap equivariance, human preference accuracy, calibration and
  numerical ambiguity. Retain historical negative accuracy evidence.
- Test real offline hard/soft full/adapter round trips in fresh processes,
  core-only installations and minimum/current dependency combinations.
- Publish the same qualified wheel/sdist bytes through the existing OIDC
  workflow, then verify actual PyPI and model downloads outside the checkout.

## 0.2.0 - 2026-07-23

### Fixed

- Preserve the packer's `max_length` guarantee for custom configurations by
  rejecting negative, non-finite, or empty truncation ratios.
- Reject mismatched batched pair and winner columns instead of silently
  truncating them.
- Validate hard one-hot and soft probability labels before training.
- Return an empty `(0, 3)` array for empty inference inputs and report invalid
  batch sizes clearly.
- Compute evaluation log-loss against the full soft-label distribution.

### Changed

- Read the final assistant message from UltraFeedback-style conversations,
  including system-prompt and multi-turn records.
- Validate training configuration values before loading data or models.
- Create pseudo-label output directories automatically.
- Test Python 3.9, 3.12, and 3.14 in CI and validate release artifacts before
  publishing.
- Update package metadata to current SPDX and Python classifier conventions.
