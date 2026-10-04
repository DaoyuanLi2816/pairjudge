# Changes from 0.2

- New default `balanced_v2` reserves framing separately and redistributes spare
  content budget. Every trained bundle records its exact format. Use explicit
  `PackerConfig(packing_format="competition_v1")` for historical packing;
  `competition/` and the golden reference remain unchanged.
- `from_pretrained` requires `pairjudge.json` by default. It will no longer
  silently initialize a backbone's random classifier. Known legacy classifiers
  need explicit mapping/config; see [artifacts](artifacts.md).
- CPU defaults to FP32. CUDA honors the saved bundle's precision: the public
  merged example defaults to FP32; other auto bundles select supported BF16/FP16.
  Quantization, arbitrary
  model-loading kwargs and multi-device maps are rejected. Use `[train]` when
  loading an adapter. `device_map='cpu'/'auto'` and `torch_dtype` remain explicit
  single-device compatibility aliases; stored packer overrides must agree.
- Remote training requires an exact commit revision. Production training uses
  a sample-normalized explicit Torch loop; the internal `SoftLabelTrainer`
  helper was removed. No Trainer extension API is advertised.
- Arena loading preserves labels by default. Historical cleaning requires
  `legacy_cleaning=True` and explicit `drop_identical/relabel_empty` flags.
  That path converts historical empty encodings, drops both-empty rows and
  retains its historical labeling limitations.
- UltraFeedback loading keeps aligned multi-turn/system context and verifies
  that the prompt column matches the conversation. Unsupported roles or
  divergent shared user histories fail instead of disappearing silently.
  It now retains repeated prompts by default; `dedup_by_prompt=True` is explicit
  and records input/output/drop counts in DataFrame provenance.
- Pseudo-labeling uses the teacher's saved packer and adds provenance.
  Output files and model directories cannot be overwritten.
- `swap_debias` keeps its name for compatibility. Its documented property is
  swap equivariance after exchanging output columns, not global bias removal.
  Numerical probability ties are `ambiguous`, separate from learned `tie`.
