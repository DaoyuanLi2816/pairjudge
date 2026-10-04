# Use a trained judge

Install `python -m pip install 'pairjudge[judge]==0.3.0'` in a Python 3.10+
environment. For CPU-only Torch, install from the official CPU wheel index
first. The first run downloads about 2 GB; it does not upload your input.

```bash
pairjudge compare --prompt "Why does ice float?" --a "Its crystal structure lowers its density." --b "Cold objects always rise." --swap --device cpu
pairjudge batch --input pairs.jsonl --output results.jsonl --swap
pairjudge demo --port 7860
```

The built-in model is a small example with measured limitations. An omitted
`--revision` pins its published commit automatically. Other Hub models are
resolved to a single snapshot before tokenizer/weights load. Local directories
must be trained bundles. A plain backbone is intentionally rejected.

JSONL keys are `id`, `prompt`, `response_a`, `response_b`. Pair columns may be
strings for a single turn, or equal-length lists of strings for multiple turns.
List entries cannot be null. Each response is the candidate's answer to the
corresponding shared prompt; histories must already be aligned. Flat strings
in the low-level packer/DataFrame API are invalid; the CLI explicitly converts
single-turn strings to lists. Duplicate IDs/indexes are allowed and retain
positional order. Empty files produce no predictions.

Results contain canonical `a_wins`, `b_wins`, `tie` probabilities; `verdict`;
model commit/device/dtype; and per-round original/kept content-token counts.
`ambiguous` means two numerical maxima within absolute tolerance `1e-8`, not a
learned preference tie. No confidence threshold or calibration is implied.

`--swap` computes both orders and averages after exchanging A/B columns. It
provides swap-equivariant distributions, with additional measured cost. To
inspect each direction separately, use the local demo or the Python API's
single-pass prediction on each input order.

```python
from pairjudge import PairwiseJudge, from_pairs
from pairjudge.catalog import DEFAULT_MODEL, DEFAULT_REVISION
judge = PairwiseJudge.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION,
                                     device="cpu", local_files_only=True)
data = from_pairs(["Explain gravity."], ["Mass attracts mass."], ["It is magic."])
print(judge.predict(data, swap_debias=True))
```

After the initial download, `--offline --cache-dir PATH` works from that same
cache. An empty cache in offline mode fails clearly. CPU supports FP32; CUDA
supports explicit FP32/FP16/BF16 with hardware checks. Auto uses one GPU if
available and respects the bundle's saved default precision. Quantization, remote code and multi-device maps are unqualified and
rejected. Adapter inference additionally needs `pairjudge[train]` for PEFT;
the public merged example needs only `[judge]`.

Both-blank answers and a budget unable to retain response content fail.
One-blank, identical answers and missing prompt context are diagnosed; no
fixed probability override is applied. CLI/UI cap pairs at 60,000 characters
and 64 rounds, HTTP bodies at 180,000 bytes, and CLI batches at 10,000 rows.
The local demo accepts one inference request at a time (concurrent requests
receive HTTP 429). It is loopback-only and has no remote fonts, scripts,
telemetry or conversation logging. It is a development demo, not a hosted
production service. The documentation site runs no model inference.
