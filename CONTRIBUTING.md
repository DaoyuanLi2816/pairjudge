# Contributing

Use an isolated Python 3.10+ environment and install `pip install -e '.[test]'`.
Run `pytest -q`, `ruff check src tests examples/public_judge.py`, and
`ruff format --check src tests examples/public_judge.py`. Core-only checks use
Python 3.9+ and need no Torch. Tiny model tests are offline; their synthetic
inputs verify software behavior, not judge quality.

Keep `competition/` and `tests/reference_impl.py` unchanged. New packing
semantics must be versioned and saved with the model. Include a regression for
actual content retention, classifier weights, or class mapping when relevant.
Do not change frozen test splits to improve metrics. Report paired comparisons
from the same samples and preserve negative results.

Useful contributions include a legally distributable external evaluation set,
another tested tokenizer/model family, or measured CPU/CUDA inference costs.
State exact versions, hardware, data rights and limitations. Avoid unsupported
accuracy or calibration claims.

Report bugs using the issue forms. Share a minimal **synthetic or authorized**
example. Remove secrets and identifying/private text; reports are public.
The library and demo do not automatically upload user examples.
