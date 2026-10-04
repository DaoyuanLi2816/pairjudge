# Compatibility and limitations

Core: Python 3.9+; no Torch. Locally verified Python 3.9.25 with NumPy 2.0.2 /
Pandas 2.3.3. CI also tests Python 3.14 core.

Model paths: Python 3.10+, single CPU FP32 or single CUDA. Two actual offline
training/loading combinations are tested:

| Combination | Torch | Transformers | PEFT | Accelerate |
| --- | --- | --- | --- | --- |
| Minimum | 2.6.0 | 4.57.6 | 0.18.1 | 1.12.0 |
| Current qualification | 2.14.1 CPU | 5.18.0 | 0.21.2 | 1.15.0 |

Local CPU tests used Python 3.12.13 on Windows; CI tests minimum on Linux
Python 3.10 and current on Linux Python 3.12. Real public training/inference
uses Windows RTX 4080, Torch 2.6.0+cu124 and the minimum combination.
Training is BF16; the corrected merged export and recommended inference are FP32.
Version ranges express API compatibility intent, not qualification of every
version/device/model Cartesian product. Qwen2.5-0.5B and an offline tiny GPT2
are exercised. Other compatible classifier families need their own measured
qualification; remote code, 8-bit loading, MPS and multiple devices are outside
the current scope.

The small example is uncalibrated, has no external benchmark guarantee, and
cannot reliably verify truth, safety or human values. Long text loses context
under its 512-token packer. Some rounds are omitted; even retained content may
be insufficient. Tie labels in Arena pool both preference ties and “both bad”
cases. The model can favor verbosity, style, or familiar domains. Use its
probabilities for experiments and diagnostics alongside human review.

The public Arena dataset may overlap previous experiments or pretraining.
The newly frozen test is held out from **this delivery's** training/selection;
it is not claimed to be an uncontaminated external benchmark. Two seeds expose
limited training variability, while grouped bootstrap describes uncertainty
from this particular test sample. Neither establishes cross-domain robustness.

The original competition ensemble, larger models, pseudo-label recipe and
award remain historical provenance; the released example does not reproduce
their scale or quality. No new distillation, packing-quality or calibration
gain is claimed without an ablation.
