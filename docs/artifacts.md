# Model bundle contract

`pairjudge.json` format version 1 binds the safe weights, tokenizer, full
`PackerConfig`, model class order, trained-head update receipt, source revision,
library version and dependency versions. Every inference file has a SHA-256
digest. Hashes detect accidental/mismatched files; a self-authored manifest is
not an independent training certificate or an authenticity signature.

Two formats are supported: `full` (`model.safetensors`, possibly sharded) and
`adapter` (`adapter_model.safetensors` plus a trained `modules_to_save` head).
Adapters depend on an exact base revision. Local tiny bases may be embedded
as safe data-only files under `base/`. The public example is a merged full
model, so it does not require an adapter fetch or PEFT at inference.

The manifest includes all field templates, round headers, terminal
instruction, ellipsis, BOS/EOS switches, length/ratios/minimum tail and format.
Tokenizer pad/eos/bos IDs and padding side are checked at load. Stored class
order may be any permutation of `a_wins/b_wins/tie`; model `id2label/label2id`
uses that order, while public output always uses canonical A/B/tie order.

Default loading restores the saved contract. A conflicting packer or class
override fails. Missing/newly initialized weights and an absent head are
rejected. Safe weights are required, remote code is disabled, and URL-style
commands are not accepted. Hub resolution pins all files to one snapshot;
offline loading cannot silently fetch missing files.

Training validates data before expensive model loading. Saves occur in a
temporary sibling directory; a final name is published only after the bundle
validates. Existing output directories are refused. A failed export is named
`.failed-*`, with no successful final bundle. A training failure/OOM before
export creates no final artifact. Best weights are selected on validation
before saving/merging. LoRA merge converts to FP32 on CPU; its full bundle
defaults to FP32 inference. BF16 merge was observed to materially change
probabilities and is not treated as a faithful round trip. These exports do **not** contain optimizer/RNG state
and are not resumable training checkpoints.

To re-save a verified full bundle, call `judge.save_pretrained(NEW_DIRECTORY)`.
The source license and model card must travel with redistributed weights;
package MIT licensing does not replace upstream weight/data licensing.

For a known legacy classifier, explicitly provide its documented class order
and full packer config:

```python
judge = PairwiseJudge.from_pretrained(
    "known-legacy-directory", device="cpu", allow_legacy=True,
    label_order=["a_wins", "b_wins", "tie"],
    packer_config=PackerConfig(packing_format="competition_v1", max_length=2048),
)
```

Legacy loading still rejects missing classifier weights and conflicting model
mapping. It does not invent a head-update receipt or establish model quality.
Convert only artifacts whose training/source rights you can verify; train a
new bundle when the old format/mapping is unknown.

Some 0.2/external classifiers have opaque config names such as `LABEL_0`.
Passing an order does not silently rewrite a conflicting model config. If you
have established the actual trained column meanings, declare them in a **new
copy** before the legacy call; retain all other config, weights, tokenizer and
legal files:

```python
import json
import shutil
from pathlib import Path

source = Path("known-legacy-directory")
target = Path("declared-legacy-directory")  # copytree refuses an existing target
order = ["a_wins", "b_wins", "tie"]  # use the verified original training order
shutil.copytree(source, target)
path = target / "config.json"
config = json.loads(path.read_text(encoding="utf-8"))
config["id2label"] = {str(i): label for i, label in enumerate(order)}
config["label2id"] = {label: i for i, label in enumerate(order)}
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
```

Load that copy with `allow_legacy=True`, the same `order`, and the **complete
original** `PackerConfig`. Confirm padding/special tokens and compare against
the original environment's predictions before distribution. This is a caller
declaration, not proof that an arbitrary external head was trained. An actual
trained tiny classifier with noncanonical tie/A/B order passed this declared
copy migration at 1e-7 tolerance (maximum difference zero); see
[the receipt](reports/v0.3.0/legacy-migration-verification.json).
