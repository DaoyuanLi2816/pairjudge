"""Load verified judge bundles and score pairs in canonical A/B/tie order."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .packing import PackerConfig, PairPacker
from .validation import CLASSES, class_order, decisions, probabilities, validate_frame

SWAP_PERMUTATION = (1, 0, 2)


def swap_average(proba, proba_swapped):
    """Swap equivariance after exchanging A/B columns; costs two passes."""
    p, q = probabilities(proba), probabilities(proba_swapped)
    if p.shape != q.shape:
        raise ValueError("both passes must have the same shape")
    return ((p + q[:, SWAP_PERMUTATION]) / 2).astype(np.float32)


def resolve_precision(device="auto", dtype="auto"):
    """Select one device and supported precision without silent fallbacks."""
    import torch

    device = "cuda:0" if device == "auto" and torch.cuda.is_available() else device
    device = "cpu" if device == "auto" else device
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("qualified devices are cpu and a single cuda device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    names = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype == "auto" or dtype is None:
        dtype = (
            (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if target.type == "cuda"
            else torch.float32
        )
    elif isinstance(dtype, str):
        if dtype not in names:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        dtype = names[dtype]
    if dtype not in names.values():
        raise ValueError("unsupported dtype")
    if target.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference/training is qualified in float32")
    if (
        target.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("requested CUDA bfloat16 is unsupported")
    return target, dtype


class PairwiseJudge:
    """A learned three-class judge. Public loading requires a trained bundle."""

    def __init__(
        self, model, tokenizer, packer_config=None, label_order=CLASSES, metadata=None
    ):
        self.model, self.tokenizer = model, tokenizer
        self.label_order = class_order(label_order)
        self.metadata = metadata or {}
        self.packer = PairPacker(tokenizer, packer_config, label_mode="none")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        packer_config=None,
        load_in_8bit=False,
        device_map=None,
        torch_dtype=None,
        *,
        device="auto",
        dtype="auto",
        revision=None,
        local_files_only=False,
        cache_dir=None,
        allow_legacy=False,
        label_order=None,
        **model_kwargs,
    ):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install pairjudge[judge] to load a trained model"
            ) from exc

        from ._compat import model_dtype_kwargs
        from .artifacts import read_manifest, resolve_bundle, sha256

        if model_kwargs or load_in_8bit:
            raise ValueError(
                "remote code, custom loading kwargs and quantization are unqualified"
            )
        if device_map is not None:
            if device_map not in ("cpu", "cuda", "cuda:0", "auto"):
                raise ValueError("only a single device is supported")
            device = device_map
        target, precision = resolve_precision(device, torch_dtype or dtype)
        if allow_legacy and (packer_config is None or label_order is None):
            raise ValueError(
                "legacy loading requires explicit packer_config and label_order"
            )
        folder, resolved = resolve_bundle(
            str(model_name_or_path), revision, local_files_only, cache_dir, allow_legacy
        )
        if (folder / "pairjudge.json").exists():
            metadata = read_manifest(folder)
            stored = PackerConfig(**metadata["packer"])
            if packer_config is not None and asdict(packer_config) != asdict(stored):
                raise ValueError("packer override conflicts with the trained artifact")
            if (
                label_order is not None
                and list(class_order(label_order)) != metadata["label_order"]
            ):
                raise ValueError("label_order conflicts with the artifact")
            packer_config, label_order = stored, metadata["label_order"]
        elif allow_legacy:
            metadata = {
                "artifact_type": "full",
                "legacy": True,
                "label_order": list(class_order(label_order)),
            }
        else:
            read_manifest(folder)
        if (
            torch_dtype is None
            and dtype == "auto"
            and metadata.get("inference", {}).get("dtype") == "float32"
        ):
            target, precision = resolve_precision(str(target), "float32")
        tokenizer_options = {}
        if metadata.get("source", {}).get("architecture") == "qwen2":
            # Transformers 4.57.3–4.57.6 misidentifies saved Qwen configs as
            # needing the Mistral regex patch. Preserve Qwen's trained regex.
            tokenizer_options["fix_mistral_regex"] = False
        tokenizer = AutoTokenizer.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, **tokenizer_options
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("saved judge tokenizer must have an explicit pad token")
        loading = dict(
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
            **model_dtype_kwargs(precision),
        )
        if metadata["artifact_type"] == "adapter":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Adapter loading needs pairjudge[train] (PEFT)"
                ) from exc
            from safetensors import safe_open

            source = metadata["source"]
            base = folder / "base" if (folder / "base").is_dir() else source["model_id"]
            loading["local_files_only"] = (
                local_files_only if isinstance(base, str) else True
            )
            if isinstance(base, str):
                loading["revision"] = source["model_revision"]
            model, info = AutoModelForSequenceClassification.from_pretrained(
                base, num_labels=3, **loading
            )
            model.config.pad_token_id = tokenizer.pad_token_id
            for key, value in metadata.get("model_config", {}).items():
                setattr(model.config, key, value)
            if metadata.get("training", {}).get("config", {}).get(
                "disable_attn_logit_softcapping"
            ) and hasattr(model.config, "attn_logit_softcapping"):
                model.config.attn_logit_softcapping = None
            with safe_open(
                folder / "adapter_model.safetensors", framework="pt"
            ) as weights:
                if not any(
                    any(
                        part in key.split(".")
                        for part in ("score", "classifier", "classification_head")
                    )
                    for key in weights.keys()
                ):
                    raise ValueError("adapter is missing its trained classifier head")
            model.config.id2label = dict(enumerate(label_order))
            model.config.label2id = {label: i for i, label in enumerate(label_order)}
            model = PeftModel.from_pretrained(model, folder, is_trainable=False)
        else:
            model, info = AutoModelForSequenceClassification.from_pretrained(
                folder, **loading
            )
            if (
                info["missing_keys"]
                or info["mismatched_keys"]
                or info.get("error_msgs")
            ):
                raise ValueError(f"incomplete trained weights: {info}")
        if model.config.num_labels != 3:
            raise ValueError("judge must have exactly three trained classes")
        if model.config.pad_token_id != tokenizer.pad_token_id:
            raise ValueError("model and saved tokenizer padding token differ")
        if [model.config.id2label.get(i) for i in range(3)] != list(label_order):
            raise ValueError("model config class mapping differs from label_order")
        if model.config.label2id != {label: i for i, label in enumerate(label_order)}:
            raise ValueError("model config label2id differs from label_order")
        if not metadata.get("legacy"):
            for key, value in metadata["tokenizer"].items():
                if getattr(tokenizer, key) != value:
                    raise ValueError(f"saved tokenizer metadata mismatch: {key}")
        model.config.use_cache = False
        model.to(target).eval()
        metadata = dict(
            metadata,
            public_model_id=str(model_name_or_path) if resolved else "local_bundle",
            resolved_revision=resolved,
            artifact_sha256=sha256(folder / "pairjudge.json")
            if (folder / "pairjudge.json").exists()
            else None,
            device=str(target),
            dtype=str(precision).removeprefix("torch."),
        )
        return cls(model, tokenizer, packer_config, label_order, metadata)

    def save_pretrained(self, destination):
        """Re-save a verified full bundle, preserving its inference contract."""
        from .artifacts import save_bundle

        if self.metadata.get("artifact_type") != "full":
            raise ValueError("re-saving requires a verified full artifact")
        return save_bundle(
            self.model,
            self.tokenizer,
            destination,
            self.packer.config,
            self.label_order,
            self.metadata["training"],
            self.metadata.get("source"),
        )

    def _pack_frame(self, df):
        validate_frame(df)
        packed = [
            self.packer.pack(p, a, b)
            for p, a, b in zip(df.prompt, df.response_a, df.response_b)
        ]
        for position, item in enumerate(packed):
            if not item.usable:
                raise ValueError(
                    f"row {position}: insufficient retained response content; {item.warnings}"
                )
        return pd.DataFrame(
            {
                "_order": np.arange(len(packed)),
                "input_ids": [x.input_ids for x in packed],
                "attention_mask": [x.attention_mask for x in packed],
                "length": [len(x.input_ids) for x in packed],
                "diagnostics": [x.diagnostics() for x in packed],
            }
        )

    def _forward(self, frame, batch_size):
        import torch

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if frame.empty:
            return np.empty((0, 3), dtype=np.float32)
        frame = frame.sort_values("length", ascending=False)
        probs = []
        device = next(self.model.parameters()).device
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start : start + batch_size]
                inputs = self.tokenizer.pad(
                    {
                        "input_ids": batch.input_ids.tolist(),
                        "attention_mask": batch.attention_mask.tolist(),
                    },
                    padding="longest",
                    return_tensors="pt",
                )
                logits = self.model(**inputs.to(device)).logits
                probs.append(logits.float().softmax(-1).cpu().numpy())
        out = np.concatenate(probs)
        out = out[np.argsort(frame._order.to_numpy())]
        probabilities(out, "model outputs")
        return out[:, [self.label_order.index(label) for label in CLASSES]]

    def predict_proba(self, df, swap_debias=False, batch_size=4):
        """Canonical probabilities; averaging offers equivariance, not an accuracy guarantee."""
        proba = self._forward(self._pack_frame(df), batch_size)
        return (
            swap_average(proba, self._swapped_proba(df, batch_size))
            if swap_debias
            else proba
        )

    def predict(self, df, swap_debias=False, batch_size=4):
        """Return stable IDs, probabilities, verdicts and content retention diagnostics."""
        frame = self._pack_frame(df)
        proba = self._forward(frame, batch_size)
        diagnostics = frame.diagnostics.tolist()
        if swap_debias:
            swapped = self._pack_frame(self._swap(df))
            proba = swap_average(proba, self._forward(swapped, batch_size))
            diagnostics = [
                {"original": a, "swapped": b}
                for a, b in zip(diagnostics, swapped.diagnostics)
            ]
        return [
            {
                "id": str(df.iloc[i].get("id", i)),
                "probabilities": dict(zip(CLASSES, map(float, row))),
                "verdict": verdict,
                "packing": diagnostics[i],
            }
            for i, (row, verdict) in enumerate(zip(proba, decisions(proba)))
        ]

    @staticmethod
    def _swap(df):
        swapped = df.copy()
        swapped["response_a"], swapped["response_b"] = (
            df["response_b"],
            df["response_a"],
        )
        return swapped

    def _swapped_proba(self, df, batch_size):
        return self._forward(self._pack_frame(self._swap(df)), batch_size)

    def position_flip_rate(self, df, batch_size=4):
        """Aligned verdict disagreement; numerical ties count as 'ambiguous'."""
        if df.empty:
            raise ValueError("position_flip_rate requires at least one row")
        p = self._forward(self._pack_frame(df), batch_size)
        q = self._swapped_proba(df, batch_size)[:, SWAP_PERMUTATION]
        return float(np.mean(np.array(decisions(p)) != np.array(decisions(q))))
