"""Versioned, data-only model bundles with staged safetensors saves."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .packing import PackerConfig
from .validation import class_order

MANIFEST = "pairjudge.json"
FORMAT_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(folder):
    folder = Path(folder)
    path = folder / MANIFEST
    if not path.exists():
        raise ValueError(
            "missing pairjudge.json: use a trained PairJudge artifact; legacy loading requires explicit label_order and packer_config"
        )
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("artifact manifest exceeds 1 MiB")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(
        name not in metadata
        for name in ("label_order", "packer", "tokenizer", "training", "files")
    ):
        raise ValueError("artifact manifest is missing required metadata")
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported PairJudge artifact format")
    if metadata.get("artifact_type") not in ("full", "adapter"):
        raise ValueError("unsupported artifact_type")
    class_order(metadata["label_order"])
    PackerConfig(**metadata["packer"])
    training = metadata.get("training", {})
    if (
        training.get("optimizer_steps", 0) <= 0
        or training.get("head_changed") is not True
    ):
        raise ValueError(
            "artifact does not identify an updated, trained classification head"
        )
    files = metadata.get("files", {})
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("artifact requires safetensors weights and file hashes")
    for name, expected in files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or ":" in name
            or "\\" in name
        ):
            raise ValueError("unsafe artifact file path")
        target = folder / relative
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"artifact file hash mismatch: {name}")
    return metadata


def resolve_bundle(
    model_id, revision=None, local_files_only=False, cache_dir=None, allow_legacy=False
):
    local = Path(model_id)
    if local.is_dir():
        return local.resolve(), None
    if "://" in model_id or model_id.startswith(("/", "\\")):
        raise ValueError(
            "expected a local model directory or Hugging Face repository ID, not a URL"
        )
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        path = Path(
            hf_hub_download(
                model_id,
                MANIFEST,
                revision=revision,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
        )
        resolved = path.parent.name
    except LocalEntryNotFoundError as exc:
        raise ValueError(
            "required artifact files are absent from the offline cache; download the pinned judge first"
        ) from exc
    except EntryNotFoundError as exc:
        if not allow_legacy:
            raise ValueError(
                "repository lacks a PairJudge artifact manifest; a backbone is not a trained judge"
            ) from exc
        config = Path(
            hf_hub_download(
                model_id,
                "config.json",
                revision=revision,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
        )
        resolved = config.parent.name
    folder = snapshot_download(
        model_id,
        revision=resolved,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.jinja",
            "README.md",
            "LICENSE",
            "NOTICE",
            "base/*.json",
            "base/*.safetensors",
            "base/*.txt",
            "base/*.model",
            "base/*.jinja",
        ],
    )
    return Path(folder), resolved


def head_parameters(model):
    names = ("score", "classifier", "classification_head")
    return {
        name: tensor
        for name, tensor in model.named_parameters()
        if any(part in names for part in name.split(".")) and tensor.requires_grad
    }


def tensor_fingerprint(tensor):
    return hashlib.sha256(
        tensor.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def save_bundle(
    model,
    tokenizer,
    destination,
    packer_config,
    label_order,
    training,
    source=None,
    artifact_type="full",
    embedded_base=None,
):
    from . import __version__

    if training.get("optimizer_steps", 0) <= 0 or not training.get("head_changed"):
        raise ValueError(
            "save requires actual optimizer updates and a changed classification head"
        )
    order = class_order(label_order)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite model directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        model.config.id2label = dict(enumerate(order))
        model.config.label2id = {label: index for index, label in enumerate(order)}
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        if embedded_base:
            base = Path(embedded_base).resolve()
            (stage / "base").mkdir()
            for path in base.iterdir():
                if path.is_file() and (
                    path.suffix in (".json", ".safetensors", ".txt", ".model", ".jinja")
                    or path.name in ("LICENSE", "NOTICE")
                ):
                    shutil.copy2(path, stage / "base" / path.name)
        files = {
            path.relative_to(stage).as_posix(): sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        metadata = {
            "format_version": FORMAT_VERSION,
            "library_version": __version__,
            "author": "Daoyuan Li",
            "artifact_type": artifact_type,
            "packer": asdict(packer_config),
            "label_order": list(order),
            "tokenizer": {
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "padding_side": tokenizer.padding_side,
            },
            "model_config": {
                name: getattr(model.config, name)
                for name in ("pad_token_id", "use_cache", "attn_logit_softcapping")
                if hasattr(model.config, name)
            },
            "inference": {
                "dtype": "float32"
                if str(next(model.parameters()).dtype) == "torch.float32"
                else "auto",
                "heuristics": "none",
                "calibration": None,
            },
            "training": training,
            "source": source or {},
            "dependencies": {},
            "files": files,
        }
        for name in ("torch", "transformers", "tokenizers", "peft", "huggingface-hub"):
            try:
                metadata["dependencies"][name] = version(name)
            except PackageNotFoundError:
                pass
        (stage / MANIFEST).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        read_manifest(stage)
        os.rename(stage, destination)
    finally:
        if stage.exists():
            if stage.resolve().parent != destination.parent:
                raise RuntimeError("unsafe save-stage cleanup path")
            shutil.rmtree(stage)
    return metadata
