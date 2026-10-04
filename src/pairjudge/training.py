"""Single-device fine-tuning with explicit sample-normalized CE/KL accumulation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import load_arena_csv
from .packing import PackerConfig, PairPacker
from .validation import (
    CLASSES,
    WINNER_COLUMNS,
    class_order,
    probabilities,
    validate_frame,
)


@dataclass
class JudgeTrainConfig:
    train_path: str = ""
    validation_path: str = ""
    label_mode: str = "hard"
    eval_holdout: float = 0.05
    seed: int = 42
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = ""
    local_files_only: bool = False
    max_length: int = 2048
    packer: dict = field(default_factory=dict)
    label_order: list = field(default_factory=lambda: list(CLASSES))
    disable_attn_logit_softcapping: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    output_dir: str = "./output/judge"
    n_epochs: float = 1.0
    max_steps: int = 0
    lr: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    warmup_steps: int = 20
    optim: str = "adamw_torch"
    device: str = "auto"
    dtype: str = "auto"
    bf16: bool = False  # explicit compatibility alias; False does not imply fp16
    num_workers: int = 0
    merge_adapter: bool = True
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.label_mode not in ("hard", "soft"):
            raise ValueError("label_mode must be hard or soft")
        if not math.isfinite(self.eval_holdout) or not 0 < self.eval_holdout < 1:
            raise ValueError("eval_holdout must be between 0 and 1")
        for name in (
            "max_length",
            "n_epochs",
            "lr",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "max_length",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "warmup_steps",
            "max_steps",
            "num_workers",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.optim != "adamw_torch":
            raise ValueError("qualified optimizer is adamw_torch")
        if self.num_workers:
            raise ValueError(
                "num_workers must be zero for the qualified deterministic loader"
            )
        if self.use_lora:
            if self.lora_r <= 0 or self.lora_alpha <= 0:
                raise ValueError("lora_r and lora_alpha must be positive")
            if not 0 <= self.lora_dropout < 1:
                raise ValueError("lora_dropout must be in [0, 1)")
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules must not be empty")
        class_order(self.label_order)
        self.packing_config()

    def packing_config(self):
        if "max_length" in self.packer and self.packer["max_length"] != self.max_length:
            raise ValueError("packer.max_length conflicts with max_length")
        return PackerConfig(**dict(self.packer, max_length=self.max_length))


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    unknown = set(raw) - set(JudgeTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return JudgeTrainConfig(**raw)


def summed_loss(logits, labels, label_mode):
    """Sum over examples; caller divides by the actual accumulation-group size."""
    import torch.nn.functional as F

    logits = logits.float()
    if label_mode == "hard":
        return F.cross_entropy(logits, labels.long(), reduction="sum")
    return F.kl_div(F.log_softmax(logits, dim=-1), labels.float(), reduction="sum")


def soft_kl_loss(logits, soft_labels):
    probabilities(soft_labels.detach().cpu().numpy(), "soft labels")
    return summed_loss(logits, soft_labels, "soft") / len(soft_labels)


def compute_metrics(eval_preds):
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    exp = np.exp(preds - preds.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    targets = labels if labels.ndim == 2 else np.eye(3)[labels]
    return {
        "log_loss": float(-(targets * np.log(np.clip(probs, 1e-15, 1))).sum(1).mean()),
        "acc": float((probs.argmax(1) == targets.argmax(1)).mean()),
    }


def _load_dataframe(path):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    if str(path).endswith(".jsonl"):
        return pd.read_json(path, lines=True, dtype={"id": str})
    return load_arena_csv(path)


def build_model_and_tokenizer(cfg):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ._compat import model_dtype_kwargs
    from .judge import resolve_precision

    target, dtype = resolve_precision(cfg.device, "bfloat16" if cfg.bf16 else cfg.dtype)
    if not Path(cfg.model_name).is_dir() and not re.fullmatch(
        r"[0-9a-f]{40}", cfg.model_revision
    ):
        raise ValueError(
            "remote training backbones require an exact 40-character commit model_revision"
        )
    kwargs = dict(
        revision=cfg.model_revision or None,
        local_files_only=cfg.local_files_only,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs an explicit pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    order = class_order(cfg.label_order)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
        id2label=dict(enumerate(order)),
        label2id={label: i for i, label in enumerate(order)},
        use_safetensors=True,
        **model_dtype_kwargs(dtype),
        **kwargs,
    )
    model.config.pad_token_id, model.config.use_cache = tokenizer.pad_token_id, False
    if cfg.disable_attn_logit_softcapping and hasattr(
        model.config, "attn_logit_softcapping"
    ):
        model.config.attn_logit_softcapping = None
    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        head = [
            name
            for name in ("score", "classifier", "classification_head")
            if hasattr(model, name)
        ]
        if not head:
            raise ValueError("no qualified classification head found")
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                target_modules=cfg.lora_target_modules,
                modules_to_save=head,
            ),
        )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.to(target)
    return model, tokenizer


def _packed(frame, packer, cfg):
    labels = frame[list(WINNER_COLUMNS)].to_numpy()[
        :, [CLASSES.index(label) for label in cfg.label_order]
    ]
    result = []
    for position, (_, row) in enumerate(frame.iterrows()):
        item = packer.pack(row.prompt, row.response_a, row.response_b)
        if not item.usable:
            raise ValueError(
                f"row {position}: unusable packed example; {item.warnings}"
            )
        result.append(
            {
                "input_ids": item.input_ids,
                "attention_mask": item.attention_mask,
                "labels": int(labels[position].argmax())
                if cfg.label_mode == "hard"
                else labels[position].tolist(),
            }
        )
    return result


def _batch(records, tokenizer, device):
    import torch

    inputs = tokenizer.pad(
        {
            name: [record[name] for record in records]
            for name in ("input_ids", "attention_mask")
        },
        padding=True,
        return_tensors="pt",
    ).to(device)
    labels = torch.tensor([record["labels"] for record in records], device=device)
    return inputs, labels


def _evaluate(model, records, tokenizer, cfg):
    import torch

    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for start in range(0, len(records), cfg.per_device_eval_batch_size):
            inputs, targets = _batch(
                records[start : start + cfg.per_device_eval_batch_size],
                tokenizer,
                next(model.parameters()).device,
            )
            logits.extend(model(**inputs).logits.float().cpu().numpy())
            labels.extend(targets.cpu().numpy())

    class Output:
        predictions, label_ids = np.array(logits), np.array(labels)

    return compute_metrics(Output())


def train(cfg):
    """Train, select on validation, then atomically save adapter/full bundles."""
    import torch
    from transformers import set_seed

    from .artifacts import head_parameters, save_bundle, sha256, tensor_fingerprint
    from .splits import assert_disjoint, grouped_holdout

    destination = Path(cfg.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite training output: {destination}")
    train_df = _load_dataframe(cfg.train_path)
    validate_frame(train_df, cfg.label_mode)
    if cfg.validation_path:
        val_df = _load_dataframe(cfg.validation_path)
        validate_frame(val_df, cfg.label_mode)
        assert_disjoint(train_df, val_df)
    else:
        train_df, val_df = grouped_holdout(train_df, cfg.eval_holdout, cfg.seed)
    if train_df.empty or val_df.empty:
        raise ValueError("training and validation must both be non-empty")
    set_seed(cfg.seed)
    model, tokenizer = build_model_and_tokenizer(cfg)
    packer = PairPacker(tokenizer, cfg.packing_config())
    training, validation = _packed(train_df, packer, cfg), _packed(val_df, packer, cfg)
    head_before = {
        name: tensor_fingerprint(tensor)
        for name, tensor in head_parameters(model).items()
    }
    if not head_before:
        raise ValueError("classification head is not trainable")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )
    micro, accum = cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps
    epoch_steps = math.ceil(len(training) / (micro * accum))
    total_steps = min(
        cfg.max_steps or math.ceil(epoch_steps * cfg.n_epochs),
        math.ceil(epoch_steps * cfg.n_epochs),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / max(1, cfg.warmup_steps))
            * 0.5
            * (
                1
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        max(0, step - cfg.warmup_steps)
                        / max(1, total_steps - cfg.warmup_steps),
                    )
                )
            )
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
        and next(model.parameters()).dtype == torch.float16,
    )
    rng = np.random.default_rng(cfg.seed)
    started, step, history, best, best_weights = (
        time.perf_counter(),
        0,
        [],
        math.inf,
        None,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < total_steps:
        indices = rng.permutation(len(training)).tolist()
        model.train()
        for start in range(0, len(indices), micro * accum):
            if step >= total_steps:
                break
            group = indices[start : start + micro * accum]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for offset in range(0, len(group), micro):
                records = [training[i] for i in group[offset : offset + micro]]
                inputs, labels = _batch(records, tokenizer, device)
                loss = summed_loss(
                    model(**inputs).logits, labels, cfg.label_mode
                ) / len(group)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss; no artifact saved")
                scaler.scale(loss).backward()
                group_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous:
                raise RuntimeError("FP16 optimizer update overflow; no artifact saved")
            scheduler.step()
            step += 1
            if step == 1 or step % 25 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "steps_total": total_steps,
                            "loss": group_loss,
                            "elapsed_s": round(time.perf_counter() - started, 2),
                        }
                    ),
                    flush=True,
                )
        metrics = _evaluate(model, validation, tokenizer, cfg)
        history.append(dict(step=step, **metrics))
        if metrics["log_loss"] < best:
            best = metrics["log_loss"]
            best_weights = {
                name: p.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
        print(json.dumps({"validation": history[-1]}), flush=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in best_weights:
                p.copy_(best_weights[name].to(p.device))
    model.eval()
    head_after = {
        name: tensor_fingerprint(tensor)
        for name, tensor in head_parameters(model).items()
    }
    receipt = {
        "optimizer_steps": step,
        "head_changed": head_before != head_after,
        "head_before": head_before,
        "head_after": head_after,
        "seed": cfg.seed,
        "label_mode": cfg.label_mode,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "history": history,
        "elapsed_s": time.perf_counter() - started,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "config": {
            key: value
            for key, value in asdict(cfg).items()
            if key not in ("train_path", "validation_path", "model_name", "output_dir")
        },
        "train_file_sha256": sha256(cfg.train_path),
        "validation_file_sha256": sha256(cfg.validation_path)
        if cfg.validation_path
        else None,
    }
    source = dict(
        cfg.provenance,
        train_data_provenance=train_df.attrs,
        model_id=cfg.model_name
        if not Path(cfg.model_name).is_dir()
        else "embedded-local-base",
        model_revision=cfg.model_revision or "local",
        architecture=model.config.model_type,
    )
    if Path(cfg.model_name).is_dir():
        from .splits import digest

        hashes = {
            path.name: sha256(path)
            for path in sorted(Path(cfg.model_name).iterdir())
            if path.is_file() and path.suffix in (".safetensors", ".json")
        }
        source["local_base_files"] = hashes
        source["model_revision"] = digest(hashes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        if cfg.use_lora:
            save_bundle(
                model,
                tokenizer,
                stage / "adapter",
                cfg.packing_config(),
                cfg.label_order,
                receipt,
                source,
                "adapter",
                cfg.model_name if Path(cfg.model_name).is_dir() else None,
            )
            if cfg.merge_adapter:
                # Merging into BF16 base weights can materially change scores.
                # Export in FP32 on CPU and bind FP32 default inference.
                model = model.to("cpu", dtype=torch.float32)
                model = model.merge_and_unload()
                save_bundle(
                    model,
                    tokenizer,
                    stage / "merged",
                    cfg.packing_config(),
                    cfg.label_order,
                    receipt,
                    source,
                )
        else:
            save_bundle(
                model,
                tokenizer,
                stage / "merged",
                cfg.packing_config(),
                cfg.label_order,
                receipt,
                source,
            )
        (stage / "training.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        os.rename(stage, destination)
    except Exception:
        if stage.resolve().parent != destination.parent:
            raise RuntimeError("unsafe failed-save path")
        os.rename(
            stage,
            destination.with_name(destination.name + ".failed-" + uuid.uuid4().hex),
        )
        raise
    return dict(
        min(history, key=lambda item: item["log_loss"]),
        output_dir=str(destination),
        optimizer_steps=step,
        head_changed=receipt["head_changed"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    print(json.dumps(train(load_config(parser.parse_args(argv).cfg)), indent=2))


if __name__ == "__main__":
    main()
