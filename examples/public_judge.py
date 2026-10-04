"""Reproduce the public judge's frozen splits and auditable evaluation.

Run from an installed pairjudge[train] environment. Text stays in .cache;
only IDs, controlled hashes, labels, diagnostics and predictions are published.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pairjudge.artifacts import sha256
from pairjudge.splits import grouped_ids, row_keys
from pairjudge.validation import WINNER_COLUMNS, decisions, validate_frame

DATASET = "lmarena-ai/arena-human-preference-55k"
DATA_REVISION = "18c298340948c0e7f7727399fd459cca6ce0ca6f"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def freeze(root, report):
    from huggingface_hub import hf_hub_download

    root, report = Path(root), Path(report)
    if (report / "split-manifest.json").exists():
        raise FileExistsError(
            "frozen split already exists; reconstruct from its IDs, do not resample"
        )
    path = hf_hub_download(
        DATASET, "train.csv", repo_type="dataset", revision=DATA_REVISION
    )
    raw = pd.read_csv(path)
    valid, excluded, seen = [], [], set()
    for _, row in raw.iterrows():
        row = row.to_dict()
        identifier = str(row["id"])
        try:
            for name in ("prompt", "response_a", "response_b"):
                row[name] = json.loads(row[name])
            validate_frame(pd.DataFrame([row]), "hard")
            if not row["prompt"] or not any(text.strip() for text in row["prompt"]):
                raise ValueError("no prompt context")
            if not any(text.strip() for text in row["response_a"]) or not any(
                text.strip() for text in row["response_b"]
            ):
                raise ValueError("one or both answers blank")
            keys = row_keys(row)
            if row["response_a"] == row["response_b"]:
                raise ValueError("identical responses")
            if keys[2] in seen:
                raise ValueError("duplicate normalized conversation or reversed pair")
            seen.add(keys[2])
            row.update(
                id=identifier,
                prompt_hash=keys[0],
                pair_hash=keys[1],
                content_hash=keys[2],
            )
            valid.append(row)
        except (ValueError, TypeError) as exc:
            excluded.append({"id": identifier, "reason": str(exc)})
    frame = pd.DataFrame(valid)
    frame["group"] = grouped_ids(frame)
    groups = sorted(frame.group.unique())
    rng = np.random.default_rng(2816)
    rng.shuffle(groups)
    targets = {"test": 1000, "validation": 768, "train": 8000}
    sizes = frame.groupby("group").size().to_dict()
    assignment, position = {}, 0
    for split, target in targets.items():
        count = 0
        while count < target:
            group = groups[position]
            assignment[group] = split
            count += sizes[group]
            position += 1
    frame["split"] = frame.group.map(assignment).fillna("unused")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for split in targets:
        part = frame[frame.split == split].sort_values("id").reset_index(drop=True)
        part.to_json(
            root / f"{split}.jsonl", orient="records", lines=True, force_ascii=False
        )
        for _, row in part.iterrows():
            records.append(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "split",
                        "group",
                        "prompt_hash",
                        "pair_hash",
                        "content_hash",
                    )
                }
            )
    protocol = {
        "dataset": DATASET,
        "dataset_revision": DATA_REVISION,
        "dataset_license": "apache-2.0",
        "dataset_file_sha256": sha256(path),
        "backbone": BASE,
        "backbone_revision": BASE_REVISION,
        "seed": 2816,
        "targets": targets,
        "seeds": [42, 43],
        "selection": "lowest validation single-pass log-loss, before final test",
        "budget": "two one-epoch 8000-row LoRA runs at length512, no test selection; pilot <=8 steps",
        "normalization": "NFKC, collapse whitespace, strip, casefold; ordered prompt rounds and unordered response lists",
        "grouping": "connected components sharing a normalized prompt or unordered response pair; reversed duplicates removed",
        "raw_rows": len(raw),
        "valid_unique_rows": len(frame),
        "independent_groups": len(groups),
        "exclusions": excluded,
        "rows": sorted(records, key=lambda row: (row["split"], row["id"])),
        "splits": {
            split: {
                "rows": int((frame.split == split).sum()),
                "groups": int(frame[frame.split == split].group.nunique()),
            }
            for split in targets
        },
        "prior_from_train": frame[frame.split == "train"][list(WINNER_COLUMNS)]
        .mean()
        .tolist(),
        "limitations": "This is a newly frozen split for this delivery, not an external benchmark. Arena is public; prior historical experiments/pretraining overlap cannot be ruled out.",
    }
    for a in targets:
        for b in targets:
            if a >= b:
                continue
            for column in ("group", "prompt_hash", "pair_hash", "content_hash"):
                assert not set(frame[frame.split == a][column]) & set(
                    frame[frame.split == b][column]
                )
    protocol["overlap_audit"] = (
        "zero group/prompt/pair/conversation hash overlap across splits"
    )
    write_json(report / "split-manifest.json", protocol)
    print(
        json.dumps(
            {
                "splits": protocol["splits"],
                "excluded": len(excluded),
                "valid": len(frame),
            }
        )
    )


def metrics(labels, probs):
    from sklearn.metrics import classification_report, f1_score

    labels, probs = np.asarray(labels), np.asarray(probs)
    prediction = probs.argmax(1)
    return {
        "rows": len(labels),
        "log_loss": float(
            -np.log(probs[np.arange(len(labels)), labels].clip(1e-15)).mean()
        ),
        "accuracy": float((prediction == labels).mean()),
        "macro_f1": float(
            f1_score(
                labels, prediction, labels=[0, 1, 2], average="macro", zero_division=0
            )
        ),
        "brier_sum_three_classes": float(
            ((probs - np.eye(3)[labels]) ** 2).sum(1).mean()
        ),
        "per_class": classification_report(
            labels,
            prediction,
            labels=[0, 1, 2],
            target_names=["a_wins", "b_wins", "tie"],
            output_dict=True,
            zero_division=0,
        ),
        "prediction_counts": np.bincount(prediction, minlength=3).tolist(),
    }


def paired_intervals(labels, single, averaged, groups):
    loss_delta = -np.log(averaged[np.arange(len(labels)), labels].clip(1e-15)) + np.log(
        single[np.arange(len(labels)), labels].clip(1e-15)
    )
    acc_delta = (averaged.argmax(1) == labels).astype(float) - (
        single.argmax(1) == labels
    ).astype(float)
    grouped = (
        pd.DataFrame({"group": groups, "loss": loss_delta, "accuracy": acc_delta})
        .groupby("group")
        .agg(["sum", "count"])
    )
    sums = grouped[[("loss", "sum"), ("accuracy", "sum")]].to_numpy()
    counts = grouped[("loss", "count")].to_numpy()
    rng, bootstrap = np.random.default_rng(17), []
    for _ in range(2000):
        sampled = rng.integers(0, len(grouped), len(grouped))
        bootstrap.append(sums[sampled].sum(0) / counts[sampled].sum())
    bounds = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return {
        "method": "2000 paired connected-group bootstrap replicates, percentile 95% intervals",
        "groups": len(grouped),
        "log_loss_delta": {
            "estimate": float(loss_delta.mean()),
            "ci95": bounds[:, 0].tolist(),
        },
        "accuracy_delta": {
            "estimate": float(acc_delta.mean()),
            "ci95": bounds[:, 1].tolist(),
        },
    }


def evaluate(model_path, root, report, device):
    import torch

    from pairjudge import PairwiseJudge
    from pairjudge.judge import swap_average

    root, report = Path(root), Path(report)
    if (report / "evaluation.json").exists():
        raise FileExistsError(
            "evaluation already exists; do not overwrite frozen predictions"
        )
    frame = pd.read_json(root / "test.jsonl", lines=True, dtype={"id": str})
    protocol = json.loads(
        (report.parent / "split-manifest.json").read_text(encoding="utf-8")
    )
    labels = frame[list(WINNER_COLUMNS)].to_numpy().argmax(1)
    started = time.perf_counter()
    judge = PairwiseJudge.from_pretrained(model_path, device=device)
    load_s = time.perf_counter() - started
    packed = judge._pack_frame(frame)
    swapped = judge._pack_frame(judge._swap(frame))

    def sync():
        if next(judge.model.parameters()).device.type == "cuda":
            torch.cuda.synchronize()

    def timed(function, size=1, repetitions=3):
        function()  # warmup excluded
        timings, peaks = [], []
        for _ in range(repetitions):
            sync()
            if next(judge.model.parameters()).device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            function()
            sync()
            timings.append(time.perf_counter() - t)
            peaks.append(
                torch.cuda.max_memory_allocated()
                if next(judge.model.parameters()).device.type == "cuda"
                else None
            )
        return {
            "repetitions": repetitions,
            "rows": size,
            "seconds": timings,
            "median_seconds": float(np.median(timings)),
            "pairs_per_second": size / float(np.median(timings)),
            "peak_cuda_allocated_bytes": peaks,
            "includes": "CPU packing/padding and GPU forward, synchronized; excludes loading and warmup",
        }

    sync()
    t = time.perf_counter()
    single = judge._forward(packed, 4)
    reverse = judge._forward(swapped, 4)
    sync()
    score_s = time.perf_counter() - t
    averaged = swap_average(single, reverse)
    backward = swap_average(reverse, single)
    equivalence = float(np.max(np.abs(backward[:, [1, 0, 2]] - averaged)))
    aligned = reverse[:, [1, 0, 2]]
    n = len(frame)
    modes = {
        "uniform": np.tile([1 / 3] * 3, (n, 1)),
        "train_prior": np.tile(protocol["prior_from_train"], (n, 1)),
        "single": single,
        "swap_average": averaged,
    }
    aggregate = {key: metrics(labels, values) for key, values in modes.items()}
    costs = {}

    def gpu_forward_intervals(sample):
        if next(judge.model.parameters()).device.type != "cuda":
            return {"measured": False, "reason": "CUDA events unavailable on CPU"}
        timings = []
        with torch.inference_mode():
            for _ in range(3):
                milliseconds = 0.0
                for start in range(0, len(sample), 4):
                    batch = sample.iloc[start : start + 4]
                    inputs = judge.tokenizer.pad(
                        {
                            "input_ids": batch.input_ids.tolist(),
                            "attention_mask": batch.attention_mask.tolist(),
                        },
                        padding=True,
                        return_tensors="pt",
                    ).to(next(judge.model.parameters()).device)
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    judge.model(**inputs)
                    end.record()
                    end.synchronize()
                    milliseconds += begin.elapsed_time(end)
                timings.append(milliseconds)
        return {
            "measured": True,
            "milliseconds": timings,
            "includes": "GPU-event forward intervals after padding/device transfer, three repetitions; shared GPU contention may affect intervals",
        }

    for mode in (False, True):
        name = "swap_average" if mode else "single"
        costs[name] = {
            "one_pair": timed(
                lambda: judge.predict_proba(
                    frame.iloc[:1], swap_debias=mode, batch_size=1
                )
            ),
            "batch64": timed(
                lambda: judge.predict_proba(
                    frame.iloc[:64], swap_debias=mode, batch_size=4
                ),
                64,
            ),
            "gpu_forward_intervals_batch64": gpu_forward_intervals(packed.iloc[:64]),
        }
        if mode:
            reverse_cost = gpu_forward_intervals(swapped.iloc[:64])
            costs[name]["gpu_forward_intervals_batch64"]["milliseconds"] = [
                a + b
                for a, b in zip(
                    costs[name]["gpu_forward_intervals_batch64"].get(
                        "milliseconds", []
                    ),
                    reverse_cost.get("milliseconds", []),
                )
            ]
    per_row = []
    for i, (_, row) in enumerate(frame.iterrows()):
        per_row.append(
            {
                "id": str(row.id),
                "group": row.group,
                "label": int(labels[i]),
                "single": single[i].tolist(),
                "reverse": reverse[i].tolist(),
                "swap_average": averaged[i].tolist(),
                "packing": packed.iloc[i].diagnostics,
                "swapped_packing": swapped.iloc[i].diagnostics,
            }
        )
    report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_row:
            handle.write(json.dumps(row) + "\n")
    slices = {}
    for name, mask in {
        "truncated": np.array([row["packing"]["truncated"] for row in per_row]),
        "untruncated": np.array([not row["packing"]["truncated"] for row in per_row]),
        "multi_turn": frame.prompt.map(len).to_numpy() > 1,
        **{f"label_{i}": labels == i for i in range(3)},
        "packed_under256": packed.length.to_numpy() < 256,
        "packed_256plus": packed.length.to_numpy() >= 256,
    }.items():
        if mask.any():
            slices[name] = {
                mode: metrics(labels[mask], proba[mask])
                for mode, proba in modes.items()
            }
    failure_indices = np.argsort(-np.log(averaged[np.arange(n), labels].clip(1e-15)))[
        -5:
    ][::-1]
    failures = [
        {
            "id": str(frame.iloc[i].id),
            "human_label": int(labels[i]),
            "predicted": decisions(averaged[i : i + 1])[0],
            "probabilities": averaged[i].tolist(),
            "truncated": per_row[i]["packing"]["truncated"],
            "note": "high-loss mismatch; retrieve authorized source by ID to inspect, no conversation text uploaded",
        }
        for i in failure_indices
    ]
    outcome = {
        "metrics": aggregate,
        "paired_swap_minus_single": paired_intervals(
            labels, single, averaged, frame.group
        ),
        "flip_rate_single": float(
            np.mean(np.array(decisions(single)) != np.array(decisions(aligned)))
        ),
        "mean_aligned_absolute_probability_difference": float(
            np.abs(single - aligned).mean()
        ),
        "swap_equivariance_max_abs": equivalence,
        "slices": slices,
        "failures": failures,
        "cost": costs,
        "loading_seconds": load_s,
        "two_test_passes_seconds": score_s,
        "hardware": {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "os": platform.platform(),
            "torch": torch.__version__,
            "device": judge.metadata["device"],
            "dtype": judge.metadata["dtype"],
            "batch_size": 4,
        },
        "training": judge.metadata["training"],
        "artifact_files": judge.metadata["files"],
        "split_manifest_sha256": sha256(report.parent / "split-manifest.json"),
        "prediction_file_sha256": sha256(report / "predictions.jsonl"),
        "quality_gate": {
            "beats_uniform_and_train_prior_log_loss": aggregate["single"]["log_loss"]
            < min(
                aggregate["uniform"]["log_loss"], aggregate["train_prior"]["log_loss"]
            ),
            "nonconstant_predictions": len(set(single.argmax(1))) > 1,
            "finite_three_class_output": bool(
                np.isfinite(single).all() and single.shape == (n, 3)
            ),
        },
        "limitations": "Uncalibrated; one dataset; no external contamination guarantee. Bootstrap measures test-sampling uncertainty, not training variability. Costs measured while sharing GPU with existing work.",
    }
    write_json(report / "evaluation.json", outcome)
    print(
        json.dumps(
            {
                "metrics": aggregate,
                "paired": outcome["paired_swap_minus_single"],
                "gate": outcome["quality_gate"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--root", default=".cache/public-judge")
    parser.add_argument("--report", default="docs/reports/v0.3.0")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root, args.report)
    else:
        evaluate(args.model, args.root, args.report, args.device)


if __name__ == "__main__":
    main()
