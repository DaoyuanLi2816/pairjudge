"""Compare a pair, stream ordered JSONL results, or open the local demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .catalog import DEFAULT_MODEL, DEFAULT_REVISION
from .validation import PAIR_COLUMNS, validate_frame


def input_record(record):
    if not isinstance(record, dict):
        raise ValueError("each input must be a JSON object")
    out = {"id": str(record.get("id", "0"))}
    for name in PAIR_COLUMNS:
        value = record.get(name)
        out[name] = [value] if isinstance(value, str) else value
    validate_frame(pd.DataFrame([out]))
    if sum(len(text) for name in PAIR_COLUMNS for text in out[name]) > 60000:
        raise ValueError("input exceeds the 60000-character per-pair limit")
    if len(out["prompt"]) > 64:
        raise ValueError("input exceeds 64 rounds")
    return out


def run_metadata(judge, swap):
    return {
        "model": judge.metadata.get(
            "public_model_id", judge.metadata.get("source", {}).get("model_id")
        ),
        "revision": judge.metadata.get("resolved_revision"),
        "artifact_sha256": judge.metadata.get("artifact_sha256"),
        "mode": "swap_average" if swap else "single",
        "class_order": ["a_wins", "b_wins", "tie"],
        "packing_format": judge.packer.config.packing_format,
        "max_length": judge.packer.config.max_length,
        "device": judge.metadata.get("device"),
        "dtype": judge.metadata.get("dtype"),
        "calibrated": False,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    from . import __version__

    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("compare", "batch", "demo"):
        command = commands.add_parser(name)
        command.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help="trained local bundle or Hub ID; first use downloads about 2 GB",
        )
        command.add_argument(
            "--revision",
            help="exact Hub revision; built-in example is pinned automatically",
        )
        command.add_argument(
            "--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"]
        )
        command.add_argument(
            "--dtype",
            default="auto",
            choices=["auto", "float32", "bfloat16", "float16"],
        )
        command.add_argument(
            "--offline", action="store_true", help="use local files/cache only"
        )
        command.add_argument("--cache-dir")
        if name == "demo":
            command.add_argument("--port", type=int, default=7860)
        else:
            command.add_argument(
                "--swap",
                action="store_true",
                help="score both orders and average aligned probabilities",
            )
            command.add_argument("--batch-size", type=int, default=4)
        if name == "compare":
            command.add_argument("--prompt", required=True)
            command.add_argument("--a", required=True)
            command.add_argument("--b", required=True)
        elif name == "batch":
            command.add_argument(
                "--input", required=True, help="JSONL path, or - for stdin"
            )
            command.add_argument(
                "--output", default="-", help="new JSONL path, or - for stdout"
            )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "batch_size") and args.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if args.command == "compare":
            records = [
                input_record(
                    {"prompt": args.prompt, "response_a": args.a, "response_b": args.b}
                )
            ]
        elif args.command == "batch":
            if args.output != "-" and Path(args.output).exists():
                raise FileExistsError("refusing to overwrite batch output")
            stream = (
                sys.stdin
                if args.input == "-"
                else Path(args.input).open(encoding="utf-8")
            )
            try:
                records = []
                for line_number, line in enumerate(stream, 1):
                    if len(line) > 180000:
                        raise ValueError(f"line {line_number}: JSONL record too large")
                    if not line.strip():
                        raise ValueError(f"line {line_number}: blank JSONL record")
                    record = input_record(json.loads(line))
                    if "id" not in json.loads(line):
                        record["id"] = str(line_number - 1)
                    records.append(record)
                    if len(records) > 10000:
                        raise ValueError("batch exceeds 10000 rows; split the input")
            finally:
                if stream is not sys.stdin:
                    stream.close()
        from .judge import PairwiseJudge

        revision = args.revision or (
            DEFAULT_REVISION if args.model == DEFAULT_MODEL else None
        )
        judge = PairwiseJudge.from_pretrained(
            args.model,
            revision=revision,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.offline,
            cache_dir=args.cache_dir,
        )
        judge.metadata["public_model_id"] = args.model
        if args.command == "demo":
            from .demo import serve

            serve(judge, port=args.port)
            return
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        frame = pd.DataFrame(records, columns=["id", *PAIR_COLUMNS])
        predictions = judge.predict(
            frame, swap_debias=args.swap, batch_size=args.batch_size
        )
        metadata = run_metadata(judge, args.swap)
        output = (
            sys.stdout
            if getattr(args, "output", "-") == "-"
            else Path(args.output).open("x", encoding="utf-8")
        )
        try:
            for prediction in predictions:
                output.write(
                    json.dumps(dict(prediction, model=metadata), ensure_ascii=False)
                    + "\n"
                )
        finally:
            if output is not sys.stdout:
                output.close()
    except (ValueError, FileExistsError, OSError, ImportError) as exc:
        print(f"pairjudge: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
