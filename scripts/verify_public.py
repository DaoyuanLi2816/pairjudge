"""Verify exact PyPI bytes and an anonymous fresh-cache model/user path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.3.0")
    parser.add_argument("--candidate-hashes", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.candidate_hashes.read_text(encoding="utf-8"))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/pairjudge/{args.version}/json", timeout=30
            ) as response:
                package = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == 5:
                raise
            time.sleep(10)
    public = {file["filename"]: file["digests"]["sha256"] for file in package["urls"]}
    if public != expected:
        raise RuntimeError("public PyPI hashes differ from the qualified wheel/sdist")
    downloads = {}
    for file in package["urls"]:
        with urllib.request.urlopen(file["url"], timeout=60) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected[file["filename"]]:
            raise RuntimeError("actual public distribution bytes differ")
        downloads[file["filename"]] = actual
    if args.install:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                f"pairjudge[judge]=={args.version}",
                "transformers==5.18.0",
            ]
        )
    import numpy as np
    import pandas as pd

    import pairjudge
    from pairjudge import PairwiseJudge
    from pairjudge.catalog import DEFAULT_MODEL, DEFAULT_REVISION

    if pairjudge.__version__ != args.version:
        raise RuntimeError("installed version mismatch")
    reference_path = (
        args.reference
        or Path(__file__).resolve().parents[1]
        / "docs/reports/v0.3.0/download-reference.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    with tempfile.TemporaryDirectory(prefix="pairjudge-public-") as temporary:
        root = Path(temporary)
        cache = root / "fresh-hub-cache"
        judge = PairwiseJudge.from_pretrained(
            DEFAULT_MODEL, revision=DEFAULT_REVISION, device="cpu", cache_dir=cache
        )
        if judge.metadata["resolved_revision"] != DEFAULT_REVISION:
            raise RuntimeError("model revision mismatch")
        frame = pd.DataFrame(reference["records"])
        proba = judge.predict_proba(frame, swap_debias=True, batch_size=2)
        np.testing.assert_allclose(
            proba,
            reference["cpu_swap_probabilities"],
            rtol=0,
            atol=reference["cpu_tolerance"],
        )
        swapped = judge.predict_proba(
            judge._swap(frame), swap_debias=True, batch_size=2
        )
        np.testing.assert_allclose(proba, swapped[:, [1, 0, 2]], rtol=0, atol=1e-7)
        manifest = judge.metadata["artifact_sha256"]
        if manifest != reference["artifact_sha256"]:
            raise RuntimeError("public model manifest changed")
        input_path = root / "input.jsonl"
        input_path.write_text(
            "\n".join(json.dumps(row) for row in reference["records"]), encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "pairjudge",
            "batch",
            "--input",
            str(input_path),
            "--device",
            "cpu",
            "--offline",
            "--cache-dir",
            str(cache),
            "--swap",
            "--batch-size",
            "2",
        ]
        run = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=True
        )
        rows = [json.loads(line) for line in run.stdout.splitlines()]
        assert [row["id"] for row in rows] == [
            str(row["id"]) for row in reference["records"]
        ]
        np.testing.assert_allclose(
            [
                [row["probabilities"][key] for key in ("a_wins", "b_wins", "tie")]
                for row in rows
            ],
            proba,
            rtol=0,
            atol=1e-7,
        )
        compare = subprocess.run(
            [
                sys.executable,
                "-m",
                "pairjudge",
                "compare",
                "--prompt",
                "Why does ice float?",
                "--a",
                "Its crystal structure lowers its density.",
                "--b",
                "Cold objects always rise.",
                "--device",
                "cpu",
                "--offline",
                "--cache-dir",
                str(cache),
                "--swap",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(compare.stdout)["model"]["revision"] == DEFAULT_REVISION
        # Offline API re-load from the newly populated isolated cache.
        offline = PairwiseJudge.from_pretrained(
            DEFAULT_MODEL,
            revision=DEFAULT_REVISION,
            device="cpu",
            cache_dir=cache,
            local_files_only=True,
        )
        np.testing.assert_allclose(
            offline.predict_proba(frame, swap_debias=True, batch_size=2),
            proba,
            atol=1e-7,
        )
    receipt = {
        "version": pairjudge.__version__,
        "pypi_sha256": downloads,
        "model_id": DEFAULT_MODEL,
        "model_revision": DEFAULT_REVISION,
        "artifact_sha256": manifest,
        "anonymous_fresh_cache": True,
        "api_single_batch_swap_offline": True,
        "cli_single_batch_ids_order": True,
        "cpu_tolerance": reference["cpu_tolerance"],
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
