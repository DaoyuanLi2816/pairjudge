"""Pseudo-label an unlabeled preference pool with a trained judge.

The semi-supervised loop that drove the gold-medal result:

1. Train a judge on human-labeled data (``label_mode: hard``).
2. Run that judge over a large unlabeled pool of (prompt, A, B) pairs —
   this module — storing the *full probability distribution*, not the argmax.
3. Train a fresh judge on human labels + soft pseudo-labels
   (``label_mode: soft``, KL loss).

Keeping the distribution matters: an 0.9/0.1 verdict and a 0.4/0.35/0.25
verdict carry very different information, and KL distillation preserves that.

Run from the command line::

    python -m pairjudge.pseudo_label \
        --model ./output/judge/merged \
        --data pool.parquet --out pool_pl.parquet
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .judge import PairwiseJudge


def pseudo_label(
    judge: PairwiseJudge,
    df: pd.DataFrame,
    batch_size: int = 4,
    swap_debias: bool = False,
) -> pd.DataFrame:
    """Attach soft winner columns predicted by ``judge`` to ``df``."""
    proba = judge.predict_proba(df, swap_debias=swap_debias, batch_size=batch_size)
    out = df.copy()
    out["winner_model_a"] = proba[:, 0]
    out["winner_model_b"] = proba[:, 1]
    out["winner_tie"] = proba[:, 2]
    out.attrs["pairjudge_teacher"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": judge.metadata.get("resolved_revision"),
        "artifact_sha256": judge.metadata.get("artifact_sha256"),
        "source": judge.metadata.get("source", {}),
        "class_order": ["a_wins", "b_wins", "tie"],
        "packer": asdict(judge.packer.config),
        "mode": "swap_average" if swap_debias else "single",
        "dtype": judge.metadata.get("dtype"),
        "labels": "teacher probabilities, not independent human annotations",
    }
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained judge (path or hub id)")
    parser.add_argument(
        "--data", required=True, help="Canonical-schema parquet to label"
    )
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--swap-debias",
        action="store_true",
        help="Average both orders after exchanging A/B columns (two passes)",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.out)
    if (
        output_path.exists()
        or output_path.with_suffix(output_path.suffix + ".json").exists()
    ):
        raise FileExistsError("refusing to overwrite pseudo-label output or provenance")

    judge = PairwiseJudge.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        local_files_only=args.offline,
    )
    df = pd.read_parquet(args.data)
    out = pseudo_label(
        judge, df, batch_size=args.batch_size, swap_debias=args.swap_debias
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(out.attrs["pairjudge_teacher"], indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(out)} pseudo-labeled rows to {output_path}")


if __name__ == "__main__":
    main()
