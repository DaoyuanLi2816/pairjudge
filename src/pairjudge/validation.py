"""Shared input contracts; no model dependencies or implicit row cleaning."""

from __future__ import annotations

from typing import Sequence

import numpy as np

CLASSES = ("a_wins", "b_wins", "tie")
PAIR_COLUMNS = ("prompt", "response_a", "response_b")
WINNER_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")


def string_list(value, name: str) -> list[str]:
    if isinstance(value, (str, bytes, dict)):
        raise ValueError(
            f"{name} must be a sequence of strings, not a scalar string/mapping"
        )
    try:
        result = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not isinstance(item, str) for item in result):
        raise ValueError(
            f"{name} must contain only strings; missing/null values are invalid"
        )
    return result


def validate_rounds(prompts, responses_a, responses_b):
    columns = [
        string_list(value, name)
        for value, name in zip((prompts, responses_a, responses_b), PAIR_COLUMNS)
    ]
    if len({len(value) for value in columns}) != 1:
        raise ValueError("prompt and responses must have the same number of rounds")
    return columns


def probabilities(values, name: str = "probabilities") -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{name} rows must sum to 1")
    return array


def class_order(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str) or len(values) != 3 or set(values) != set(CLASSES):
        raise ValueError(f"label_order must be a permutation of {CLASSES}")
    return tuple(values)


def validate_frame(df, label_mode: str = "none"):
    missing = [name for name in PAIR_COLUMNS if name not in df]
    if missing:
        raise ValueError(f"missing pair columns: {missing}")
    for position, values in enumerate(zip(*(df[name] for name in PAIR_COLUMNS))):
        try:
            validate_rounds(*values)
        except ValueError as exc:
            raise ValueError(f"invalid pair at position {position}: {exc}") from exc
    if label_mode != "none":
        missing = [name for name in WINNER_COLUMNS if name not in df]
        if missing:
            raise ValueError(f"missing winner columns: {missing}")
        labels = probabilities(df[list(WINNER_COLUMNS)].to_numpy(), "winner labels")
        if label_mode == "hard" and not np.all((labels == 0) | (labels == 1)):
            raise ValueError("hard winner labels must be one-hot")
    return df


def decisions(proba, atol: float = 1e-8) -> list[str]:
    """Numerical maximum ties are ambiguous, not the learned tie category."""
    array = probabilities(proba)
    result = []
    for row in array:
        maximum = np.flatnonzero(np.isclose(row, row.max(), rtol=0, atol=atol))
        result.append(CLASSES[int(maximum[0])] if len(maximum) == 1 else "ambiguous")
    return result
