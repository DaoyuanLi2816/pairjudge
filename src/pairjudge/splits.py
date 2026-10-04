"""Deterministic groups joining repeated prompts and reversed response pairs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import numpy as np


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def normalized(texts):
    return tuple(
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
        for text in texts
    )


def row_keys(row):
    prompt = normalized(row["prompt"])
    a, b = normalized(row["response_a"]), normalized(row["response_b"])
    return digest(prompt), digest(sorted((a, b))), digest((prompt, sorted((a, b))))


def grouped_ids(frame):
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    keys = [row_keys(row) for _, row in frame.iterrows()]
    for i, row in enumerate(keys):
        for category, key in enumerate(row[:2]):
            value = (category, key)
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    components = {}
    for i in range(len(frame)):
        components.setdefault(find(i), []).append(keys[i][2])
    hashes = {root: digest(sorted(set(values))) for root, values in components.items()}
    return [hashes[find(i)] for i in range(len(frame))]


def grouped_holdout(frame, fraction, seed):
    groups = np.array(grouped_ids(frame))
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("need at least two independent groups for validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
    mask = np.isin(groups, unique[:count])
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(
        drop=True
    )


def assert_disjoint(train, validation):
    left = {key for _, row in train.iterrows() for key in row_keys(row)[:2]}
    right = {key for _, row in validation.iterrows() for key in row_keys(row)[:2]}
    if left & right:
        raise ValueError(
            "training and validation share normalized prompts or response pairs"
        )
