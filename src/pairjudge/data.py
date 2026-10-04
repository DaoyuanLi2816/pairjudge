"""Validated canonical preference data; labels and conversation history are preserved."""

from __future__ import annotations

import json
import random

import pandas as pd

from .validation import PAIR_COLUMNS, WINNER_COLUMNS, string_list, validate_frame

EMPTY_RESPONSE_PATTERNS = ("[null]", "[]", "[ ]", "[  ]", '[""]', '["",""]')


def _is_empty(series):
    return series.isin(EMPTY_RESPONSE_PATTERNS)


def load_arena_csv(
    path_or_df, drop_identical=False, relabel_empty=False, legacy_cleaning=False
):
    """Decode Arena lists without changing human labels.

    Historical dropping/relabeling requires ``legacy_cleaning=True`` and explicit
    ``drop_identical``/``relabel_empty`` flags. Null content is otherwise invalid.
    """
    df = (
        path_or_df.copy()
        if isinstance(path_or_df, pd.DataFrame)
        else pd.read_csv(path_or_df, encoding="utf-8")
    )
    if (drop_identical or relabel_empty) and not legacy_cleaning:
        raise ValueError("cleaning/relabeling requires explicit legacy_cleaning=True")
    if legacy_cleaning:
        a_empty, b_empty = _is_empty(df.response_a), _is_empty(df.response_b)
        df = df[~(a_empty & b_empty)].copy()
        if drop_identical:
            df = df[df.response_a != df.response_b].copy()
        if relabel_empty:
            df.loc[_is_empty(df.response_a), list(WINNER_COLUMNS)] = [0.0, 1.0, 0.0]
            df.loc[_is_empty(df.response_b), list(WINNER_COLUMNS)] = [1.0, 0.0, 0.0]
    for column in PAIR_COLUMNS:
        df[column] = df[column].map(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
    if legacy_cleaning:
        for index, row in df.iterrows():
            for column in ("response_a", "response_b"):
                if row[column] == [] or row[column] == [None]:
                    df.at[index, column] = [""] * len(row.prompt)
    df = df.reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    validate_frame(df, label_mode="hard")
    return df


def _chat_rounds(messages, row_index):
    if not isinstance(messages, (list, tuple)):
        raise ValueError(f"row {row_index}: expected a list of chat messages")
    prompts, replies, systems = [], [], []
    pending = None
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError(f"row {row_index}: message content must be a string")
        role, text = message.get("role"), message["content"]
        if role == "system" and not prompts and pending is None:
            systems.append(text)
        elif role == "user" and pending is None:
            pending = text
        elif role == "assistant" and pending is not None:
            prefix = (
                "[System]\n" + "\n".join(systems) + "\n[User]\n"
                if systems and not prompts
                else ""
            )
            prompts.append(prefix + pending)
            replies.append(text)
            pending = None
        else:
            raise ValueError(f"row {row_index}: unsupported or unaligned chat roles")
    if not replies:
        raise ValueError(f"row {row_index}: no assistant message")
    if pending is not None:
        raise ValueError(
            f"row {row_index}: trailing user message has no assistant reply"
        )
    return prompts, replies


def load_ultrafeedback(path_or_df, seed=42, dedup_by_prompt=False):
    """Retain every aligned round and shared system context; randomize A/B positions."""
    df = (
        path_or_df
        if isinstance(path_or_df, pd.DataFrame)
        else pd.read_parquet(path_or_df)
    )
    rng, records = random.Random(seed), []
    for index, row in df.iterrows():
        if not isinstance(row.prompt, str):
            raise ValueError(f"prompt at row {index} must be a string")
        cp, chosen = _chat_rounds(row.chosen, index)
        rp, rejected = _chat_rounds(row.rejected, index)
        if cp != rp:
            raise ValueError(f"chosen/rejected shared prompts differ at row {index}")
        first_user = next(m["content"] for m in row.chosen if m["role"] == "user")
        if row.prompt != first_user:
            raise ValueError(f"prompt column differs from conversation at row {index}")
        choose_a = rng.random() > 0.5
        records.append(
            {
                "prompt": cp,
                "response_a": chosen if choose_a else rejected,
                "response_b": rejected if choose_a else chosen,
                "winner_model_a": float(choose_a),
                "winner_model_b": float(not choose_a),
                "winner_tie": 0.0,
            }
        )
    out = pd.DataFrame(records, columns=list(PAIR_COLUMNS) + list(WINNER_COLUMNS))
    if dedup_by_prompt:
        out["_key"] = out.prompt.map(tuple)
        out = out.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    out["id"] = out.index.astype(str)
    out.attrs["preprocessing"] = {
        "dedup_by_prompt": bool(dedup_by_prompt),
        "input_rows": len(df),
        "output_rows": len(out),
        "dropped_rows": len(df) - len(out),
    }
    validate_frame(out, label_mode="hard")
    return out


def from_pairs(prompts, responses_a, responses_b, winners=None):
    """Construct single-turn pairs from equal-length sequences of strings."""
    values = [
        string_list(value, name)
        for value, name in zip(
            (prompts, responses_a, responses_b),
            ("prompts", "responses_a", "responses_b"),
        )
    ]
    if len({len(value) for value in values}) != 1:
        raise ValueError(
            "prompts, responses_a and responses_b must have the same length"
        )
    df = pd.DataFrame(
        {name: [[text] for text in value] for name, value in zip(PAIR_COLUMNS, values)}
    )
    if winners is not None:
        winners = string_list(winners, "winners")
        if len(winners) != len(df):
            raise ValueError("winners must match the pair count")
        if set(winners) - {"a", "b", "tie"}:
            raise ValueError("winners must be 'a', 'b' or 'tie'")
        for name, label in zip(WINNER_COLUMNS, ("a", "b", "tie")):
            df[name] = [float(winner == label) for winner in winners]
    df["id"] = df.index.astype(str)
    return df


def empty_and_identical_masks(df):
    """Historical raw-CSV diagnostics; inference never applies fixed probabilities."""
    return (
        _is_empty(df.response_a),
        _is_empty(df.response_b),
        df.response_a == df.response_b,
    )
