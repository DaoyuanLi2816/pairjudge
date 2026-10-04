"""Shared loading keyword for the qualified Transformers 4.57.6+ / 5.x stack."""

from __future__ import annotations

from typing import Any


def model_dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Both qualified lines accept dtype; torch_dtype is already deprecated."""
    return {"dtype": dtype}
