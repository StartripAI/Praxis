"""
Praxis Utilities — Shared helper functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default when denominator is zero or non-finite."""
    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if np.isfinite(result) else default


def clip_ratio(value: float, lo: float = 0.5, hi: float = 2.0) -> float:
    """Clip a ratio to reasonable bounds."""
    if not np.isfinite(value):
        return 1.0
    return max(lo, min(hi, value))


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names: lowercase, underscores, stripped."""
    result = df.copy()
    result.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in result.columns
    ]
    return result
