"""
Data Ingestion — CSV/Excel/Sheets loader with schema validation.

Provides standardized data loading for various sources,
normalizing column names and date formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(
    path: str | Path,
    date_col: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load a CSV file with date parsing.

    Normalizes column names to lowercase with underscores.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def load_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    date_col: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load an Excel file with date parsing.

    Normalizes column names to lowercase with underscores.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def validate_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    date_col: str = "date",
) -> dict:
    """Validate DataFrame schema against requirements.

    Returns dict with: valid (bool), missing_cols, extra_cols, n_rows,
    date_range, null_counts.
    """
    missing = [c for c in required_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in required_cols]

    date_range = None
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        date_range = {
            "min": str(dates.min().date()) if not dates.isna().all() else None,
            "max": str(dates.max().date()) if not dates.isna().all() else None,
        }

    null_counts = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}

    return {
        "valid": len(missing) == 0,
        "missing_cols": missing,
        "extra_cols": extra,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "date_range": date_range,
        "null_counts": null_counts,
    }
