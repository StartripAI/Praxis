"""
Feature Engineering — Lag, rolling, and calendar features for forecasting.

Builds standard time-series features:
- Lag features (lag7, lag14, lag28, lag364)
- Rolling statistics (mean, std over 7/14/28 days)
- Calendar features (from CalendarEngine)
- Optional: tsfresh auto-extraction (requires [full] install)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_lag_features(
    df: pd.DataFrame,
    value_col: str = "value",
    lags: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Add lag features to a time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date.
    value_col : str
        Column to create lags for.
    lags : list[int]
        Lag periods in days. Default: [7, 14, 28, 364].

    Returns
    -------
    DataFrame with lag columns added.
    """
    if lags is None:
        lags = [7, 14, 28, 364]

    result = df.copy()
    for lag in lags:
        result[f"lag_{lag}"] = result[value_col].shift(lag)

    return result


def build_rolling_features(
    df: pd.DataFrame,
    value_col: str = "value",
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Add rolling statistics features.

    Parameters
    ----------
    windows : list[int]
        Rolling window sizes in days. Default: [7, 14, 28].
    """
    if windows is None:
        windows = [7, 14, 28]

    result = df.copy()
    for w in windows:
        result[f"rolling_mean_{w}"] = result[value_col].rolling(w, min_periods=1).mean()
        result[f"rolling_std_{w}"] = result[value_col].rolling(w, min_periods=1).std().fillna(0)

    return result


def build_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add calendar-based features from date column.

    Adds: weekday, month, day_of_month, week_of_year, is_weekend,
          is_month_start, is_month_end, quarter.
    """
    result = df.copy()
    dates = pd.to_datetime(result[date_col])

    result["weekday"] = dates.dt.weekday
    result["month"] = dates.dt.month
    result["day_of_month"] = dates.dt.day
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["is_weekend"] = (dates.dt.weekday >= 5).astype(int)
    result["is_month_start"] = dates.dt.is_month_start.astype(int)
    result["is_month_end"] = dates.dt.is_month_end.astype(int)
    result["quarter"] = dates.dt.quarter

    return result


def build_all_features(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    lags: Optional[list[int]] = None,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Build all standard features (lags + rolling + calendar).

    Convenience function combining all feature builders.
    """
    result = df.copy()
    result = build_lag_features(result, value_col, lags)
    result = build_rolling_features(result, value_col, windows)
    result = build_calendar_features(result, date_col)
    return result
