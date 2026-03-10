"""
Backtest Scoring — Bias, WAPE, MAPE, and coverage metrics.

Standard forecasting accuracy metrics with pass/fail gates
to enable verification culture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreCard:
    """Scoring results for a backtest period."""

    bias: float        # (forecast - actual) / actual, signed
    abs_bias: float    # |bias|
    wape: float        # weighted absolute percentage error
    mape: float        # mean absolute percentage error
    rmse: float        # root mean squared error
    coverage_90: float # % of actuals within P10-P90 interval (if available)
    n_observations: int
    passed: bool       # whether all gates were met


def compute_bias(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Signed bias: (sum(forecast) - sum(actual)) / sum(actual)."""
    total_actual = actuals.sum()
    if total_actual == 0:
        return 0.0
    return float((forecasts.sum() - total_actual) / total_actual)


def compute_wape(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum(|f-a|) / sum(a)."""
    total_actual = actuals.sum()
    if total_actual == 0:
        return 0.0
    return float(np.abs(forecasts - actuals).sum() / total_actual)


def compute_mape(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Mean Absolute Percentage Error: mean(|f-a| / a)."""
    mask = actuals > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(forecasts[mask] - actuals[mask]) / actuals[mask]))


def compute_rmse(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((forecasts - actuals) ** 2)))


def compute_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical coverage: % of actuals within [lower, upper]."""
    if len(actuals) == 0:
        return 0.0
    within = ((actuals >= lower) & (actuals <= upper)).sum()
    return float(within / len(actuals))


def score(
    actuals: pd.Series | np.ndarray,
    forecasts: pd.Series | np.ndarray,
    lower: Optional[pd.Series | np.ndarray] = None,
    upper: Optional[pd.Series | np.ndarray] = None,
    gate: Optional[dict] = None,
) -> ScoreCard:
    """Compute all scoring metrics and check gates.

    Parameters
    ----------
    actuals, forecasts : array-like
    lower, upper : array-like, optional
        Prediction interval bounds for coverage calculation.
    gate : dict, optional
        Pass/fail thresholds: {"max_bias": 0.15, "max_wape": 0.20}

    Returns
    -------
    ScoreCard
    """
    a = np.asarray(actuals, dtype=float)
    f = np.asarray(forecasts, dtype=float)

    bias = compute_bias(a, f)
    abs_bias = abs(bias)
    wape = compute_wape(a, f)
    mape = compute_mape(a, f)
    rmse = compute_rmse(a, f)

    coverage = 0.0
    if lower is not None and upper is not None:
        coverage = compute_coverage(a, np.asarray(lower), np.asarray(upper))

    # Check gates
    passed = True
    if gate:
        if "max_bias" in gate and abs_bias > gate["max_bias"]:
            passed = False
        if "max_wape" in gate and wape > gate["max_wape"]:
            passed = False

    return ScoreCard(
        bias=bias,
        abs_bias=abs_bias,
        wape=wape,
        mape=mape,
        rmse=rmse,
        coverage_90=coverage,
        n_observations=len(a),
        passed=passed,
    )
