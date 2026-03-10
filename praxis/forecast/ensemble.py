"""
Ensemble Router — Route and blend multiple forecasters.

Combines baseline, LGBM, and optional Darts/AutoTS backends
using backtest-score-based weighting or simple averaging.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    """Protocol for forecaster backends."""

    def predict(self, data: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        ...


class EnsembleRouter:
    """Route and blend multiple forecasters.

    Parameters
    ----------
    weight_method : str
        "backtest_score" (default) — weight by inverse error from backtest.
        "equal" — equal weight for all methods.
    """

    def __init__(self, weight_method: str = "backtest_score"):
        self.weight_method = weight_method
        self._forecasters: dict[str, Forecaster] = {}
        self._weights: dict[str, float] = {}

    def register(self, name: str, forecaster: Forecaster, weight: float = 1.0):
        """Register a forecaster with an optional initial weight."""
        self._forecasters[name] = forecaster
        self._weights[name] = weight

    def set_weights_from_backtest(self, scores: dict[str, float]):
        """Set weights from backtest scores (lower = better).

        Weights are proportional to 1/score. Pass WAPE or MAPE scores.
        """
        total_inv = sum(1.0 / max(s, 0.001) for s in scores.values())
        for name, score in scores.items():
            self._weights[name] = (1.0 / max(score, 0.001)) / total_inv

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate blended ensemble forecast.

        Returns DataFrame with columns: forecast, plus individual method columns.
        """
        if not self._forecasters:
            raise RuntimeError("No forecasters registered. Call .register() first.")

        result = data.copy()
        weighted_sum = np.zeros(len(data))
        total_weight = 0.0

        for name, forecaster in self._forecasters.items():
            preds = forecaster.predict(data)
            if isinstance(preds, pd.DataFrame):
                col = "forecast" if "forecast" in preds.columns else preds.columns[-1]
                preds = preds[col].values
            result[f"forecast_{name}"] = preds
            w = self._weights.get(name, 1.0)
            weighted_sum += preds * w
            total_weight += w

        if total_weight > 0:
            result["forecast"] = weighted_sum / total_weight
        else:
            result["forecast"] = weighted_sum

        return result

    @property
    def weights(self) -> dict[str, float]:
        """Return current weights."""
        return self._weights.copy()
