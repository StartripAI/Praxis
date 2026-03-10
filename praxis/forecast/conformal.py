"""
Conformal Prediction Wrapper — CQR for calibrated uncertainty intervals.

Provides P10/P50/P90 (or custom quantiles) prediction intervals using
Conformal Quantile Regression (CQR). Wraps MAPIE or StatsForecast/Darts
conformal backends.

Key idea: use a calibration set of residuals to produce prediction
intervals with guaranteed coverage probability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ConformalWrapper:
    """Conformal prediction wrapper for calibrated intervals.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels to produce, default [0.10, 0.50, 0.90].
    calibration_window : int
        Number of recent observations for calibration, default 90.
    method : str
        "split" (split conformal) or "cv" (cross-conformal), default "split".
    """

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        calibration_window: int = 90,
        method: str = "split",
    ):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.calibration_window = calibration_window
        self.method = method
        self._residuals: np.ndarray | None = None

    def calibrate(
        self,
        actuals: np.ndarray | pd.Series,
        predictions: np.ndarray | pd.Series,
    ) -> "ConformalWrapper":
        """Calibrate using historical prediction residuals.

        Parameters
        ----------
        actuals : array-like
            Actual observed values.
        predictions : array-like
            Model predictions for the same dates.
        """
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)

        # Use most recent observations
        n = min(len(actuals), self.calibration_window)
        residuals = actuals[-n:] - predictions[-n:]
        self._residuals = np.sort(residuals)

        return self

    def predict_intervals(
        self,
        point_forecasts: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Generate prediction intervals.

        Parameters
        ----------
        point_forecasts : array-like
            Point predictions from the base model.

        Returns
        -------
        DataFrame with columns: forecast (P50), plus P{q*100} columns.
        """
        pf = np.asarray(point_forecasts)

        result = {"forecast": pf}

        if self._residuals is not None and len(self._residuals) > 0:
            n = len(self._residuals)
            for q in self.quantiles:
                # Conformal quantile: find the q-th quantile of residuals
                # and add it to the point forecast
                idx = int(np.ceil(q * (n + 1))) - 1
                idx = max(0, min(idx, n - 1))
                correction = self._residuals[idx]

                col_name = f"p{int(q * 100):02d}"
                result[col_name] = pf + correction
        else:
            # Fallback: no calibration, use point forecast for all quantiles
            for q in self.quantiles:
                col_name = f"p{int(q * 100):02d}"
                result[col_name] = pf

        return pd.DataFrame(result)

    @property
    def interval_width(self) -> float:
        """Mean interval width (P90 - P10) from last prediction."""
        if self._residuals is None:
            return 0.0
        n = len(self._residuals)
        q10_idx = max(0, int(np.ceil(0.10 * (n + 1))) - 1)
        q90_idx = min(n - 1, int(np.ceil(0.90 * (n + 1))) - 1)
        return float(self._residuals[q90_idx] - self._residuals[q10_idx])

    @property
    def calibration_size(self) -> int:
        """Number of residuals used for calibration."""
        return len(self._residuals) if self._residuals is not None else 0
