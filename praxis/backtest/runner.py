"""
Backtest Runner — Walk-forward rolling-origin validation.

Implements a proper walk-forward backtest:
1. For each origin date, train on all data before the origin
2. Forecast the next `horizon` days
3. Score forecast against actuals
4. Move to next origin and repeat

This creates a "verification culture" with auditable results.
"""

from __future__ import annotations

from typing import Optional, Callable

import pandas as pd

from praxis.backtest.scoring import score, ScoreCard


class BacktestRunner:
    """Walk-forward rolling-origin backtest.

    Parameters
    ----------
    n_origins : int
        Number of rolling forecast origins, default 6.
    horizon_days : int
        Forecast horizon in days, default 31.
    step_days : int
        Days between origins, default 30.
    gate : dict, optional
        Pass/fail thresholds for scoring.
    """

    def __init__(
        self,
        n_origins: int = 6,
        horizon_days: int = 31,
        step_days: int = 30,
        gate: Optional[dict] = None,
    ):
        self.n_origins = n_origins
        self.horizon_days = horizon_days
        self.step_days = step_days
        self.gate = gate or {"max_bias": 0.15, "max_wape": 0.20}
        self._results: list[dict] = []

    def run(
        self,
        data: pd.DataFrame,
        fit_predict_fn: Callable,
        date_col: str = "date",
        value_col: str = "value",
    ) -> "BacktestReport":
        """Run walk-forward backtest.

        Parameters
        ----------
        data : pd.DataFrame
            Full historical data with date and value columns.
        fit_predict_fn : callable
            Function(train_df, horizon) → forecast_df.
            Must return DataFrame with 'forecast' column.

        Returns
        -------
        BacktestReport
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        max_date = df[date_col].max()
        self._results = []

        for i in range(self.n_origins):
            # Calculate origin date (working backward from max_date)
            origin = max_date - pd.Timedelta(days=(self.n_origins - i) * self.step_days)
            forecast_end = origin + pd.Timedelta(days=self.horizon_days)

            # Split
            train = df[df[date_col] <= origin]
            test = df[(df[date_col] > origin) & (df[date_col] <= forecast_end)]

            if len(train) < 30 or len(test) == 0:
                continue

            # Fit and predict
            try:
                forecast_df = fit_predict_fn(train, self.horizon_days)
                if len(forecast_df) == 0:
                    continue

                # Align forecast and actuals
                n = min(len(test), len(forecast_df))
                actuals = test[value_col].values[:n]
                forecasts = forecast_df["forecast"].values[:n]

                # Score
                sc = score(actuals, forecasts, gate=self.gate)

                self._results.append(
                    {
                        "origin": origin,
                        "origin_idx": i,
                        "n_test": n,
                        "bias": sc.bias,
                        "abs_bias": sc.abs_bias,
                        "wape": sc.wape,
                        "mape": sc.mape,
                        "rmse": sc.rmse,
                        "passed": sc.passed,
                    }
                )
            except Exception as e:
                self._results.append(
                    {
                        "origin": origin,
                        "origin_idx": i,
                        "n_test": 0,
                        "error": str(e),
                        "passed": False,
                    }
                )

        return BacktestReport(self._results, self.gate)


class BacktestReport:
    """Container for backtest results with summary and pass/fail verdict."""

    def __init__(self, results: list[dict], gate: dict):
        self.results = results
        self.gate = gate
        self._df = pd.DataFrame(results) if results else pd.DataFrame()

    def summary(self) -> dict:
        """Generate summary statistics."""
        if self._df.empty:
            return {"n_origins": 0, "pass_rate": 0.0, "verdict": "NO_DATA"}

        n = len(self._df)
        n_pass = self._df["passed"].sum() if "passed" in self._df.columns else 0

        return {
            "n_origins": n,
            "n_pass": int(n_pass),
            "n_fail": int(n - n_pass),
            "pass_rate": round(n_pass / n * 100, 1),
            "avg_bias": round(self._df.get("bias", pd.Series([0])).mean(), 4),
            "avg_abs_bias": round(self._df.get("abs_bias", pd.Series([0])).mean(), 4),
            "avg_wape": round(self._df.get("wape", pd.Series([0])).mean(), 4),
            "gate": self.gate,
            "verdict": "PASS" if n_pass == n else "FAIL",
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        return self._df.copy()

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BacktestReport(origins={s['n_origins']}, "
            f"pass_rate={s['pass_rate']}%, "
            f"avg_bias={s['avg_bias']:.1%}, "
            f"verdict={s['verdict']})"
        )
