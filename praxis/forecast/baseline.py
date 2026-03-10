"""
Baseline Forecaster — Daytype-average template method.

Uses historical daytype averages (optionally per-entity adjusted)
to produce a forecast. This is the "template method" approach:

  forecast(date) = mean(historical values for same daytype)
                 × entity_adjustment_factor

The daytype is determined by the CalendarEngine (weekday + holiday + vacation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class BaselineForecaster:
    """Template-based forecaster using daytype averages.

    Parameters
    ----------
    method : str
        "daytype_avg" (default) or "dow_avg" (weekday-only, ignoring events).
    lookback_days : int
        Number of historical days to use for averaging, default 365.
    entity_adjustment : bool
        Whether to apply per-entity scaling, default True.
    """

    def __init__(
        self,
        method: str = "daytype_avg",
        lookback_days: int = 365,
        entity_adjustment: bool = True,
    ):
        self.method = method
        self.lookback_days = lookback_days
        self.entity_adjustment = entity_adjustment
        self._templates: dict[str, float] = {}
        self._entity_factors: dict[str, float] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        daytype_col: str = "daytype",
        entity_col: Optional[str] = None,
    ) -> "BaselineForecaster":
        """Learn daytype templates from historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Must have date, value, and daytype columns.
        entity_col : str, optional
            If provided, learns per-entity adjustments.
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter to lookback window
        max_date = df[date_col].max()
        min_date = max_date - pd.Timedelta(days=self.lookback_days)
        df = df[df[date_col] >= min_date]

        self._global_mean = df[value_col].mean()

        # Build daytype templates
        if self.method == "daytype_avg":
            templates = df.groupby(daytype_col)[value_col].mean()
        else:  # dow_avg
            df["_dow"] = df[date_col].dt.weekday
            templates = df.groupby("_dow")[value_col].mean()

        self._templates = templates.to_dict()

        # Entity adjustment factors
        if entity_col and self.entity_adjustment:
            entity_means = df.groupby(entity_col)[value_col].mean()
            global_mean = df[value_col].mean()
            if global_mean > 0:
                self._entity_factors = (entity_means / global_mean).to_dict()

        return self

    def predict(
        self,
        dates: pd.DataFrame,
        daytype_col: str = "daytype",
        entity_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate forecasts for future dates.

        Parameters
        ----------
        dates : pd.DataFrame
            Must have daytype column (from CalendarEngine).
        entity_col : str, optional
            If provided, applies per-entity scaling.

        Returns
        -------
        DataFrame with additional 'forecast' column.
        """
        result = dates.copy()

        if self.method == "daytype_avg":
            result["forecast"] = result[daytype_col].map(self._templates)
        else:
            result["forecast"] = result["date"].dt.weekday.map(self._templates)

        # Fill missing daytypes with global mean
        result["forecast"] = result["forecast"].fillna(self._global_mean)

        # Apply entity adjustment
        if entity_col and self.entity_adjustment and self._entity_factors:
            result["forecast"] = result.apply(
                lambda row: row["forecast"] * self._entity_factors.get(row[entity_col], 1.0),
                axis=1,
            )

        return result

    @property
    def templates(self) -> dict:
        """Return learned daytype templates."""
        return self._templates.copy()
