"""
DOW Learner — Day-of-week share estimation from data.

Learns the typical distribution of a metric across weekdays
(what % of weekly total falls on Mon, Tue, ... Sun).
Replaces hardcoded DOW_SHARES tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DOWLearner:
    """Learn day-of-week distributional shares from data.

    Parameters
    ----------
    min_weeks : int
        Minimum weeks of data needed for reliable estimation, default 4.
    """

    def __init__(self, min_weeks: int = 4):
        self.min_weeks = min_weeks
        self._shares: dict[str, np.ndarray] = {}

    def learn(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        entity_col: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Learn DOW shares from historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Must have date and value columns.
        entity_col : str, optional
            If provided, learns shares per entity.

        Returns
        -------
        dict mapping entity (or "__global__") → array of 7 shares (Mon-Sun)
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["_dow"] = df[date_col].dt.weekday
        df["_week"] = df[date_col].dt.isocalendar().week.astype(int)
        df["_year"] = df[date_col].dt.year

        groups = [entity_col] if entity_col else []

        if groups:
            for entity, edf in df.groupby(entity_col):
                shares = self._compute_shares(edf, value_col)
                if shares is not None:
                    self._shares[str(entity)] = shares
        else:
            shares = self._compute_shares(df, value_col)
            if shares is not None:
                self._shares["__global__"] = shares

        return self._shares

    def _compute_shares(self, df: pd.DataFrame, value_col: str) -> np.ndarray | None:
        """Compute DOW shares for a subset of data."""
        n_weeks = df.groupby(["_year", "_week"]).ngroups
        if n_weeks < self.min_weeks:
            return None

        # Weekly-normalized shares
        weekly = df.groupby(["_year", "_week", "_dow"])[value_col].sum().reset_index()
        week_totals = weekly.groupby(["_year", "_week"])[value_col].transform("sum")
        weekly["_share"] = np.where(week_totals > 0, weekly[value_col] / week_totals, 0)

        # Median share per DOW (robust to outliers)
        dow_shares = weekly.groupby("_dow")["_share"].median()
        shares = np.zeros(7)
        for dow, share in dow_shares.items():
            shares[dow] = share

        # Normalize to sum to 1
        total = shares.sum()
        if total > 0:
            shares /= total

        return shares

    def get_shares(self, entity: str = "__global__") -> np.ndarray:
        """Get learned DOW shares for an entity.

        Returns array of 7 floats (Mon-Sun) summing to 1.0.
        Falls back to uniform (1/7 each) if not learned.
        """
        if entity in self._shares:
            return self._shares[entity]
        if "__global__" in self._shares:
            return self._shares["__global__"]
        return np.full(7, 1.0 / 7.0)

    def distribute_weekly(self, weekly_total: float, entity: str = "__global__") -> np.ndarray:
        """Distribute a weekly total across days using learned shares.

        Returns array of 7 daily values (Mon-Sun).
        """
        return weekly_total * self.get_shares(entity)
