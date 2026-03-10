"""
Entity Tier — Data-driven entity classification.

Classifies entities (stores, products, regions) into tiers
based on their metric values, using quantile-based or
custom-threshold grouping.

Replaces hardcoded store tier seeds with auto-detection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class EntityTier:
    """Classify entities into tiers based on metric values.

    Parameters
    ----------
    method : str
        "quantile" (default) or "threshold".
    n_tiers : int
        Number of tiers for quantile method, default 3.
    thresholds : list[float], optional
        Custom thresholds for "threshold" method.
    tier_labels : list[str], optional
        Labels for tiers. Default: ["S", "A", "B"] for 3 tiers.
    """

    def __init__(
        self,
        method: str = "quantile",
        n_tiers: int = 3,
        thresholds: Optional[list[float]] = None,
        tier_labels: Optional[list[str]] = None,
    ):
        self.method = method
        self.n_tiers = n_tiers
        self.thresholds = thresholds
        self.tier_labels = tier_labels or self._default_labels(n_tiers)

    @staticmethod
    def _default_labels(n: int) -> list[str]:
        """Generate default tier labels: S, A, B, C, ..."""
        labels = ["S", "A", "B", "C", "D", "E"]
        return labels[:n] if n <= len(labels) else [f"T{i+1}" for i in range(n)]

    def classify(
        self,
        data: pd.DataFrame,
        entity_col: str = "entity",
        value_col: str = "value",
    ) -> pd.DataFrame:
        """Classify entities into tiers.

        Parameters
        ----------
        data : pd.DataFrame
            Must have entity and value columns. Value is the metric
            to use for tiering (e.g., average monthly revenue).

        Returns
        -------
        DataFrame with columns: entity, value, tier, tier_rank
        """
        agg = data.groupby(entity_col)[value_col].mean().reset_index()
        agg = agg.sort_values(value_col, ascending=False).reset_index(drop=True)

        if self.method == "quantile":
            quantiles = np.linspace(1.0, 0.0, self.n_tiers + 1)
            cuts = [agg[value_col].quantile(q) for q in quantiles]
            tiers = []
            for _, row in agg.iterrows():
                tier_idx = 0
                for i in range(len(cuts) - 1):
                    if row[value_col] >= cuts[i + 1]:
                        tier_idx = i
                        break
                tiers.append(self.tier_labels[min(tier_idx, len(self.tier_labels) - 1)])
            agg["tier"] = tiers

        elif self.method == "threshold" and self.thresholds:
            sorted_thresh = sorted(self.thresholds, reverse=True)
            tiers = []
            for _, row in agg.iterrows():
                assigned = self.tier_labels[-1]
                for i, t in enumerate(sorted_thresh):
                    if row[value_col] >= t:
                        assigned = self.tier_labels[min(i, len(self.tier_labels) - 1)]
                        break
                tiers.append(assigned)
            agg["tier"] = tiers

        else:
            agg["tier"] = self.tier_labels[0]

        agg["tier_rank"] = agg["tier"].map(
            {label: i for i, label in enumerate(self.tier_labels)}
        )

        return agg

    def auto_detect_bounds(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
    ) -> dict[str, tuple[float, float]]:
        """Auto-detect reasonable bounds for each tier.

        Returns dict mapping tier_label → (lower_bound, upper_bound).
        Replaces hardcoded per-tier bound constants with data-driven detection.
        """
        values = data[value_col].dropna()
        if len(values) == 0:
            return {label: (0.0, float("inf")) for label in self.tier_labels}

        quantiles = np.linspace(1.0, 0.0, self.n_tiers + 1)
        bounds = {}
        for i, label in enumerate(self.tier_labels):
            upper = float(values.quantile(quantiles[i])) if i > 0 else float("inf")
            lower = float(values.quantile(quantiles[i + 1]))
            bounds[label] = (lower, upper)

        return bounds
