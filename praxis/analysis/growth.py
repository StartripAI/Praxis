"""
Growth Decomposition — DM-Growth with YoY/MoM blend and Bayesian shrinkage.

Decomposes observed growth into components:
- Year-over-Year (YoY) growth
- Month-over-Month (MoM) growth
- Blended growth factor (weighted combination)

Uses coverage-based Bayesian shrinkage: when data coverage is low,
the growth estimate is shrunk toward the global mean to avoid
noise-driven outlier estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GrowthResult:
    """Container for growth decomposition results."""

    yoy_growth: float
    mom_growth: float
    blended_growth: float
    yoy_weight: float
    mom_weight: float
    coverage: float
    shrinkage_applied: bool
    raw_blended: float  # before shrinkage


class GrowthDecomposer:
    """Decompose metric growth into YoY/MoM with optional Bayesian shrinkage.

    Parameters
    ----------
    yoy_weight : float
        Weight for year-over-year component, default 0.70.
    mom_weight : float
        Weight for month-over-month component, default 0.30.
    shrinkage : str
        Shrinkage method: "bayesian" or "none", default "bayesian".
    min_coverage : float
        Minimum data coverage (0-1) to avoid full shrinkage, default 0.5.
    """

    def __init__(
        self,
        yoy_weight: float = 0.70,
        mom_weight: float = 0.30,
        shrinkage: str = "bayesian",
        min_coverage: float = 0.5,
    ):
        assert abs(yoy_weight + mom_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.yoy_weight = yoy_weight
        self.mom_weight = mom_weight
        self.shrinkage = shrinkage
        self.min_coverage = min_coverage

    def decompose(
        self,
        current: float,
        yoy_reference: float,
        mom_reference: float,
        coverage: float = 1.0,
        global_growth: Optional[float] = None,
    ) -> GrowthResult:
        """Decompose growth for a single metric/entity.

        Parameters
        ----------
        current : float
            Current period value
        yoy_reference : float
            Same period last year value
        mom_reference : float
            Previous month value
        coverage : float
            Data coverage ratio (0-1), used for shrinkage
        global_growth : float, optional
            Global/population mean growth for shrinkage target

        Returns
        -------
        GrowthResult
        """
        # Calculate raw growth rates
        yoy = (current / yoy_reference - 1.0) if yoy_reference > 0 else 0.0
        mom = (current / mom_reference - 1.0) if mom_reference > 0 else 0.0

        # Blend
        raw_blended = yoy * self.yoy_weight + mom * self.mom_weight

        # Apply Bayesian shrinkage based on coverage
        shrinkage_applied = False
        blended = raw_blended

        if self.shrinkage == "bayesian" and coverage < 1.0:
            target = global_growth if global_growth is not None else 0.0
            # Shrinkage factor: more data → less shrinkage
            alpha = min(coverage / self.min_coverage, 1.0)
            blended = alpha * raw_blended + (1 - alpha) * target
            shrinkage_applied = alpha < 1.0

        return GrowthResult(
            yoy_growth=yoy,
            mom_growth=mom,
            blended_growth=blended,
            yoy_weight=self.yoy_weight,
            mom_weight=self.mom_weight,
            coverage=coverage,
            shrinkage_applied=shrinkage_applied,
            raw_blended=raw_blended,
        )

    def decompose_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str = "value",
        entity_col: str = "entity",
        period_col: str = "period",
    ) -> pd.DataFrame:
        """Batch growth decomposition over a DataFrame.

        Expected columns: entity, period, value.
        Periods should be sortable (e.g. "2025-01", "2025-02").

        Returns DataFrame with growth decomposition per entity-period.
        """
        results = []
        entities = df[entity_col].unique()

        for entity in entities:
            entity_df = df[df[entity_col] == entity].sort_values(period_col)
            periods = entity_df[period_col].tolist()
            values = entity_df[value_col].tolist()

            for i, (period, current) in enumerate(zip(periods, values)):
                # YoY: need period - 12 months (if available)
                yoy_ref = values[i - 12] if i >= 12 else current
                # MoM: previous period
                mom_ref = values[i - 1] if i >= 1 else current
                # Coverage: simple proxy based on data availability
                coverage = min(i / 12.0, 1.0) if i > 0 else 0.0

                # Global growth for shrinkage
                global_vals = df[df[period_col] == period][value_col]
                global_prev = df[df[period_col] == periods[i - 1]][value_col] if i > 0 else global_vals
                global_growth = (
                    (global_vals.sum() / global_prev.sum() - 1.0) if global_prev.sum() > 0 else 0.0
                )

                result = self.decompose(current, yoy_ref, mom_ref, coverage, global_growth)
                results.append(
                    {
                        entity_col: entity,
                        period_col: period,
                        value_col: current,
                        "yoy_growth": result.yoy_growth,
                        "mom_growth": result.mom_growth,
                        "blended_growth": result.blended_growth,
                        "coverage": result.coverage,
                        "shrinkage_applied": result.shrinkage_applied,
                    }
                )

        return pd.DataFrame(results)
