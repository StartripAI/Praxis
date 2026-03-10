"""
Event Effect Learner — Auto-detect event impacts from data.

Replaces hardcoded event coefficients with data-driven estimation.
Uses before/after comparison with optional CausalImpact backend.

Learns: "How much does event X affect metric Y?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EventEffect:
    """Estimated effect of an event on a metric."""

    event_name: str
    effect_pct: float       # % change attributed to event
    confidence: float       # 0-1 confidence in estimate
    n_observations: int     # how many historical events used
    method: str             # "before_after" or "causal_impact"


class EventLearner:
    """Learn event effects from historical data.

    Approach:
    1. Identify event windows (from calendar)
    2. Compare event-period values to counterfactual (non-event baseline)
    3. Estimate effect as % deviation from counterfactual

    Parameters
    ----------
    min_observations : int
        Minimum historical occurrences needed to learn, default 2.
    window_days : int
        Days before/after event to use as control, default 7.
    """

    def __init__(self, min_observations: int = 2, window_days: int = 7):
        self.min_observations = min_observations
        self.window_days = window_days
        self._effects: dict[str, EventEffect] = {}

    def learn(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        event_col: str = "event_name",
    ) -> dict[str, EventEffect]:
        """Learn event effects from historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Must have columns: date, value, event_name.
            event_name is "" or NaN for non-event days.

        Returns
        -------
        dict mapping event_name → EventEffect
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Identify unique events
        events = df[event_col].dropna().unique()
        events = [e for e in events if e and str(e).strip()]

        for event_name in events:
            event_dates = df[df[event_col] == event_name][date_col].tolist()
            if len(event_dates) < self.min_observations:
                continue

            effects = []
            for ed in event_dates:
                # Control window: days around the event that are NOT the event
                window_start = ed - pd.Timedelta(days=self.window_days)
                window_end = ed + pd.Timedelta(days=self.window_days)
                control = df[
                    (df[date_col] >= window_start)
                    & (df[date_col] <= window_end)
                    & (df[event_col] != event_name)
                ][value_col]

                event_val = df[df[date_col] == ed][value_col].values
                if len(event_val) > 0 and len(control) > 0 and control.mean() > 0:
                    effect = (event_val[0] / control.mean()) - 1.0
                    effects.append(effect)

            if effects:
                self._effects[event_name] = EventEffect(
                    event_name=event_name,
                    effect_pct=float(np.median(effects)),
                    confidence=min(len(effects) / 5.0, 1.0),
                    n_observations=len(effects),
                    method="before_after",
                )

        return self._effects

    def get_effect(self, event_name: str) -> Optional[EventEffect]:
        """Get the learned effect for a specific event."""
        return self._effects.get(event_name)

    def apply(
        self,
        baseline: float,
        event_name: str,
    ) -> float:
        """Apply learned event effect to a baseline value.

        Returns adjusted value, or baseline if event not learned.
        """
        effect = self._effects.get(event_name)
        if effect is None:
            return baseline
        return baseline * (1.0 + effect.effect_pct)

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all learned effects."""
        if not self._effects:
            return pd.DataFrame(columns=["event_name", "effect_pct", "confidence", "n_observations", "method"])
        return pd.DataFrame([vars(e) for e in self._effects.values()])
