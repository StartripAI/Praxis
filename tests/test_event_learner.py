"""Tests for praxis.analysis.event_learner module."""

import pandas as pd
import numpy as np
from praxis.analysis.event_learner import EventLearner


class TestEventLearner:
    def _make_data(self):
        """Create synthetic data with a known event effect."""
        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        np.random.seed(42)
        values = 100 + np.random.normal(0, 5, len(dates))

        events = [""] * len(dates)
        # Add "big_promo" events with +30% spike
        for d in ["2024-06-18", "2024-11-11", "2025-06-18", "2025-11-11"]:
            idx = (dates == pd.Timestamp(d)).argmax()
            values[idx] *= 1.30
            events[idx] = "big_promo"

        return pd.DataFrame({"date": dates, "value": values, "event_name": events})

    def test_learn_detects_event(self):
        data = self._make_data()
        learner = EventLearner(min_observations=2, window_days=7)
        effects = learner.learn(data)
        assert "big_promo" in effects
        assert effects["big_promo"].effect_pct > 0.10  # Should detect positive effect

    def test_apply_effect(self):
        data = self._make_data()
        learner = EventLearner(min_observations=2)
        learner.learn(data)
        adjusted = learner.apply(100.0, "big_promo")
        assert adjusted > 100.0

    def test_unknown_event_returns_baseline(self):
        learner = EventLearner()
        assert learner.apply(100.0, "nonexistent") == 100.0

    def test_summary_table(self):
        data = self._make_data()
        learner = EventLearner(min_observations=2)
        learner.learn(data)
        summary = learner.summary()
        assert len(summary) >= 1
        assert "effect_pct" in summary.columns

    def test_min_observations_filter(self):
        data = self._make_data()
        learner = EventLearner(min_observations=10)  # Too high
        effects = learner.learn(data)
        assert "big_promo" not in effects


class TestDOWLearner:
    def test_learn_global(self):
        from praxis.analysis.dow_learner import DOWLearner
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        np.random.seed(42)
        # Weekends higher
        values = [100 + (30 if d.weekday() >= 5 else 0) + np.random.normal(0, 5) for d in dates]
        data = pd.DataFrame({"date": dates, "value": values})

        learner = DOWLearner(min_weeks=4)
        shares = learner.learn(data)
        assert "__global__" in shares
        # Weekend shares should be higher
        global_shares = shares["__global__"]
        assert global_shares[5] > global_shares[0]  # Sat > Mon
        assert abs(global_shares.sum() - 1.0) < 0.01

    def test_distribute_weekly(self):
        from praxis.analysis.dow_learner import DOWLearner
        learner = DOWLearner()
        daily = learner.distribute_weekly(700.0)
        assert abs(daily.sum() - 700.0) < 0.01
