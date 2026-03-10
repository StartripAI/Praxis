"""Tests for praxis.backtest module."""

import numpy as np
import pandas as pd
import pytest
from praxis.backtest.scoring import score, compute_bias, compute_wape, compute_mape, compute_coverage
from praxis.backtest.runner import BacktestRunner, BacktestReport


class TestScoring:
    def test_zero_bias(self):
        a = np.array([100, 200, 300])
        f = np.array([100, 200, 300])
        assert compute_bias(a, f) == 0.0

    def test_positive_bias(self):
        a = np.array([100.0])
        f = np.array([110.0])
        assert abs(compute_bias(a, f) - 0.10) < 0.001

    def test_negative_bias(self):
        a = np.array([100.0])
        f = np.array([90.0])
        assert abs(compute_bias(a, f) - (-0.10)) < 0.001

    def test_wape(self):
        a = np.array([100.0, 200.0])
        f = np.array([110.0, 190.0])
        wape = compute_wape(a, f)
        assert abs(wape - 20 / 300) < 0.001

    def test_mape(self):
        a = np.array([100.0, 200.0])
        f = np.array([110.0, 190.0])
        mape = compute_mape(a, f)
        expected = (0.10 + 0.05) / 2
        assert abs(mape - expected) < 0.001

    def test_coverage(self):
        a = np.array([100, 105, 110, 200])
        lower = np.array([95, 100, 105, 110])
        upper = np.array([115, 110, 115, 120])
        cov = compute_coverage(a, lower, upper)
        assert cov == 0.75  # 3 of 4 within bounds

    def test_score_pass(self):
        a = np.array([100.0, 200.0, 300.0])
        f = np.array([102.0, 198.0, 305.0])
        sc = score(a, f, gate={"max_bias": 0.15, "max_wape": 0.20})
        assert sc.passed is True

    def test_score_fail(self):
        a = np.array([100.0])
        f = np.array([200.0])
        sc = score(a, f, gate={"max_bias": 0.15})
        assert sc.passed is False


class TestBacktestRunner:
    def test_basic_run(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": 100 + np.random.normal(0, 5, 365),
        })

        def simple_predict(train_df, horizon):
            mean_val = train_df["value"].mean()
            return pd.DataFrame({"forecast": [mean_val] * horizon})

        runner = BacktestRunner(n_origins=3, horizon_days=14, step_days=30)
        report = runner.run(data, simple_predict)

        summary = report.summary()
        assert summary["n_origins"] >= 2
        assert "verdict" in summary

    def test_report_repr(self):
        report = BacktestReport(
            [{"origin": "2024-01-01", "bias": 0.05, "abs_bias": 0.05, "wape": 0.08, "passed": True}],
            gate={"max_bias": 0.15},
        )
        r = repr(report)
        assert "BacktestReport" in r
        assert "PASS" in r


class TestEntityTier:
    def test_quantile_classification(self):
        from praxis.analysis.entity_tier import EntityTier
        data = pd.DataFrame({
            "entity": [f"store_{i}" for i in range(30)],
            "value": list(range(30)),
        })
        tier = EntityTier(n_tiers=3)
        result = tier.classify(data)
        assert "tier" in result.columns
        tiers = result["tier"].unique()
        assert len(tiers) <= 3

    def test_auto_detect_bounds(self):
        from praxis.analysis.entity_tier import EntityTier
        data = pd.DataFrame({"value": list(range(100))})
        tier = EntityTier(n_tiers=3)
        bounds = tier.auto_detect_bounds(data)
        assert len(bounds) == 3
        assert "S" in bounds
