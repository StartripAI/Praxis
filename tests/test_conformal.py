"""Tests for praxis.forecast.conformal module."""

import numpy as np
import pandas as pd
import pytest
from praxis.forecast.conformal import ConformalWrapper


class TestConformalWrapper:
    def test_calibrate_and_predict(self):
        np.random.seed(42)
        actuals = np.array([100, 105, 98, 110, 95, 102, 108, 97, 103, 106])
        predictions = np.array([101, 103, 100, 108, 96, 104, 106, 99, 101, 104])

        cw = ConformalWrapper(quantiles=[0.10, 0.50, 0.90])
        cw.calibrate(actuals, predictions)

        future_preds = np.array([100, 110, 105])
        intervals = cw.predict_intervals(future_preds)

        assert "p10" in intervals.columns
        assert "p50" in intervals.columns
        assert "p90" in intervals.columns
        assert len(intervals) == 3
        # P10 <= P50 <= P90
        assert (intervals["p10"] <= intervals["p90"]).all()

    def test_no_calibration_fallback(self):
        cw = ConformalWrapper()
        preds = np.array([100, 200])
        intervals = cw.predict_intervals(preds)
        # Without calibration, all quantiles equal forecast
        assert (intervals["p10"] == preds).all()
        assert (intervals["p90"] == preds).all()

    def test_calibration_size(self):
        cw = ConformalWrapper(calibration_window=5)
        actuals = np.arange(10, dtype=float)
        predictions = np.arange(10, dtype=float) + 1
        cw.calibrate(actuals, predictions)
        assert cw.calibration_size == 5

    def test_interval_width(self):
        np.random.seed(42)
        actuals = np.random.normal(100, 10, 50)
        predictions = np.random.normal(100, 5, 50)
        cw = ConformalWrapper()
        cw.calibrate(actuals, predictions)
        assert cw.interval_width > 0


class TestBaselineForecaster:
    def test_fit_predict(self):
        from praxis.forecast.baseline import BaselineForecaster
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": [100 + d.weekday() * 10 for d in dates],
            "daytype": [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()] for d in dates],
        })

        f = BaselineForecaster(method="daytype_avg")
        f.fit(data)

        future = pd.DataFrame({
            "date": pd.date_range("2025-01-01", "2025-01-07"),
            "daytype": ["Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue"],
        })
        result = f.predict(future)
        assert "forecast" in result.columns
        assert len(result) == 7
        # Mon forecast should be ~100, Sun should be ~160
        assert result[result["daytype"] == "Mon"]["forecast"].values[0] < result[result["daytype"] == "Sun"]["forecast"].values[0]


class TestEnsembleRouter:
    def test_equal_blend(self):
        from praxis.forecast.ensemble import EnsembleRouter

        class DummyForecaster:
            def __init__(self, val):
                self.val = val
            def predict(self, data):
                return np.full(len(data), self.val)

        router = EnsembleRouter(weight_method="equal")
        router.register("low", DummyForecaster(80))
        router.register("high", DummyForecaster(120))

        data = pd.DataFrame({"x": [1, 2, 3]})
        result = router.predict(data)
        assert "forecast" in result.columns
        assert abs(result["forecast"].mean() - 100) < 1

    def test_weighted_blend(self):
        from praxis.forecast.ensemble import EnsembleRouter

        class DummyForecaster:
            def __init__(self, val):
                self.val = val
            def predict(self, data):
                return np.full(len(data), self.val)

        router = EnsembleRouter(weight_method="backtest_score")
        router.register("good", DummyForecaster(100))
        router.register("bad", DummyForecaster(200))
        router.set_weights_from_backtest({"good": 0.05, "bad": 0.20})

        data = pd.DataFrame({"x": [1]})
        result = router.predict(data)
        # Good forecaster should dominate
        assert result["forecast"].values[0] < 150
