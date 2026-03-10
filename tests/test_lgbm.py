"""Tests for praxis.forecast.lgbm module."""

import numpy as np
import pandas as pd
import pytest
from praxis.forecast.features import (
    build_lag_features,
    build_rolling_features,
    build_calendar_features,
    build_all_features,
)


class TestFeatures:
    def test_lag_features(self):
        dates = pd.date_range("2024-01-01", periods=400, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.arange(400, dtype=float)})
        result = build_lag_features(df, lags=[7, 14])
        assert "lag_7" in result.columns
        assert "lag_14" in result.columns
        assert result["lag_7"].iloc[7] == 0.0  # value at day 0

    def test_rolling_features(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.ones(30) * 100})
        result = build_rolling_features(df, windows=[7])
        assert "rolling_mean_7" in result.columns
        assert abs(result["rolling_mean_7"].iloc[-1] - 100) < 0.01

    def test_calendar_features(self):
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        df = pd.DataFrame({"date": dates, "value": range(7)})
        result = build_calendar_features(df)
        assert "weekday" in result.columns
        assert "is_weekend" in result.columns
        assert "quarter" in result.columns

    def test_build_all(self):
        dates = pd.date_range("2024-01-01", periods=400, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.normal(100, 10, 400)})
        result = build_all_features(df)
        assert "lag_7" in result.columns
        assert "rolling_mean_7" in result.columns
        assert "weekday" in result.columns


class TestLGBMForecaster:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=400, freq="D")
        values = 100 + np.random.normal(0, 5, 400)
        df = pd.DataFrame({"date": dates, "value": values})
        return build_all_features(df).dropna()

    def test_fit_and_predict(self, sample_data):
        pytest.importorskip("lightgbm")
        from praxis.forecast.lgbm import LGBMForecaster

        f = LGBMForecaster(n_estimators=50)
        f.fit(sample_data, target_col="value")
        preds = f.predict(sample_data.head(10))
        assert len(preds) == 10
        assert all(np.isfinite(preds))

    def test_feature_importance(self, sample_data):
        pytest.importorskip("lightgbm")
        from praxis.forecast.lgbm import LGBMForecaster

        f = LGBMForecaster(n_estimators=50)
        f.fit(sample_data, target_col="value")
        fi = f.feature_importance
        assert len(fi) > 0
        assert "feature" in fi.columns
        assert "importance" in fi.columns
