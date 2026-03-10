"""Tests for praxis.analysis.growth module."""

import pytest
from praxis.analysis.growth import GrowthDecomposer, GrowthResult
import pandas as pd


class TestGrowthDecomposer:
    def test_basic_decompose(self):
        gd = GrowthDecomposer(yoy_weight=0.7, mom_weight=0.3)
        result = gd.decompose(
            current=110,
            yoy_reference=100,
            mom_reference=105,
        )
        assert isinstance(result, GrowthResult)
        assert abs(result.yoy_growth - 0.10) < 0.001
        assert abs(result.mom_growth - 0.0476) < 0.01

    def test_zero_reference(self):
        gd = GrowthDecomposer()
        result = gd.decompose(current=100, yoy_reference=0, mom_reference=100)
        assert result.yoy_growth == 0.0

    def test_shrinkage_applied(self):
        gd = GrowthDecomposer(shrinkage="bayesian")
        result = gd.decompose(
            current=110,
            yoy_reference=100,
            mom_reference=105,
            coverage=0.3,
            global_growth=0.05,
        )
        assert result.shrinkage_applied is True
        # Blended should be pulled toward global_growth
        assert result.blended_growth != result.raw_blended

    def test_no_shrinkage(self):
        gd = GrowthDecomposer(shrinkage="none")
        result = gd.decompose(
            current=110,
            yoy_reference=100,
            mom_reference=105,
            coverage=0.3,
        )
        assert result.shrinkage_applied is False
        assert result.blended_growth == result.raw_blended

    def test_full_coverage_no_shrinkage(self):
        gd = GrowthDecomposer(shrinkage="bayesian")
        result = gd.decompose(
            current=110,
            yoy_reference=100,
            mom_reference=105,
            coverage=1.0,
        )
        assert result.shrinkage_applied is False

    def test_weights_sum_validation(self):
        with pytest.raises(AssertionError):
            GrowthDecomposer(yoy_weight=0.5, mom_weight=0.3)

    def test_dataframe_decompose(self):
        data = pd.DataFrame({
            "entity": ["A"] * 3 + ["B"] * 3,
            "period": ["2025-01", "2025-02", "2025-03"] * 2,
            "value": [100, 110, 115, 200, 210, 220],
        })
        gd = GrowthDecomposer()
        result = gd.decompose_dataframe(data)
        assert len(result) == 6
        assert "yoy_growth" in result.columns
        assert "blended_growth" in result.columns
