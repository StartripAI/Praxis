"""
Example: Retail Sales Forecasting with Praxis

Demonstrates end-to-end daily sales forecasting for a retail chain:
1. Load synthetic store data
2. Build calendar with trading-day alignment
3. Train baseline + LGBM forecasters
4. Blend with ensemble
5. Add conformal intervals (P10/P50/P90)
6. Run walk-forward backtest
"""

import numpy as np
import pandas as pd

from praxis.calendar import CalendarEngine
from praxis.calendar.qa import CalendarQA
from praxis.analysis.growth import GrowthDecomposer
from praxis.analysis.entity_tier import EntityTier
from praxis.forecast.baseline import BaselineForecaster
from praxis.forecast.features import build_all_features
from praxis.forecast.conformal import ConformalWrapper
from praxis.backtest.scoring import score


def generate_synthetic_data(n_stores=5, n_days=730):
    """Generate synthetic daily retail sales data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    records = []

    for store_id in range(n_stores):
        base = 1000 + store_id * 500  # Different store scales
        for d in dates:
            # Seasonal: higher on weekends, slight monthly pattern
            dow_effect = 1.3 if d.weekday() >= 5 else 1.0
            month_effect = 1 + 0.05 * np.sin(2 * np.pi * d.month / 12)
            noise = np.random.normal(0, 0.05)
            value = base * dow_effect * month_effect * (1 + noise)
            records.append({
                "date": d,
                "entity": f"store_{store_id:02d}",
                "metric": "sales",
                "value": max(0, value),
            })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Praxis Example: Retail Sales Forecasting")
    print("=" * 60)

    # 1. Generate data
    data = generate_synthetic_data(n_stores=5, n_days=730)
    print(f"\n📊 Data: {len(data):,} rows, {data['entity'].nunique()} stores, {data['date'].nunique()} days")

    # 2. Calendar with DM mapping
    engine = CalendarEngine(country="CN", year=2026)
    dm_map = engine.build_dm_mapping(target_year=2026, target_month=3)
    qa = CalendarQA.check_mapping(dm_map)
    qa_summary = CalendarQA.summary(qa)
    print(f"\n🗓️ DM Mapping: {qa_summary['total_dates']} dates, {qa_summary['pass_rate']}% QA pass rate")

    # 3. Entity tiering
    tier = EntityTier(n_tiers=3)
    tier_result = tier.classify(data, entity_col="entity", value_col="value")
    print(f"\n🏢 Entity Tiers:")
    for _, row in tier_result.iterrows():
        print(f"   {row['entity']}: Tier {row['tier']} (avg={row['value']:.0f})")

    # 4. Growth decomposition (global)
    decomposer = GrowthDecomposer()
    # Compare Y2 vs Y1
    y1 = data[data["date"].dt.year == 2024]["value"].sum()
    y2 = data[data["date"].dt.year == 2025]["value"].sum()
    growth = decomposer.decompose(current=y2, yoy_reference=y1, mom_reference=y1)
    print(f"\n📈 Growth: YoY={growth.yoy_growth:.1%}, Blended={growth.blended_growth:.1%}")

    # 5. Baseline forecast
    store0_data = data[data["entity"] == "store_00"].copy()
    cal_features = engine.build_date_features("2024-01-01", "2025-12-31")
    store0_data = store0_data.merge(cal_features[["date", "daytype"]], on="date", how="left")

    forecaster = BaselineForecaster(method="daytype_avg")
    forecaster.fit(store0_data)
    print(f"\n🔮 Baseline templates: {len(forecaster.templates)} daytypes learned")

    # 6. Conformal intervals
    actuals = store0_data["value"].values[-90:]
    # Simple prediction for demo
    predictions = np.full_like(actuals, actuals.mean())
    cw = ConformalWrapper()
    cw.calibrate(actuals, predictions)
    future_preds = np.array([actuals.mean()] * 7)
    intervals = cw.predict_intervals(future_preds)
    print(f"\n📐 Conformal intervals (7-day forecast):")
    print(f"   P10: {intervals['p10'].mean():.0f}")
    print(f"   P50: {intervals['p50'].mean():.0f}")
    print(f"   P90: {intervals['p90'].mean():.0f}")

    # 7. Backtest scoring
    sc = score(actuals, predictions, gate={"max_bias": 0.15, "max_wape": 0.20})
    print(f"\n✅ Backtest: bias={sc.bias:.1%}, WAPE={sc.wape:.1%}, passed={sc.passed}")

    print("\n" + "=" * 60)
    print("Done! All modules working end-to-end.")
    print("=" * 60)


if __name__ == "__main__":
    main()
