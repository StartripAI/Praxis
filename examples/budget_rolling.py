"""
Example: Financial Rolling Budget Forecast with Praxis

Demonstrates monthly budget forecasting with growth decomposition:
1. Synthetic departmental budget data
2. YoY/MoM growth decomposition with Bayesian shrinkage
3. Entity tiering (by department performance)
4. Conformal intervals for budget ranges
"""

import numpy as np
import pandas as pd

from praxis.analysis.growth import GrowthDecomposer
from praxis.analysis.entity_tier import EntityTier
from praxis.forecast.conformal import ConformalWrapper


def generate_budget_data():
    """Generate synthetic departmental budget data."""
    np.random.seed(42)
    departments = ["Engineering", "Marketing", "Sales", "Support", "Operations"]
    months = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    records = []

    base_budgets = {"Engineering": 500000, "Marketing": 300000, "Sales": 400000,
                    "Support": 150000, "Operations": 200000}

    for dept in departments:
        base = base_budgets[dept]
        for i, month in enumerate(months):
            growth = 1.005 ** i
            seasonal = 1 + 0.05 * np.sin(2 * np.pi * month.month / 12)
            noise = np.random.normal(0, 0.03)
            val = base * growth * seasonal * (1 + noise)
            records.append({
                "date": month, "entity": dept,
                "period": month.strftime("%Y-%m"), "value": val,
            })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Praxis Example: Financial Rolling Budget")
    print("=" * 60)

    data = generate_budget_data()
    print(f"\n📊 {len(data)} records, {data['entity'].nunique()} departments")

    # Entity tiering
    tier = EntityTier(n_tiers=3, tier_labels=["High", "Medium", "Low"])
    tier_result = tier.classify(data)
    print(f"\n🏢 Department Tiers:")
    for _, row in tier_result.iterrows():
        print(f"   {row['entity']}: {row['tier']} (avg={row['value']:,.0f})")

    # Growth decomposition per department
    gd = GrowthDecomposer(yoy_weight=0.6, mom_weight=0.4, shrinkage="bayesian")
    print(f"\n📈 Growth Decomposition (latest month):")
    for dept in data["entity"].unique():
        dept_data = data[data["entity"] == dept].sort_values("date")
        vals = dept_data["value"].values
        if len(vals) >= 13:
            result = gd.decompose(
                current=vals[-1],
                yoy_reference=vals[-12],
                mom_reference=vals[-2],
                coverage=min(len(vals) / 24, 1.0),
            )
            shrink_tag = " [shrunk]" if result.shrinkage_applied else ""
            print(f"   {dept}: YoY={result.yoy_growth:+.1%}, MoM={result.mom_growth:+.1%}, "
                  f"Blended={result.blended_growth:+.1%}{shrink_tag}")

    # Conformal budget ranges
    total_monthly = data.groupby("date")["value"].sum().values
    preds = np.roll(total_monthly, 12)
    preds[:12] = total_monthly[:12]
    cw = ConformalWrapper(quantiles=[0.10, 0.50, 0.90])
    cw.calibrate(total_monthly[12:], preds[12:])
    next_forecast = np.array([total_monthly[-1] * 1.01])
    intervals = cw.predict_intervals(next_forecast)
    print(f"\n📐 Next Month Total Budget:")
    print(f"   Conservative (P10): ${intervals['p10'].values[0]:,.0f}")
    print(f"   Expected    (P50): ${intervals['p50'].values[0]:,.0f}")
    print(f"   Aggressive  (P90): ${intervals['p90'].values[0]:,.0f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
