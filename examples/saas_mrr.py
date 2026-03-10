"""
Example: SaaS MRR/ARR Forecasting with Praxis

Demonstrates monthly SaaS revenue forecasting:
1. Generate synthetic MRR data with churn & expansion
2. Growth decomposition (MoM + YoY)
3. Baseline forecast with conformal intervals
4. Backtest validation
"""

import numpy as np
import pandas as pd

from praxis.analysis.growth import GrowthDecomposer
from praxis.forecast.conformal import ConformalWrapper
from praxis.backtest.scoring import score


def generate_saas_data(n_months=24, n_accounts=50):
    """Generate synthetic SaaS MRR data."""
    np.random.seed(42)
    records = []
    months = pd.date_range("2024-01-01", periods=n_months, freq="MS")

    for acct in range(n_accounts):
        base_mrr = np.random.uniform(500, 5000)
        for i, month in enumerate(months):
            # Natural growth + churn risk
            growth_rate = 1 + np.random.normal(0.02, 0.03)
            churn = 0 if np.random.random() > 0.03 else -base_mrr
            mrr = base_mrr * growth_rate ** i + churn
            records.append({
                "date": month,
                "entity": f"acct_{acct:03d}",
                "metric": "mrr",
                "value": max(0, mrr),
            })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Praxis Example: SaaS MRR Forecasting")
    print("=" * 60)

    data = generate_saas_data()
    monthly = data.groupby("date")["value"].sum().reset_index()
    print(f"\n📊 {len(data):,} records, {data['entity'].nunique()} accounts, {len(monthly)} months")
    print(f"   Latest MRR: ${monthly['value'].iloc[-1]:,.0f}")
    print(f"   12-month growth: {(monthly['value'].iloc[-1] / monthly['value'].iloc[-12] - 1):.1%}")

    # Growth decomposition
    gd = GrowthDecomposer(yoy_weight=0.5, mom_weight=0.5)
    result = gd.decompose(
        current=monthly["value"].iloc[-1],
        yoy_reference=monthly["value"].iloc[-12],
        mom_reference=monthly["value"].iloc[-2],
    )
    print(f"\n📈 Growth Decomposition:")
    print(f"   YoY: {result.yoy_growth:.1%}")
    print(f"   MoM: {result.mom_growth:.1%}")
    print(f"   Blended: {result.blended_growth:.1%}")

    # Conformal forecast
    actuals = monthly["value"].values
    simple_preds = np.roll(actuals, 1)  # lag-1 naive
    simple_preds[0] = actuals[0]
    cw = ConformalWrapper()
    cw.calibrate(actuals[1:], simple_preds[1:])
    next_pred = np.array([actuals[-1] * (1 + result.mom_growth)])
    intervals = cw.predict_intervals(next_pred)
    print(f"\n📐 Next Month Forecast:")
    print(f"   P10: ${intervals['p10'].values[0]:,.0f}")
    print(f"   P50: ${intervals['p50'].values[0]:,.0f}")
    print(f"   P90: ${intervals['p90'].values[0]:,.0f}")

    # Backtest
    sc = score(actuals[1:], simple_preds[1:], gate={"max_bias": 0.15})
    print(f"\n✅ Naive baseline: bias={sc.bias:.1%}, WAPE={sc.wape:.1%}, passed={sc.passed}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
