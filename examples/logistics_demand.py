"""
Example: Logistics Demand Planning with Praxis

Demonstrates daily demand forecasting for logistics/supply chain:
1. Synthetic warehouse demand with DOW and seasonal patterns
2. DOW learner for weekday distribution
3. Event effect detection (holiday surges)
4. LGBM forecast + conformal intervals
"""

import numpy as np
import pandas as pd

from praxis.calendar import CalendarEngine
from praxis.analysis.dow_learner import DOWLearner
from praxis.analysis.event_learner import EventLearner
from praxis.forecast.features import build_all_features
from praxis.backtest.scoring import score


def generate_logistics_data(n_days=730):
    """Generate synthetic logistics demand data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    records = []

    for d in dates:
        # Base demand with DOW pattern (low weekend) and seasonal
        base = 500
        dow_mult = {0: 1.1, 1: 1.2, 2: 1.15, 3: 1.1, 4: 1.0, 5: 0.6, 6: 0.4}
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365)
        noise = np.random.normal(0, 0.08)

        # Holiday effect: -50% on holidays
        event = ""
        holiday_mult = 1.0
        if d.month == 1 and d.day == 1:
            holiday_mult = 0.5
            event = "new_year"
        elif d.month == 10 and 1 <= d.day <= 7:
            holiday_mult = 0.3
            event = "national_day"

        value = base * dow_mult[d.weekday()] * seasonal * holiday_mult * (1 + noise)
        records.append({"date": d, "value": max(0, value), "event_name": event})

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Praxis Example: Logistics Demand Planning")
    print("=" * 60)

    data = generate_logistics_data()
    print(f"\n📊 {len(data)} days of demand data")

    # DOW analysis
    learner = DOWLearner(min_weeks=4)
    shares = learner.learn(data)
    global_shares = learner.get_shares()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"\n📅 DOW Shares (learned):")
    for i, name in enumerate(dow_names):
        print(f"   {name}: {global_shares[i]:.1%}")

    # Event effect learning
    event_learner = EventLearner(min_observations=2, window_days=7)
    effects = event_learner.learn(data)
    print(f"\n🎯 Learned Events:")
    for name, effect in effects.items():
        print(f"   {name}: {effect.effect_pct:+.1%} (n={effect.n_observations})")

    # Distribute next week's demand
    weekly_total = data["value"].tail(7).sum()
    daily = learner.distribute_weekly(weekly_total)
    print(f"\n📦 Weekly demand distribution ({weekly_total:.0f} total):")
    for i, name in enumerate(dow_names):
        print(f"   {name}: {daily[i]:.0f}")

    # Quick backtest
    recent = data["value"].tail(60).values
    naive = np.full_like(recent, data["value"].mean())
    sc = score(recent, naive, gate={"max_bias": 0.15})
    print(f"\n✅ Naive baseline: bias={sc.bias:.1%}, WAPE={sc.wape:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
