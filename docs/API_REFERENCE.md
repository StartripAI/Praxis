# API Reference

## Calendar

### `CalendarEngine`

```python
from praxis.calendar import CalendarEngine

engine = CalendarEngine(
    country="CN",          # ISO country code
    year=2026,             # Reference year
    school_vacations=[     # Optional vacation periods
        ("2026-01-15", "2026-02-15"),
        ("2026-07-01", "2026-08-31"),
    ],
)

# Build date-level features
df = engine.build_date_features(start_date, end_date)
# → date, year, month, day, weekday, daytype, is_holiday, holiday_name, is_vacation, is_workday

# Build DM comparable day mapping
dm = engine.build_dm_mapping(target_year, target_month, reference_year=None)
# → date, comparable_date, daytype, comparable_daytype, match_quality
```

### `CalendarQA`

```python
from praxis.calendar.qa import CalendarQA

qa_result = CalendarQA.check_mapping(dm_mapping)  # → DataFrame with qa_status, qa_issues
summary = CalendarQA.summary(qa_result)            # → dict with pass_rate, issues_breakdown
```

### `periods`

```python
from praxis.calendar.periods import month_range, mtd_progress, build_period_index

first, last = month_range(2026, 3)
progress = mtd_progress(2026, 3, as_of=date(2026, 3, 15))
index = build_period_index(start_year=2026, start_month=1, n_months=12)
```

---

## Analysis

### `GrowthDecomposer`

```python
from praxis.analysis.growth import GrowthDecomposer

gd = GrowthDecomposer(yoy_weight=0.7, mom_weight=0.3, shrinkage="bayesian")
result = gd.decompose(current, yoy_reference, mom_reference, coverage=1.0)
# → GrowthResult(yoy_growth, mom_growth, blended_growth, shrinkage_applied, ...)

batch = gd.decompose_dataframe(df, value_col="value", entity_col="entity", period_col="period")
```

### `EventLearner`

```python
from praxis.analysis.event_learner import EventLearner

learner = EventLearner(min_observations=2, window_days=7)
effects = learner.learn(data, date_col="date", value_col="value", event_col="event_name")
adjusted = learner.apply(baseline=1000, event_name="spring_festival")
summary = learner.summary()
```

### `DOWLearner`

```python
from praxis.analysis.dow_learner import DOWLearner

learner = DOWLearner(min_weeks=4)
shares = learner.learn(data)           # → dict[entity → 7-element array]
daily = learner.distribute_weekly(700) # → array of 7 daily values
```

### `EntityTier`

```python
from praxis.analysis.entity_tier import EntityTier

tier = EntityTier(method="quantile", n_tiers=3)
result = tier.classify(data, entity_col="entity", value_col="value")
bounds = tier.auto_detect_bounds(data, value_col="value")
```

---

## Forecast

### `BaselineForecaster`

```python
from praxis.forecast.baseline import BaselineForecaster

f = BaselineForecaster(method="daytype_avg", lookback_days=365)
f.fit(data, daytype_col="daytype")
result = f.predict(future_dates)               # → DataFrame with 'forecast'
```

### `LGBMForecaster`

```python
from praxis.forecast.lgbm import LGBMForecaster

f = LGBMForecaster(n_estimators=200, learning_rate=0.05)
f.fit(data, target_col="value")
preds = f.predict(test_data)                   # → numpy array
recursive = f.predict_recursive(seed, horizon=31)
importance = f.feature_importance               # → DataFrame
```

### `ConformalWrapper`

```python
from praxis.forecast.conformal import ConformalWrapper

cw = ConformalWrapper(quantiles=[0.10, 0.50, 0.90], calibration_window=90)
cw.calibrate(actuals, predictions)
intervals = cw.predict_intervals(point_forecasts)  # → DataFrame with p10, p50, p90
```

### `EnsembleRouter`

```python
from praxis.forecast.ensemble import EnsembleRouter

router = EnsembleRouter(weight_method="backtest_score")
router.register("baseline", baseline_forecaster)
router.register("lgbm", lgbm_forecaster)
router.set_weights_from_backtest({"baseline": 0.12, "lgbm": 0.08})
result = router.predict(data)                  # → DataFrame with blended 'forecast'
```

### Feature Engineering

```python
from praxis.forecast.features import build_all_features

df = build_all_features(data, value_col="value", date_col="date")
# Adds: lag_7/14/28/364, rolling_mean/std_7/14/28, weekday, month, quarter, ...
```

---

## Backtest

### `BacktestRunner`

```python
from praxis.backtest.runner import BacktestRunner

runner = BacktestRunner(n_origins=6, horizon_days=31, gate={"max_bias": 0.15})
report = runner.run(data, fit_predict_fn)
print(report.summary())                        # → dict with verdict, pass_rate
print(report.to_dataframe())                   # → per-origin results
```

### `scoring`

```python
from praxis.backtest.scoring import score

sc = score(actuals, forecasts, lower=p10, upper=p90, gate={"max_bias": 0.15})
# → ScoreCard(bias, abs_bias, wape, mape, rmse, coverage_90, passed)
```

---

## Data

### `DuckDBStore`

```python
from praxis.data.duckdb import DuckDBStore

with DuckDBStore("analytics.duckdb") as store:
    store.load_csv("metrics", "data.csv")
    store.load_dataframe("calendar", cal_df)
    result = store.execute("SELECT * FROM metrics WHERE date > '2025-01-01'")
    tables = store.tables()
```
