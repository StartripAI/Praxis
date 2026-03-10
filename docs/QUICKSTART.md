# Quick Start Guide

## Installation

```bash
pip install praxis-engine              # Core dependencies
pip install praxis-engine[full]        # All optional backends
```

**Or from source:**

```bash
git clone https://github.com/StartripAI/Praxis.git
cd Praxis
pip install -e ".[dev]"
```

## 5-Minute Demo

### 1. Build a Calendar with Trading-Day Alignment

```python
from praxis.calendar import CalendarEngine

engine = CalendarEngine(country="CN", year=2026)

# Date features for March 2026
features = engine.build_date_features("2026-03-01", "2026-03-31")
print(features[["date", "daytype", "is_holiday", "is_workday"]].head())

# DM mapping: find comparable dates from last year
dm = engine.build_dm_mapping(target_year=2026, target_month=3)
print(dm[["date", "comparable_date", "match_quality"]].head())
```

### 2. Check Mapping Quality

```python
from praxis.calendar.qa import CalendarQA

qa_result = CalendarQA.check_mapping(dm)
summary = CalendarQA.summary(qa_result)
print(f"QA Pass Rate: {summary['pass_rate']}%")
```

### 3. Growth Decomposition

```python
from praxis.analysis.growth import GrowthDecomposer

gd = GrowthDecomposer(yoy_weight=0.7, mom_weight=0.3)
result = gd.decompose(
    current=115000,
    yoy_reference=100000,
    mom_reference=110000,
)
print(f"YoY: {result.yoy_growth:.1%}")
print(f"MoM: {result.mom_growth:.1%}")
print(f"Blended: {result.blended_growth:.1%}")
```

### 4. Forecast with Conformal Intervals

```python
from praxis.forecast.conformal import ConformalWrapper
import numpy as np

cw = ConformalWrapper(quantiles=[0.10, 0.50, 0.90])
cw.calibrate(actuals=historical_actuals, predictions=historical_forecasts)

intervals = cw.predict_intervals(np.array([forecast_value]))
print(f"P10={intervals['p10'][0]:.0f}, P50={intervals['p50'][0]:.0f}, P90={intervals['p90'][0]:.0f}")
```

### 5. Backtest with Pass/Fail Gate

```python
from praxis.backtest.scoring import score

sc = score(actuals, forecasts, gate={"max_bias": 0.15, "max_wape": 0.20})
print(f"Bias: {sc.bias:.1%}, WAPE: {sc.wape:.1%}, Passed: {sc.passed}")
```

## Run Examples

```bash
python examples/retail_sales.py
python examples/saas_mrr.py
python examples/logistics_demand.py
python examples/budget_rolling.py
```

## Run Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Next Steps

- [METHODOLOGY.md](METHODOLOGY.md) — DM / Trading-Day Alignment theory
- [BACKTEST_GUIDE.md](BACKTEST_GUIDE.md) — Walk-forward backtest design
- [EVENT_EFFECT_GUIDE.md](EVENT_EFFECT_GUIDE.md) — Event effect learning
- [API_REFERENCE.md](API_REFERENCE.md) — Full API documentation
