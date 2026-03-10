# Backtest Guide — Walk-Forward Validation

## Why Backtest?

> "A forecast without a backtest is just a guess with extra steps."

Backtesting creates a **verification culture**: every model must prove its accuracy on historical data before being trusted for real decisions.

## Walk-Forward Design

```
Data Timeline:
|-------- Train --------|-- Test --|
                        Origin 1

|----------- Train ----------|-- Test --|
                             Origin 2

|--------------- Train -----------|-- Test --|
                                 Origin 3
```

**Key principle**: at each origin, only use data available *before* the origin date. Never let future data leak into training.

## Configuration

```python
from praxis.backtest.runner import BacktestRunner

runner = BacktestRunner(
    n_origins=6,           # 6 rolling forecast origins
    horizon_days=31,       # Forecast 31 days ahead
    step_days=30,          # 30 days between origins
    gate={
        "max_bias": 0.15,  # |bias| must be ≤ 15%
        "max_wape": 0.20,  # WAPE must be ≤ 20%
    },
)
```

## Scoring Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **Bias** | Σ(f-a) / Σa | Systematic over/under-forecasting |
| **WAPE** | Σ\|f-a\| / Σa | Overall accuracy, volume-weighted |
| **MAPE** | mean(\|f-a\|/a) | Per-observation accuracy |
| **RMSE** | √mean((f-a)²) | Penalizes large errors |
| **Coverage** | % of actuals within [P10, P90] | Interval calibration |

## Pass/Fail Gates

Gates enforce minimum quality standards:

```python
gate = {
    "max_bias": 0.15,   # ≤ 15% total bias
    "max_wape": 0.20,   # ≤ 20% weighted error
}
```

A forecast origin **passes** only if ALL gates are met.

## Reading the Report

```python
report = runner.run(data, fit_predict_fn)
print(report.summary())
# {
#   "n_origins": 6,
#   "n_pass": 5,
#   "n_fail": 1,
#   "pass_rate": 83.3,
#   "avg_bias": 0.079,
#   "verdict": "FAIL"
# }
```

**Verdict**:
- `PASS` — ALL origins passed ALL gates
- `FAIL` — At least one origin failed at least one gate
- `NO_DATA` — Insufficient data for backtesting

## Best Practices

1. **Use ≥ 6 origins** — Less is statistically unreliable
2. **Don't cherry-pick** — Report all origins, not just the good ones
3. **Match the horizon** — Backtest horizon should equal production horizon
4. **Seasonal coverage** — Ensure origins span different seasons/events
5. **Re-run after changes** — Every model update needs a fresh backtest
