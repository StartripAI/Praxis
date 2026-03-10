# Praxis — Operating Intelligence Engine

> **Calendar-aligned forecasting, growth decomposition, conformal uncertainty, and backtest governance in one package.**

[![Security Check](https://github.com/StartripAI/Praxis/actions/workflows/security.yml/badge.svg)](https://github.com/StartripAI/Praxis/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What is Praxis?

Praxis is an open-source Python engine that turns raw business data into **reliable, auditable forecasts** — with built-in calendar alignment, growth decomposition, uncertainty quantification, and backtest governance.

Most forecasting tools stop at the model. Praxis covers the **last mile**:

| Capability | What it does | Why it matters |
|---|---|---|
| 🗓️ **Trading-Day Alignment** | Maps each day to its comparable "like day" last year (DM mapping) | Makes YoY comparisons actually comparable — equivalent to NRF 4-5-4 retail calendar |
| 📈 **Growth Decomposition** | Decomposes growth into YoY + MoM with Bayesian shrinkage | Explains *why* numbers changed, not just *that* they changed |
| 🎯 **Conformal Uncertainty** | Outputs calibrated P10/P50/P90 intervals (CQR) | Turns "a number" into a risk-aware range — managers understand this |
| 🔬 **Event Effect Learning** | Auto-detects holiday/promo/weather impacts from data | Replaces fragile hardcoded coefficients |
| ✅ **Backtest Governance** | Walk-forward backtesting with pass/fail gates | Creates a "verification culture" — no more gut-feel targets |
| 🔌 **Multi-Source Data** | DuckDB + BigQuery/Snowflake MCP + CSV/Excel | One engine for all your data, no more Excel hell |

## Install

```bash
pip install praxis-engine              # Core
pip install praxis-engine[full]        # + Darts, tsfresh, MAPIE, Greykite
```

## Quick Start

```python
from praxis.calendar import CalendarEngine
from praxis.forecast import BaselineForecaster
from praxis.backtest import BacktestRunner

# 1. Build calendar with trading-day alignment
calendar = CalendarEngine(country="CN", year=2026)
dm_map = calendar.build_dm_mapping(target_month=3)

# 2. Forecast with conformal intervals
forecaster = BaselineForecaster(method="daytype_avg")
forecast = forecaster.predict(data, horizon=31, conformal=True)

# 3. Validate with walk-forward backtest
runner = BacktestRunner(n_origins=6, gate={"max_bias": 0.15})
report = runner.run(forecaster, data)
print(report.summary())
```

## Architecture — Thin Wrapper Pattern

Praxis builds **only the business abstraction layer**. All heavy lifting is delegated to battle-tested open-source backends:

| Layer | What | Built by |
|---|---|---|
| Business Logic | DM mapping, growth decomp, backtest gate, report output | **Praxis** (this repo) |
| Adapters | Forecast wrappers, conformal wrappers, feature pipelines | **Praxis** (thin) |
| Backends | StatsForecast, Darts, MAPIE, tsfresh, DuckDB, chinese-calendar | **Open source** |

## Applicable Industries

Retail · E-commerce · SaaS · Logistics · F&B · Finance · Education · Local services — any business with daily/weekly/monthly metric data.

## License

Apache License 2.0
