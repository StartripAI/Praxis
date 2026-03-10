# Praxis

> The last mile of business forecasting — calendar alignment, growth decomposition, conformal uncertainty, and backtest governance. One engine, all industries.

[![Security Check](https://github.com/StartripAI/Praxis/actions/workflows/security.yml/badge.svg)](https://github.com/StartripAI/Praxis/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## 5 Seconds to Start

```bash
pip install praxis-engine
```

## 1 Minute to Learn

```python
from praxis.calendar import CalendarEngine
from praxis.analysis.growth import GrowthDecomposer
from praxis.forecast.conformal import ConformalWrapper
from praxis.backtest.scoring import score

# 1. Which day last year is comparable to March 15, 2026?
engine = CalendarEngine(country="CN", year=2026)
dm = engine.build_dm_mapping(target_year=2026, target_month=3)

# 2. How much did we grow, and why?
growth = GrowthDecomposer().decompose(
    current=115000, yoy_reference=100000, mom_reference=110000
)

# 3. What's the forecast range? (not a single number)
cw = ConformalWrapper(quantiles=[0.10, 0.50, 0.90])
cw.calibrate(actuals, predictions)
intervals = cw.predict_intervals(forecast)

# 4. Should we trust this model?
sc = score(actuals, predictions, gate={"max_bias": 0.15})
print(f"Verdict: {'PASS' if sc.passed else 'FAIL'}")
```

## What Just Happened?

1. **Calendar** — Praxis mapped March 2026 dates to their comparable "like days" from 2025, aligning weekdays, holidays, and school vacations. Same concept as NRF 4-5-4 retail calendar.
2. **Growth** — Decomposed the YoY and MoM components with Bayesian shrinkage, so low-data entities don't produce wild estimates.
3. **Conformal** — Instead of "the forecast is 115K", you now say "P10–P90 range is 108K–122K" — managers understand risk language.
4. **Backtest** — The model proved itself against a 15% bias gate before anyone trusted it.

---

## User-first

For operators, analysts, and finance teams:

- **your YoY comparisons actually compare like to like** — no more "but March 1 was a Saturday last year"
- **your forecasts come with calibrated ranges** — P10/P50/P90 by default
- **your models must pass a gate before going live** — bias ≤ 15%, WAPE ≤ 20%, or it fails
- **your growth is explained, not just reported** — "12% blended = 15% YoY × 0.7 + 5% MoM × 0.3"

## Developer-first

For builders and integrators:

- **Thin wrapper pattern** — Praxis only builds the business abstraction layer; forecasting, intervals, and features are delegated to StatsForecast, Darts, MAPIE, tsfresh
- **Protocol-based backends** — swap any forecaster, any conformal method, any calendar
- **Zero hardcoded logic** — all event effects, DOW shares, and entity tiers are learned from your data
- **DuckDB-native** — embedded analytics, replacement scans, star schema out of the box

## Product Boundaries

This engine intentionally does not:
- replace your BI dashboard (use FineBI, Metabase, Superset)
- replace your planning platform (use Anaplan, Pigment)
- train foundation models or do AutoML research
- provide a GUI (you write Python; if you need a GUI, you don't need Praxis)

## North Star

- Calendar alignment is infrastructure, not a feature.
- Uncertainty is the default output, not an add-on.
- Every model earns trust through backtesting, or it doesn't ship.
- Event effects are learned from data, never hardcoded.

---

## Why Praxis?

Most forecasting tools stop at the model. Praxis covers the **last mile** that no one else does:

| Gap | Who has it | Praxis |
|---|---|---|
| Calendar alignment / comparable trading days | ❌ Darts, Prophet, StatsForecast | ✅ DM mapping + QA |
| Growth decomposition (YoY/MoM + shrinkage) | ❌ All | ✅ Built-in |
| Conformal P10/P50/P90 as default | ⚠️ Partial in Darts/StatsForecast | ✅ Default output |
| Event effect auto-learning | ⚠️ Greykite (manual) | ✅ Data-driven |
| Backtest pass/fail gates | ❌ All (report only) | ✅ PASS/FAIL verdict |

## Applicable Industries

Retail · E-commerce · SaaS · Logistics · F&B · Finance · Education · Local services — any business with daily/weekly/monthly metric data.

---

## Install

```bash
# Core
pip install praxis-engine

# Full (+ Darts, tsfresh, MAPIE, Greykite)
pip install praxis-engine[full]

# From source
git clone https://github.com/StartripAI/Praxis.git
cd Praxis && pip install -e ".[dev]"
```

## Examples

```bash
python examples/retail_sales.py       # Retail daily sales forecasting
python examples/saas_mrr.py           # SaaS MRR/ARR forecasting
python examples/logistics_demand.py   # Logistics demand planning
python examples/budget_rolling.py     # Financial rolling budget
```

## Architecture

```
Input → ┌──────────────────────────────────────────┐
        │    Praxis — Business Abstraction (you)   │
        │  DM Mapping │ Growth │ Gate │ Report     │
        ├──────────────────────────────────────────┤
        │    Thin Wrappers (Praxis adapters)        │
        │  forecast/ │ conformal/ │ features/      │
        ├──────────────────────────────────────────┤
        │    Open-source Backends (pip install)     │
        │  StatsForecast│Darts│MAPIE│tsfresh│...   │
        ├──────────────────────────────────────────┤
        │    Data Layer                             │
        │  DuckDB │ BigQuery/Snowflake MCP │ CSV   │
        └──────────────────────────────────────────┘
```

## Links

- [Quick Start](docs/QUICKSTART.md)
- [Methodology — DM / Trading-Day Alignment](docs/METHODOLOGY.md)
- [Backtest Guide](docs/BACKTEST_GUIDE.md)
- [Event Effect Guide](docs/EVENT_EFFECT_GUIDE.md)
- [Outlier Patterns](docs/OUTLIER_PATTERNS.md)
- [API Reference](docs/API_REFERENCE.md)

## License

Apache-2.0
