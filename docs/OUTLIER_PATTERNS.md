# Outlier Patterns — 5 Common Anomaly Types

This guide documents the 5 most common anomaly patterns in daily business data and how Praxis handles each.

## 1. Holiday Spike / Dip

**Pattern**: Sharp increase or decrease centered on a holiday/event.

**Example**: 618 shopping festival (+50%), National Day closures (-80%)

**Detection**: Calendar-based — events flagged by `CalendarEngine` and quantified by `EventLearner`.

**Handling**: `EventLearner.apply()` adjusts baseline by learned effect percentage.

## 2. Post-Event Rebound / Siphon

**Pattern**: After a spike, demand dips below baseline (cannibalization), or after a dip, demand rebounds above baseline (pent-up).

**Example**: Post-Spring Festival foot traffic recovery over 2-3 weeks.

**Detection**: Analyze residuals in the window *after* known events. If systematic, the `EventLearner` captures multi-day effects when configured with appropriate window sizes.

**Handling**: Extend event window in `EventLearner(window_days=14)` to capture post-event patterns.

## 3. Day-of-Week (DOW) Anomaly

**Pattern**: Specific weekdays consistently differ from average (e.g., weekends higher for retail, lower for B2B).

**Example**: Saturday revenue +30% vs weekday average.

**Detection**: `DOWLearner` computes per-weekday distributional shares.

**Handling**: Baseline forecaster uses `daytype_avg` which naturally incorporates DOW patterns.

## 4. Structural Break / Level Shift

**Pattern**: Permanent change in mean level — new store opening, pricing change, market exit.

**Example**: Store renovation closes for 2 weeks, then reopens at different demand level.

**Detection**: Monitor rolling mean features (`rolling_mean_28`) for sudden shifts. In backtesting, origins straddling a break will show elevated WAPE.

**Handling**: Use `lookback_days` parameter in `BaselineForecaster` to limit training data to post-break period.

## 5. Seasonal Ramp / Decay

**Pattern**: Gradual increase or decrease over weeks (not sudden). Back-to-school ramp, winter seasonal decline.

**Example**: Foot traffic gradually increasing Sept → Oct as schools reopen.

**Detection**: Growth decomposition (`GrowthDecomposer`) shows persistent MoM growth.

**Handling**: MoM component in blended growth captures trend; LGBM lag features capture momentum.

## Summary Table

| Pattern | Duration | Praxis Module | Key Parameter |
|---|---|---|---|
| Holiday spike/dip | 1-3 days | `EventLearner` | `window_days` |
| Post-event siphon | 1-3 weeks | `EventLearner` | `window_days=14` |
| DOW anomaly | Recurring weekly | `DOWLearner` | `min_weeks` |
| Structural break | Permanent | `BaselineForecaster` | `lookback_days` |
| Seasonal ramp | 4-12 weeks | `GrowthDecomposer` | `yoy_weight`/`mom_weight` |
