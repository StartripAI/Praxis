# Event Effect Guide

## Overview

The `EventLearner` module automatically detects and quantifies the impact of events (holidays, promotions, weather, etc.) on business metrics — replacing fragile hardcoded coefficients.

## How It Works

### 1. Event Identification

Events are identified from your data's `event_name` column. This can come from:
- Calendar holidays (`CalendarEngine`)
- Business events (promotions, campaigns)
- External events (weather, competitor actions)

### 2. Effect Estimation

For each event, the learner compares the event-day value to a control window:

```
effect = (event_value / control_mean) - 1.0
```

Where `control_mean` is the average of non-event days within ±`window_days`.

### 3. Robust Aggregation

When an event occurs multiple times (e.g., annual holidays), the **median** effect across all occurrences is used — robust to outliers.

## Usage

```python
from praxis.analysis.event_learner import EventLearner

learner = EventLearner(min_observations=2, window_days=7)

# Learn from historical data
effects = learner.learn(data)

# Check what was learned
print(learner.summary())

# Apply to a forecast
adjusted = learner.apply(baseline=1000, event_name="spring_festival")
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `min_observations` | 2 | Minimum times an event must occur to learn |
| `window_days` | 7 | Control window size (days before/after) |

## Output: EventEffect

| Field | Type | Description |
|---|---|---|
| `event_name` | str | Identifier |
| `effect_pct` | float | Median % change (e.g., +0.30 = +30%) |
| `confidence` | float | 0-1, based on sample size |
| `n_observations` | int | Historical occurrences used |
| `method` | str | "before_after" or "causal_impact" |

## Best Practices

1. **Label events consistently** — Same event should always have the same `event_name`
2. **2+ observations minimum** — A single occurrence is not reliable
3. **Separate multi-day events** — Label each day (e.g., "golden_week_day1", "golden_week_day2") or use a single label for the entire window
4. **Review learned effects** — Use `.summary()` to sanity-check before applying
5. **Re-learn periodically** — Event effects can change over time (COVID, new competitors)
