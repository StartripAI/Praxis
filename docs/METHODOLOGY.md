# DM / Trading-Day Alignment — Methodology

## The Problem

Year-over-year (YoY) comparisons are misleading when the same calendar date has different characteristics across years:

| Issue | Example |
|---|---|
| **Weekday shift** | March 1, 2025 = Saturday; March 1, 2026 = Sunday |
| **Holiday drift** | Chinese New Year: 2025 Jan 29, 2026 Feb 17 |
| **Calendar composition** | February has 28 or 29 days; NRF 4-5-4 splits vary |

**Result**: raw YoY comparisons can show ±30% bias purely from calendar effects.

## Industry Equivalents

Praxis's DM mapping is not a novel invention — it formalizes a well-established industry practice:

| System | Organization | Core Idea |
|---|---|---|
| **NRF 4-5-4 Calendar** | US National Retail Federation | Split fiscal year into 4-5-4 week months to guarantee same weekday mix |
| **Trading Day Adjustment** | US Census X-13ARIMA-SEATS | Model the number of each weekday per month as an explicit regressor |
| **Comparable Store Sales** | Retail industry standard | Report same-store sales on matched periods |

## How DM Mapping Works

### Step 1: Daytype Classification

Each date is classified into a **daytype** — a composite of:

```
daytype = weekday + holiday_status + vacation_status
```

Examples:
- `Mon` — regular Monday
- `Sat_hol:spring_festival` — Saturday during Spring Festival
- `Wed_vac` — Wednesday during school vacation

### Step 2: Greedy Matching

For each date in the target month, find the best matching date in the reference (prior year) month:

| Score Component | Points |
|---|---|
| Same weekday | +2 |
| Same holiday status | +1 |
| Same vacation status | +1 |
| Same holiday name | +1 |

Matching is greedy: once a reference date is claimed, it cannot be reused.

### Step 3: Quality Assessment

Each mapping gets a quality label:
- **exact** (score ≥ 3): Reliable for YoY comparison
- **partial** (score 1-2): Usable with caution
- **fallback** (score 0): Calendar misalignment too large
- **unmatched**: No reference date available

### Step 4: QA Reporting

The CalendarQA module checks for:
- Weekday misalignment (Mon mapped to Thu)
- Holiday/non-holiday mixing
- Vacation period mismatches
- Unmapped dates

## When to Use

| Scenario | Use DM? |
|---|---|
| Monthly YoY reporting | ✅ Always |
| Daily forecasting with YoY features | ✅ Use comparable_date |
| Weekly comparisons | ✅ Reduces weekday composition effects |
| Intraday analysis | ⚠️ Overkill |

## Limitations

1. **Novel events**: New holidays or unprecedented events have no historical comparable
2. **Structural breaks**: Store openings/closings break comparability regardless of calendar
3. **Data sparsity**: New entities with <12 months of data cannot do YoY mapping
