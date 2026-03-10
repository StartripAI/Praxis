"""
Calendar Periods — DM period management and MTD tracking.

Provides tools for defining and working with Discrete Month periods,
including month-to-date (MTD) progress and period comparisons.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def month_range(year: int, month: int) -> tuple[date, date]:
    """Return (first_day, last_day) of a month."""
    first = date(year, month, 1)
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return first, last


def count_workdays(start: date, end: date, holiday_checker=None) -> int:
    """Count workdays (Mon-Fri, excluding holidays) in a date range."""
    count = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            if holiday_checker:
                is_hol, _ = holiday_checker.check(d)
                if not is_hol:
                    count += 1
            else:
                count += 1
        d += timedelta(days=1)
    return count


def mtd_progress(
    year: int,
    month: int,
    as_of: date,
) -> dict:
    """Calculate month-to-date progress.

    Returns dict with:
    - days_elapsed, days_total, days_remaining
    - pct_elapsed (0-100)
    """
    first, last = month_range(year, month)
    total = (last - first).days + 1
    elapsed = min((as_of - first).days + 1, total)
    elapsed = max(elapsed, 0)

    return {
        "year": year,
        "month": month,
        "as_of": str(as_of),
        "days_elapsed": elapsed,
        "days_total": total,
        "days_remaining": total - elapsed,
        "pct_elapsed": round(elapsed / total * 100, 1) if total > 0 else 0,
    }


def build_period_index(
    start_year: int,
    start_month: int,
    n_months: int,
) -> pd.DataFrame:
    """Build a period index for multiple months.

    Returns DataFrame with: year, month, start_date, end_date, n_days.
    """
    records = []
    y, m = start_year, start_month
    for _ in range(n_months):
        first, last = month_range(y, m)
        records.append(
            {
                "year": y,
                "month": m,
                "start_date": pd.Timestamp(first),
                "end_date": pd.Timestamp(last),
                "n_days": (last - first).days + 1,
            }
        )
        # Advance to next month
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

    return pd.DataFrame(records)
