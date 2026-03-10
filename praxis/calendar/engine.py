"""
Praxis Calendar Engine — Trading-Day Alignment (DM Mapping)

Maps each date in a target period to its "comparable trading day" from
a reference period, ensuring like-for-like YoY comparisons.

This implements the same concept as the NRF 4-5-4 retail calendar and
X-13ARIMA-SEATS trading day / moving holiday adjustments, but at a
day-level granularity suitable for operational forecasting.

Key concepts:
- DM (Discrete Month): a month-level period defined by its daytype mix
- Daytype: a combination of (weekday, holiday_status, vacation_status)
- Comparable date: the date in the reference period with the same daytype
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from praxis.calendar.holidays import get_holiday_engine


def _daytype(
    d: date,
    *,
    is_holiday: bool = False,
    is_vacation: bool = False,
    holiday_name: str = "",
) -> str:
    """Build a canonical daytype string for a date.

    Format: "{weekday}[_hol:{name}][_vac]"
    Examples: "Mon", "Sat_hol:spring_festival", "Wed_vac"
    """
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dt = weekday_names[d.weekday()]
    if is_holiday and holiday_name:
        dt += f"_hol:{holiday_name}"
    elif is_holiday:
        dt += "_hol"
    if is_vacation:
        dt += "_vac"
    return dt


class CalendarEngine:
    """Trading-Day Alignment engine for comparable period mapping.

    Parameters
    ----------
    country : str
        ISO country code, default "CN".
    year : int
        Reference year for calendar construction.
    school_vacations : list[tuple[str, str]] | None
        Optional list of (start_date, end_date) for school vacation periods.
        Example: [("2025-01-20", "2025-02-14"), ("2025-07-01", "2025-08-31")]
    """

    def __init__(
        self,
        country: str = "CN",
        year: int = 2026,
        school_vacations: Optional[list[tuple[str, str]]] = None,
    ):
        self.country = country
        self.year = year
        self.school_vacations = school_vacations or []
        self._holiday_engine = get_holiday_engine(country)

    def _is_vacation(self, d: date) -> bool:
        """Check if a date falls within any school vacation period."""
        for start_s, end_s in self.school_vacations:
            start = pd.Timestamp(start_s).date()
            end = pd.Timestamp(end_s).date()
            if start <= d <= end:
                return True
        return False

    def _get_daytype(self, d: date) -> str:
        """Get the canonical daytype for a single date."""
        is_hol, hol_name = self._holiday_engine.check(d)
        is_vac = self._is_vacation(d)
        return _daytype(d, is_holiday=is_hol, is_vacation=is_vac, holiday_name=hol_name)

    def build_date_features(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Build a date-level feature table for a range.

        Returns DataFrame with columns:
        - date, year, month, day, weekday, daytype
        - is_holiday, holiday_name, is_vacation, is_workday
        """
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()

        records = []
        d = start
        while d <= end:
            is_hol, hol_name = self._holiday_engine.check(d)
            is_vac = self._is_vacation(d)
            dt = _daytype(d, is_holiday=is_hol, is_vacation=is_vac, holiday_name=hol_name)

            records.append(
                {
                    "date": pd.Timestamp(d),
                    "year": d.year,
                    "month": d.month,
                    "day": d.day,
                    "weekday": d.weekday(),
                    "weekday_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()],
                    "daytype": dt,
                    "is_holiday": is_hol,
                    "holiday_name": hol_name,
                    "is_vacation": is_vac,
                    "is_workday": d.weekday() < 5 and not is_hol,
                }
            )
            d += timedelta(days=1)

        return pd.DataFrame(records)

    def build_dm_mapping(
        self,
        target_year: int,
        target_month: int,
        reference_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Build Discrete Month (DM) comparable day mapping.

        For each date in the target month, find the best matching date
        in the reference period (same month, previous year) based on
        daytype similarity.

        Parameters
        ----------
        target_year : int
        target_month : int
        reference_year : int, optional
            Default: target_year - 1

        Returns
        -------
        DataFrame with columns: date, comparable_date, daytype,
            comparable_daytype, match_quality
        """
        if reference_year is None:
            reference_year = target_year - 1

        # Build features for both periods
        target_start = date(target_year, target_month, 1)
        if target_month == 12:
            target_end = date(target_year + 1, 1, 1) - timedelta(days=1)
        else:
            target_end = date(target_year, target_month + 1, 1) - timedelta(days=1)

        ref_start = date(reference_year, target_month, 1)
        if target_month == 12:
            ref_end = date(reference_year + 1, 1, 1) - timedelta(days=1)
        else:
            ref_end = date(reference_year, target_month + 1, 1) - timedelta(days=1)

        target_df = self.build_date_features(target_start, target_end)
        ref_df = self.build_date_features(ref_start, ref_end)

        # Match by daytype — greedy assignment
        mappings = []
        ref_used = set()

        for _, trow in target_df.iterrows():
            best_idx = None
            best_score = -1

            for idx, rrow in ref_df.iterrows():
                if idx in ref_used:
                    continue
                score = self._match_score(trow, rrow)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                ref_used.add(best_idx)
                rrow = ref_df.loc[best_idx]
                mappings.append(
                    {
                        "date": trow["date"],
                        "comparable_date": rrow["date"],
                        "daytype": trow["daytype"],
                        "comparable_daytype": rrow["daytype"],
                        "match_quality": "exact" if best_score >= 3 else "partial" if best_score >= 1 else "fallback",
                    }
                )
            else:
                mappings.append(
                    {
                        "date": trow["date"],
                        "comparable_date": pd.NaT,
                        "daytype": trow["daytype"],
                        "comparable_daytype": "",
                        "match_quality": "unmatched",
                    }
                )

        return pd.DataFrame(mappings)

    @staticmethod
    def _match_score(target_row: pd.Series, ref_row: pd.Series) -> int:
        """Score how well two dates match as comparable trading days.

        Scoring:
        - Same weekday: +2
        - Same holiday status: +1
        - Same vacation status: +1
        - Same holiday name: +1
        """
        score = 0
        if target_row["weekday"] == ref_row["weekday"]:
            score += 2
        if target_row["is_holiday"] == ref_row["is_holiday"]:
            score += 1
        if target_row["is_vacation"] == ref_row["is_vacation"]:
            score += 1
        if target_row["holiday_name"] == ref_row["holiday_name"] and target_row["holiday_name"]:
            score += 1
        return score
