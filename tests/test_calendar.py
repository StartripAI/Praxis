"""Tests for praxis.calendar module."""

from datetime import date

import pandas as pd
import pytest

from praxis.calendar.engine import CalendarEngine, _daytype
from praxis.calendar.qa import CalendarQA
from praxis.calendar.periods import month_range, mtd_progress, build_period_index


class TestDaytype:
    def test_weekday(self):
        # 2026-03-10 is a Tuesday
        result = _daytype(date(2026, 3, 10))
        assert result == "Tue"

    def test_holiday(self):
        result = _daytype(date(2026, 1, 1), is_holiday=True, holiday_name="new_year")
        assert result == "Thu_hol:new_year"

    def test_vacation(self):
        result = _daytype(date(2026, 7, 15), is_vacation=True)
        assert "Wed_vac" == result

    def test_holiday_and_vacation(self):
        result = _daytype(date(2026, 2, 1), is_holiday=True, holiday_name="spring_festival", is_vacation=True)
        assert "hol:spring_festival" in result
        assert "_vac" in result


class TestCalendarEngine:
    def test_build_date_features(self):
        engine = CalendarEngine(country="CN", year=2026)
        df = engine.build_date_features("2026-03-01", "2026-03-31")
        assert len(df) == 31
        assert "date" in df.columns
        assert "daytype" in df.columns
        assert "is_holiday" in df.columns
        assert "is_workday" in df.columns

    def test_date_features_types(self):
        engine = CalendarEngine(country="CN", year=2026)
        df = engine.build_date_features("2026-03-01", "2026-03-07")
        assert df["weekday"].dtype in ("int64", "int32")
        assert df["is_holiday"].dtype == bool

    def test_build_dm_mapping(self):
        engine = CalendarEngine(country="CN", year=2026)
        dm = engine.build_dm_mapping(target_year=2026, target_month=3)
        assert len(dm) == 31
        assert "date" in dm.columns
        assert "comparable_date" in dm.columns
        assert "match_quality" in dm.columns

    def test_dm_mapping_quality(self):
        engine = CalendarEngine(country="CN", year=2026)
        dm = engine.build_dm_mapping(target_year=2026, target_month=3)
        # Most should be exact or partial matches
        quality_counts = dm["match_quality"].value_counts()
        assert quality_counts.get("exact", 0) + quality_counts.get("partial", 0) > 20

    def test_school_vacation(self):
        engine = CalendarEngine(
            country="CN",
            year=2026,
            school_vacations=[("2026-07-01", "2026-08-31")],
        )
        df = engine.build_date_features("2026-07-15", "2026-07-15")
        assert df.iloc[0]["is_vacation"] == True

    def test_no_vacation(self):
        engine = CalendarEngine(country="CN", year=2026)
        df = engine.build_date_features("2026-03-10", "2026-03-10")
        assert df.iloc[0]["is_vacation"] == False


class TestCalendarQA:
    def test_check_mapping(self):
        engine = CalendarEngine(country="CN", year=2026)
        dm = engine.build_dm_mapping(target_year=2026, target_month=3)
        qa_result = CalendarQA.check_mapping(dm)
        assert "qa_status" in qa_result.columns
        assert "qa_issues" in qa_result.columns
        assert qa_result["qa_status"].isin(["PASS", "WARN"]).all()

    def test_summary(self):
        engine = CalendarEngine(country="CN", year=2026)
        dm = engine.build_dm_mapping(target_year=2026, target_month=3)
        qa_result = CalendarQA.check_mapping(dm)
        summary = CalendarQA.summary(qa_result)
        assert summary["total_dates"] == 31
        assert 0 <= summary["pass_rate"] <= 100


class TestPeriods:
    def test_month_range(self):
        first, last = month_range(2026, 3)
        assert first == date(2026, 3, 1)
        assert last == date(2026, 3, 31)

    def test_month_range_feb(self):
        first, last = month_range(2026, 2)
        assert first == date(2026, 2, 1)
        assert last == date(2026, 2, 28)

    def test_month_range_dec(self):
        first, last = month_range(2026, 12)
        assert first == date(2026, 12, 1)
        assert last == date(2026, 12, 31)

    def test_mtd_progress(self):
        result = mtd_progress(2026, 3, date(2026, 3, 15))
        assert result["days_elapsed"] == 15
        assert result["days_total"] == 31
        assert result["days_remaining"] == 16
        assert 40 < result["pct_elapsed"] < 55

    def test_build_period_index(self):
        pi = build_period_index(2026, 1, 6)
        assert len(pi) == 6
        assert pi.iloc[0]["month"] == 1
        assert pi.iloc[5]["month"] == 6
