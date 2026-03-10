"""
Holiday engine — thin wrapper over chinese-calendar and holidays libraries.

Provides a unified API for holiday detection across countries.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol


class HolidayChecker(Protocol):
    """Protocol for holiday checking backends."""

    def check(self, d: date) -> tuple[bool, str]:
        """Check if a date is a holiday.

        Returns (is_holiday, holiday_name).
        """
        ...


class ChinaHolidayEngine:
    """Holiday engine for China using chinese-calendar."""

    def __init__(self):
        try:
            import chinese_calendar as cc
            self._cc = cc
        except ImportError:
            raise ImportError(
                "chinese-calendar is required for CN holidays. "
                "Install with: pip install chinese-calendar"
            )

    def check(self, d: date) -> tuple[bool, str]:
        """Check if a date is a Chinese holiday.

        Returns (is_holiday, normalized_holiday_name).
        """
        try:
            is_holiday = self._cc.is_holiday(d)
            if is_holiday:
                detail = self._cc.get_holiday_detail(d)
                if detail and detail[1]:
                    name = self._normalize_name(detail[1].name if hasattr(detail[1], "name") else str(detail[1]))
                    return True, name
                return True, "holiday"
            return False, ""
        except (NotImplementedError, ValueError):
            # Date out of range for chinese-calendar
            return False, ""

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize holiday name to a generic English identifier."""
        mapping = {
            "spring_festival": ["春节", "Spring Festival"],
            "new_year": ["元旦", "New Year"],
            "qingming": ["清明", "Tomb"],
            "labor_day": ["劳动", "Labour", "Labor"],
            "dragon_boat": ["端午", "Dragon Boat"],
            "mid_autumn": ["中秋", "Mid-autumn", "Mid-Autumn"],
            "national_day": ["国庆", "National Day"],
        }
        for key, patterns in mapping.items():
            for pattern in patterns:
                if pattern in name:
                    return key
        return name.lower().replace(" ", "_")


class GlobalHolidayEngine:
    """Holiday engine using the holidays library for international support."""

    def __init__(self, country: str = "US"):
        try:
            import holidays as hol_lib
            self._holidays = hol_lib.country_holidays(country)
            self._country = country
        except ImportError:
            raise ImportError(
                "holidays library is required for international holidays. "
                "Install with: pip install holidays"
            )

    def check(self, d: date) -> tuple[bool, str]:
        """Check if a date is a holiday."""
        if d in self._holidays:
            name = self._holidays.get(d, "holiday")
            return True, name.lower().replace(" ", "_").replace("'", "")
        return False, ""


def get_holiday_engine(country: str = "CN") -> HolidayChecker:
    """Factory function to get the appropriate holiday engine.

    Parameters
    ----------
    country : str
        ISO country code. "CN" uses chinese-calendar, others use holidays lib.
    """
    if country.upper() == "CN":
        return ChinaHolidayEngine()
    return GlobalHolidayEngine(country)
