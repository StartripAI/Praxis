"""
Calendar QA — Date mapping quality assurance.

Detects mismatches in DM mappings such as:
- Weekday misalignment (target Mon mapped to ref Wed)
- Holiday/non-holiday mismatch
- Vacation status mismatch
- Unmapped dates
"""

from __future__ import annotations

import pandas as pd


class CalendarQA:
    """Quality assurance checks for DM comparable date mappings."""

    @staticmethod
    def check_mapping(dm_mapping: pd.DataFrame) -> pd.DataFrame:
        """Run all QA checks on a DM mapping DataFrame.

        Parameters
        ----------
        dm_mapping : pd.DataFrame
            Output from CalendarEngine.build_dm_mapping()

        Returns
        -------
        DataFrame with additional columns: qa_status, qa_issues
        """
        result = dm_mapping.copy()
        issues_list = []

        for _, row in result.iterrows():
            issues = []

            # Check: unmapped
            if pd.isna(row.get("comparable_date")):
                issues.append("unmapped")
                issues_list.append(issues)
                continue

            # Check: match quality
            if row.get("match_quality") == "fallback":
                issues.append("low_match_quality")

            # Check: weekday alignment
            target_dt = row["daytype"]
            comp_dt = row.get("comparable_daytype", "")
            if target_dt and comp_dt:
                t_weekday = target_dt.split("_")[0]
                c_weekday = comp_dt.split("_")[0]
                if t_weekday != c_weekday:
                    issues.append(f"weekday_mismatch:{t_weekday}→{c_weekday}")

            # Check: holiday mismatch
            t_has_hol = "_hol" in target_dt
            c_has_hol = "_hol" in comp_dt
            if t_has_hol != c_has_hol:
                issues.append("holiday_mismatch")

            # Check: vacation mismatch
            t_has_vac = "_vac" in target_dt
            c_has_vac = "_vac" in comp_dt
            if t_has_vac != c_has_vac:
                issues.append("vacation_mismatch")

            issues_list.append(issues)

        result["qa_issues"] = ["; ".join(i) if i else "" for i in issues_list]
        result["qa_status"] = result["qa_issues"].apply(lambda x: "PASS" if not x else "WARN")

        return result

    @staticmethod
    def summary(qa_result: pd.DataFrame) -> dict:
        """Generate a summary of QA results.

        Returns dict with counts: total, pass, warn, unmapped, issues_breakdown.
        """
        total = len(qa_result)
        n_pass = (qa_result["qa_status"] == "PASS").sum()
        n_warn = (qa_result["qa_status"] == "WARN").sum()
        n_unmapped = qa_result["qa_issues"].str.contains("unmapped").sum()

        # Breakdown by issue type
        all_issues = []
        for issues_str in qa_result["qa_issues"]:
            if issues_str:
                all_issues.extend(issues_str.split("; "))

        breakdown = {}
        for issue in all_issues:
            key = issue.split(":")[0]
            breakdown[key] = breakdown.get(key, 0) + 1

        return {
            "total_dates": int(total),
            "pass": int(n_pass),
            "warn": int(n_warn),
            "pass_rate": round(n_pass / total * 100, 1) if total > 0 else 0,
            "unmapped": int(n_unmapped),
            "issues_breakdown": breakdown,
        }
