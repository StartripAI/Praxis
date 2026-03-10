"""Tests for praxis.data module."""

import pandas as pd
import pytest
from praxis.data.schema import get_create_sql, get_required_columns, SCHEMAS
from praxis.data.ingestion import validate_schema


class TestSchema:
    def test_metrics_schema(self):
        sql = get_create_sql("metrics")
        assert "CREATE TABLE" in sql
        assert "date" in sql
        assert "value" in sql

    def test_calendar_dim_schema(self):
        sql = get_create_sql("calendar_dim")
        assert "daytype" in sql

    def test_unknown_schema_raises(self):
        with pytest.raises(ValueError, match="Unknown schema"):
            get_create_sql("nonexistent_table")

    def test_required_columns(self):
        cols = get_required_columns("metrics")
        assert "date" in cols
        assert "entity" in cols
        assert "value" in cols

    def test_no_proprietary_names(self):
        """Ensure no proprietary table names in schemas."""
        for name in SCHEMAS:
            assert "fact_dashboard" not in name
            assert "fact_store" not in name
            assert "fact_hq" not in name


class TestValidation:
    def test_valid_schema(self):
        df = pd.DataFrame({"date": ["2024-01-01"], "entity": ["A"], "value": [100]})
        result = validate_schema(df, required_cols=["date", "entity", "value"])
        assert result["valid"] is True

    def test_missing_columns(self):
        df = pd.DataFrame({"date": ["2024-01-01"]})
        result = validate_schema(df, required_cols=["date", "entity", "value"])
        assert result["valid"] is False
        assert "entity" in result["missing_cols"]

    def test_date_range(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30)})
        result = validate_schema(df, required_cols=["date"])
        assert result["date_range"]["min"] == "2024-01-01"


class TestDuckDBStore:
    def test_memory_store(self):
        from praxis.data.duckdb import DuckDBStore
        with DuckDBStore(":memory:") as store:
            df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
            store.load_dataframe("test_table", df)
            result = store.execute("SELECT * FROM test_table")
            assert len(result) == 3

    def test_tables_list(self):
        from praxis.data.duckdb import DuckDBStore
        with DuckDBStore(":memory:") as store:
            df = pd.DataFrame({"a": [1]})
            store.load_dataframe("my_table", df)
            tables = store.tables()
            assert "my_table" in tables

    def test_replacement_scan(self):
        from praxis.data.duckdb import DuckDBStore
        with DuckDBStore(":memory:") as store:
            df = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
            result = store.query_df(df, "SELECT * FROM df WHERE x > 15")
            assert len(result) == 2
