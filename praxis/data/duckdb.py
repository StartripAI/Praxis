"""
DuckDB Data Layer — Embedded analytics truth store.

Provides a lightweight DuckDB-based data layer for:
- Schema management (star schema: fact + dimension tables)
- Data ingestion from CSV/Excel
- SQL query interface with pandas integration
- Replacement scans (query pandas DataFrames directly in SQL)

Generic table names — no proprietary schema references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class DuckDBStore:
    """Embedded DuckDB analytics store.

    Parameters
    ----------
    path : str | Path
        Path to .duckdb file. Use ":memory:" for in-memory.
    """

    def __init__(self, path: str | Path = ":memory:"):
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        self.path = str(path)
        self._con = duckdb.connect(self.path)

    def execute(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Execute SQL and return results as DataFrame."""
        if params:
            return self._con.execute(sql, params).fetchdf()
        return self._con.execute(sql).fetchdf()

    def load_csv(
        self,
        table_name: str,
        csv_path: str | Path,
        if_exists: str = "replace",
    ):
        """Load a CSV file into a DuckDB table.

        Parameters
        ----------
        table_name : str
            Target table name.
        csv_path : str | Path
            Path to CSV file.
        if_exists : str
            "replace" or "append".
        """
        csv_path = str(csv_path)
        if if_exists == "replace":
            self._con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
        else:
            self._con.execute(f"INSERT INTO {table_name} SELECT * FROM read_csv_auto('{csv_path}')")

    def load_dataframe(
        self,
        table_name: str,
        df: pd.DataFrame,
        if_exists: str = "replace",
    ):
        """Load a pandas DataFrame into a DuckDB table."""
        if if_exists == "replace":
            self._con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        else:
            self._con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    def query_df(self, df: pd.DataFrame, sql: str) -> pd.DataFrame:
        """Query a pandas DataFrame using SQL (replacement scan).

        Example:
            store.query_df(my_df, "SELECT * FROM df WHERE value > 100")
        """
        return self._con.execute(sql).fetchdf()

    def tables(self) -> list[str]:
        """List all tables in the database."""
        result = self._con.execute("SHOW TABLES").fetchdf()
        return result["name"].tolist() if len(result) > 0 else []

    def schema(self, table_name: str) -> pd.DataFrame:
        """Get column schema for a table."""
        return self._con.execute(f"DESCRIBE {table_name}").fetchdf()

    def close(self):
        """Close the database connection."""
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
