"""
Schema Definitions — Standard table schemas for Praxis data layer.

Defines the expected schema for metric, calendar, and entity tables.
All names are generic — no proprietary references.
"""

# Standard schemas as column lists with types
SCHEMAS = {
    "metrics": {
        "description": "Daily metric data (fact table)",
        "columns": {
            "date": "DATE",
            "entity": "VARCHAR",
            "metric": "VARCHAR",
            "value": "DOUBLE",
        },
        "primary_key": ["date", "entity", "metric"],
    },
    "calendar_dim": {
        "description": "Calendar dimension with holiday/vacation flags",
        "columns": {
            "date": "DATE",
            "weekday": "INTEGER",
            "weekday_name": "VARCHAR",
            "daytype": "VARCHAR",
            "is_holiday": "BOOLEAN",
            "holiday_name": "VARCHAR",
            "is_vacation": "BOOLEAN",
            "is_workday": "BOOLEAN",
            "comparable_date": "DATE",
        },
    },
    "entity_dim": {
        "description": "Entity dimension (store/product/region)",
        "columns": {
            "entity": "VARCHAR",
            "name": "VARCHAR",
            "tier": "VARCHAR",
            "group": "VARCHAR",
            "is_active": "BOOLEAN",
        },
    },
    "entity_alias": {
        "description": "Entity name crosswalk / alias mapping",
        "columns": {
            "raw_name": "VARCHAR",
            "canonical_name": "VARCHAR",
        },
    },
}


def get_create_sql(table_name: str) -> str:
    """Generate CREATE TABLE SQL for a standard schema."""
    if table_name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {table_name}. Available: {list(SCHEMAS.keys())}")

    schema = SCHEMAS[table_name]
    cols = ", ".join(f"{name} {dtype}" for name, dtype in schema["columns"].items())
    return f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})"


def get_required_columns(table_name: str) -> list[str]:
    """Get required column names for a standard schema."""
    if table_name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {table_name}")
    return list(SCHEMAS[table_name]["columns"].keys())
