"""
Data ingestion and loading modules.

This package provides utilities for:
- Loading CSV data into SQLite database
- Generating synthetic financial and behavioral datasets
- Querying database tables
"""

from .sql_ingestion import (
    load_to_sql,
    create_connection,
    execute_query,
    ingest_all_raw_data
)

__all__ = [
    'load_to_sql',
    'create_connection',
    'execute_query',
    'ingest_all_raw_data'
]


