"""
SQL ingestion pipeline for loading CSV data into SQLite database.

This module provides functions to load raw CSV files into SQLite tables
for efficient querying and data management.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional


def load_to_sql(
    csv_path: str,
    table_name: str,
    db_path: str = "data/database.db",
    if_exists: str = "replace",
    index: bool = None
) -> None:
    """
    Load CSV file into SQLite database table.
    
    Args:
        csv_path: Path to CSV file
        table_name: Name of the SQL table
        db_path: Path to SQLite database file
        if_exists: How to behave if table exists ('replace', 'append', 'fail')
        index: Whether to write DataFrame index as a column (None = auto-detect)
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to read with index if first column looks like dates
    df_test = pd.read_csv(csv_path, nrows=1)
    first_col = df_test.columns[0]
    
    # Auto-detect if first column is a date index
    if index is None:
        if (first_col in ['date', 'Date', 'DATE'] or 'Unnamed: 0' in first_col):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = 'date'
            df = df.reset_index()
            index = False
        else:
            df = pd.read_csv(csv_path)
            index = False
    else:
        if index and (first_col in ['date', 'Date', 'DATE'] or 'Unnamed: 0' in first_col):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = 'date'
            df = df.reset_index()
            index = False
        else:
            df = pd.read_csv(csv_path)
    
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=index)
        print(f"Loaded {len(df)} rows from {csv_path} into table '{table_name}'")
    finally:
        conn.close()


def create_connection(db_path: str = "data/database.db") -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite connection object
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def execute_query(
    query: str,
    db_path: str = "data/database.db",
    return_df: bool = True
) -> Optional[pd.DataFrame]:
    """
    Execute a SQL query and optionally return results as DataFrame.
    
    Args:
        query: SQL query string
        db_path: Path to SQLite database file
        return_df: If True, return results as DataFrame; if False, return None
        
    Returns:
        DataFrame with query results or None
    """
    conn = create_connection(db_path)
    try:
        if return_df:
            df = pd.read_sql_query(query, conn)
            return df
        else:
            conn.execute(query)
            conn.commit()
            return None
    finally:
        conn.close()


def ingest_all_raw_data(
    raw_data_dir: str = "data/raw",
    db_path: str = "data/database.db"
) -> None:
    """
    Ingest all CSV files from raw data directory into SQLite database.
    
    Args:
        raw_data_dir: Directory containing raw CSV files
        db_path: Path to SQLite database file
    """
    raw_path = Path(raw_data_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    csv_files = list(raw_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {raw_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to ingest")
    
    for csv_file in csv_files:
        table_name = csv_file.stem
        
        if table_name == "README":
            continue
        
        try:
            load_to_sql(str(csv_file), table_name, db_path)
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")


if __name__ == "__main__":
    ingest_all_raw_data()

