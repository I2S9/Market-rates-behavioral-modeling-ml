"""
Test SQL queries to validate database ingestion.

This script tests basic SQL operations: SELECT, filtering, and date-range queries.
"""

from sql_ingestion import execute_query, create_connection
import pandas as pd


def test_basic_select():
    """Test basic SELECT queries."""
    print("=" * 60)
    print("Test 1: Basic SELECT queries")
    print("=" * 60)
    
    queries = [
        ("SELECT COUNT(*) as total_rows FROM behavioral_data", "Count behavioral data"),
        ("SELECT COUNT(*) as total_rows FROM DGS10", "Count DGS10 data"),
        ("SELECT * FROM behavioral_data LIMIT 5", "Sample behavioral data"),
        ("SELECT * FROM DGS10 LIMIT 5", "Sample DGS10 data")
    ]
    
    for query, description in queries:
        print(f"\n{description}:")
        result = execute_query(query)
        print(result)


def test_filtering():
    """Test filtering queries."""
    print("\n" + "=" * 60)
    print("Test 2: Filtering queries")
    print("=" * 60)
    
    queries = [
        (
            "SELECT customer_segment, COUNT(*) as count FROM behavioral_data GROUP BY customer_segment",
            "Count by customer segment"
        ),
        (
            "SELECT * FROM behavioral_data WHERE outflow_event = 1 LIMIT 10",
            "Customers with outflow events"
        ),
        (
            "SELECT AVG(balance_volatility) as avg_volatility, customer_segment FROM behavioral_data GROUP BY customer_segment",
            "Average volatility by segment"
        ),
        (
            "SELECT * FROM DGS10 WHERE DGS10 > 3.5 ORDER BY date DESC LIMIT 10",
            "DGS10 rates above 3.5%"
        )
    ]
    
    for query, description in queries:
        print(f"\n{description}:")
        result = execute_query(query)
        print(result)


def test_date_range():
    """Test date-range queries."""
    print("\n" + "=" * 60)
    print("Test 3: Date-range queries")
    print("=" * 60)
    
    queries = [
        (
            "SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count FROM DGS10",
            "DGS10 date range"
        ),
        (
            "SELECT * FROM DGS10 WHERE date >= '2024-01-01' AND date <= '2024-12-31' ORDER BY date LIMIT 10",
            "DGS10 data for 2024"
        ),
        (
            "SELECT * FROM behavioral_data WHERE date >= '2024-01-01' LIMIT 10",
            "Behavioral data from 2024 onwards"
        ),
        (
            "SELECT DGS10.date, DGS10.DGS10, DFF.DFF FROM DGS10 JOIN DFF ON DGS10.date = DFF.date WHERE DGS10.date >= '2024-01-01' LIMIT 10",
            "Join DGS10 and DFF for 2024"
        )
    ]
    
    for query, description in queries:
        print(f"\n{description}:")
        try:
            result = execute_query(query)
            print(result)
        except Exception as e:
            print(f"Error: {e}")


def test_table_structure():
    """Test table structure queries."""
    print("\n" + "=" * 60)
    print("Test 4: Table structure")
    print("=" * 60)
    
    conn = create_connection()
    cursor = conn.cursor()
    
    tables = ['behavioral_data', 'DGS10', 'DFF', 'DGS30']
    
    for table in tables:
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                print(f"\n{table} columns:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
        except Exception as e:
            print(f"Error checking {table}: {e}")
    
    conn.close()


if __name__ == "__main__":
    test_basic_select()
    test_filtering()
    test_date_range()
    test_table_structure()
    print("\n" + "=" * 60)
    print("All SQL tests completed")
    print("=" * 60)

