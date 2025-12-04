"""
Simple demonstration of SQL queries on ingested data.
"""

from sql_ingestion import execute_query

print("SQL Query Demonstration")
print("=" * 60)

# Basic SELECT
print("\n1. Count records in each table:")
result = execute_query("SELECT 'behavioral_data' as table_name, COUNT(*) as count FROM behavioral_data UNION ALL SELECT 'DGS10', COUNT(*) FROM DGS10 UNION ALL SELECT 'DFF', COUNT(*) FROM DFF")
print(result)

# Filtering
print("\n2. Customer segments distribution:")
result = execute_query("SELECT customer_segment, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM behavioral_data), 2) as percentage FROM behavioral_data GROUP BY customer_segment")
print(result)

# Date range query
print("\n3. DGS10 rates in 2024:")
result = execute_query("SELECT COUNT(*) as count, MIN(DGS10) as min_rate, MAX(DGS10) as max_rate, AVG(DGS10) as avg_rate FROM DGS10 WHERE date >= '2024-01-01' AND date < '2025-01-01'")
print(result)

# Join query
print("\n4. Combined rates (DGS10 and DFF) for recent dates:")
result = execute_query("SELECT DGS10.date, DGS10.DGS10, DFF.DFF FROM DGS10 JOIN DFF ON DGS10.date = DFF.date ORDER BY DGS10.date DESC LIMIT 10")
print(result)

print("\n" + "=" * 60)
print("All queries executed successfully!")

