import sqlite3
from pathlib import Path

db_path = Path("data/telemetry.db")

if not db_path.exists():
    print(f"ERROR: Database not found at {db_path}")
    exit(1)

print(f"\n{'='*70}")
print(f"Database Diagnostics: {db_path}")
print(f"{'='*70}\n")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print(f"Tables found: {len(tables)}")
for table in tables:
    print(f"  - {table[0]}")

print()

# Check each table for row counts and columns
for table in tables:
    table_name = table[0]
    print(f"\nTable: {table_name}")
    print("-" * 70)
    
    # Get column info
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"Columns ({len(columns)}):")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Row count: {count}")
    
    # Show sample data if exists
    if count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cursor.fetchall()
        print(f"Sample data (first 3 rows):")
        for i, row in enumerate(rows, 1):
            print(f"  Row {i}: {row[:5]}..." if len(row) > 5 else f"  Row {i}: {row}")

conn.close()

print(f"\n{'='*70}")
print("Diagnostics complete")
print(f"{'='*70}\n")