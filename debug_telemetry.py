"""
Telemetry Debugging and Diagnostic Tool

This script diagnoses issues with telemetry collection and storage.
Run this to identify what's going wrong with phase metrics.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import json


class TelemetryDebugger:
    """Comprehensive telemetry diagnostics."""
    
    def __init__(self, db_path="data/telemetry.db"):
        self.db_path = Path(db_path)
        
    def check_database_exists(self):
        """Check if database file exists."""
        print("\n" + "="*70)
        print("1. DATABASE FILE CHECK")
        print("="*70)
        
        if not self.db_path.exists():
            print(f"❌ CRITICAL: Database file not found: {self.db_path}")
            print(f"   Expected location: {self.db_path.absolute()}")
            return False
        
        print(f"✓ Database file exists: {self.db_path}")
        print(f"  Size: {self.db_path.stat().st_size / 1024:.2f} KB")
        return True
    
    def check_schema(self):
        """Verify database schema."""
        print("\n" + "="*70)
        print("2. SCHEMA VERIFICATION")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['metrics', 'policy_updates', 'safety_events', 
                          'meta_learning_events']
        
        print(f"\nExpected tables: {expected_tables}")
        print(f"Found tables: {tables}")
        
        missing = set(expected_tables) - set(tables)
        if missing:
            print(f"\n❌ MISSING TABLES: {missing}")
        else:
            print("\n✓ All expected tables present")
        
        # Check metrics table structure
        if 'metrics' in tables:
            cursor.execute("PRAGMA table_info(metrics)")
            columns = cursor.fetchall()
            print(f"\nMetrics table columns ({len(columns)} total):")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        return len(missing) == 0
    
    def check_data_count(self):
        """Check how much data is in each table."""
        print("\n" + "="*70)
        print("3. DATA COUNT CHECK")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        tables = ['metrics', 'policy_updates', 'safety_events', 
                 'meta_learning_events']
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    print(f"⚠️  {table}: 0 rows (EMPTY)")
                else:
                    print(f"✓ {table}: {count:,} rows")
                    
            except sqlite3.Error as e:
                print(f"❌ {table}: Error - {e}")
        
        conn.close()
    
    def check_phase_distribution(self):
        """Check how metrics are distributed across phases."""
        print("\n" + "="*70)
        print("4. PHASE DISTRIBUTION ANALYSIS")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get phase distribution
        cursor.execute("""
            SELECT phase, COUNT(*) as count 
            FROM metrics 
            GROUP BY phase 
            ORDER BY MIN(timestamp)
        """)
        
        phases = cursor.fetchall()
        
        if not phases:
            print("❌ CRITICAL: No metrics found in database!")
            print("   This means telemetry collection is not working.")
        else:
            print(f"\nFound {len(phases)} distinct phases:")
            total = 0
            for phase, count in phases:
                print(f"  - '{phase}': {count:,} metrics")
                total += count
            print(f"\nTotal metrics: {total:,}")
        
        conn.close()
        return phases
    
    def check_phase_names(self):
        """Check for phase naming issues."""
        print("\n" + "="*70)
        print("5. PHASE NAME VALIDATION")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get unique phase names
        cursor.execute("SELECT DISTINCT phase FROM metrics")
        phases = [row[0] for row in cursor.fetchall()]
        
        expected_phases = ['baseline', 'level0_learning', 'level1_learning', 
                          'level2_learning']
        
        print(f"\nExpected phases: {expected_phases}")
        print(f"Found phases: {phases}")
        
        # Check for issues
        issues = []
        
        for phase in phases:
            # Check for whitespace
            if phase != phase.strip():
                issues.append(f"Phase '{phase}' has leading/trailing whitespace")
            
            # Check for case issues
            if phase.lower() in [p.lower() for p in expected_phases]:
                if phase not in expected_phases:
                    issues.append(f"Phase '{phase}' has wrong capitalization")
        
        if issues:
            print("\n⚠️  PHASE NAME ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✓ Phase names look correct")
        
        conn.close()
        return phases
    
    def sample_recent_metrics(self, limit=5):
        """Show sample of recent metrics."""
        print("\n" + "="*70)
        print("6. SAMPLE RECENT METRICS")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM metrics 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ No metrics found!")
            return
        
        print(f"\nShowing {len(rows)} most recent metrics:\n")
        
        for i, row in enumerate(rows, 1):
            print(f"Metric {i}:")
            print(f"  Timestamp: {datetime.fromtimestamp(row['timestamp'])}")
            print(f"  Phase: '{row['phase']}'")
            print(f"  Query Type: {row['query_type']}")
            print(f"  Execution Time: {row['execution_time']:.4f}s")
            print(f"  Success: {row['success']}")
            print(f"  Query Hash: {row['query_hash']}")
            print()
        
        conn.close()
    
    def check_timestamps(self):
        """Check timestamp ranges and gaps."""
        print("\n" + "="*70)
        print("7. TIMESTAMP ANALYSIS")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                MIN(timestamp) as first,
                MAX(timestamp) as last,
                COUNT(*) as total
            FROM metrics
        """)
        
        result = cursor.fetchone()
        
        if result[2] == 0:
            print("❌ No metrics to analyze")
            return
        
        first_time = datetime.fromtimestamp(result[0])
        last_time = datetime.fromtimestamp(result[1])
        duration = result[1] - result[0]
        
        print(f"\nFirst metric: {first_time}")
        print(f"Last metric: {last_time}")
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"Total metrics: {result[2]:,}")
        print(f"Average rate: {result[2]/(duration/60):.1f} metrics/minute")
        
        conn.close()
    
    def test_phase_query(self, phase_name='baseline'):
        """Test the exact query used in get_phase_metrics."""
        print("\n" + "="*70)
        print(f"8. TESTING PHASE QUERY FOR '{phase_name}'")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # This is the exact query from telemetry/storage.py
        cursor.execute("""
            SELECT * FROM metrics
            WHERE phase = ?
            ORDER BY timestamp ASC
        """, (phase_name,))
        
        rows = cursor.fetchall()
        
        print(f"\nQuery: SELECT * FROM metrics WHERE phase = '{phase_name}'")
        print(f"Result: {len(rows)} rows")
        
        if len(rows) == 0:
            print("\n❌ PROBLEM IDENTIFIED!")
            print(f"   The phase query for '{phase_name}' returns 0 rows.")
            print("   This is why phase summaries show 0 metrics.")
            
            # Try to find similar phase names
            cursor.execute("SELECT DISTINCT phase FROM metrics")
            actual_phases = [row[0] for row in cursor.fetchall()]
            
            print(f"\n   Actual phases in database: {actual_phases}")
            print(f"   Looking for: '{phase_name}'")
            
            # Check for close matches
            for actual in actual_phases:
                if phase_name.lower() in actual.lower() or actual.lower() in phase_name.lower():
                    print(f"   ⚠️  Found similar: '{actual}'")
        else:
            print(f"\n✓ Query returned {len(rows)} metrics for phase '{phase_name}'")
        
        conn.close()
        return len(rows)
    
    def run_full_diagnostics(self):
        """Run all diagnostic checks."""
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*15 + "TELEMETRY DIAGNOSTIC REPORT" + " "*26 + "║")
        print("╚" + "="*68 + "╝")
        
        if not self.check_database_exists():
            print("\n❌ FATAL: Cannot proceed without database file")
            return
        
        self.check_schema()
        self.check_data_count()
        phases = self.check_phase_distribution()
        found_phases = self.check_phase_names()
        self.sample_recent_metrics()
        self.check_timestamps()
        
        # Test phase queries for all expected phases
        print("\n" + "="*70)
        print("9. TESTING ALL PHASE QUERIES")
        print("="*70)
        
        expected_phases = ['baseline', 'level0_learning', 'level1_learning', 
                          'level2_learning']
        
        results = {}
        for phase in expected_phases:
            count = self.test_phase_query(phase)
            results[phase] = count
        
        # Final summary
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        total_issues = 0
        
        print("\nPhase Query Results:")
        for phase, count in results.items():
            if count == 0:
                print(f"  ❌ {phase}: 0 metrics (PROBLEM!)")
                total_issues += 1
            else:
                print(f"  ✓ {phase}: {count} metrics")
        
        if total_issues == 0:
            print("\n✅ All checks passed! Telemetry system appears healthy.")
        else:
            print(f"\n⚠️  Found {total_issues} issues that need fixing.")
            print("\nRecommended fixes:")
            print("  1. Check phase name assignment in main.py")
            print("  2. Verify telemetry_collector.set_phase() is being called")
            print("  3. Run fix_telemetry.py to correct phase names")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Debug telemetry collection and storage"
    )
    parser.add_argument(
        "--db",
        default="data/telemetry.db",
        help="Path to telemetry database"
    )
    parser.add_argument(
        "--phase",
        help="Test specific phase query"
    )
    
    args = parser.parse_args()
    
    debugger = TelemetryDebugger(args.db)
    
    if args.phase:
        debugger.test_phase_query(args.phase)
    else:
        debugger.run_full_diagnostics()


if __name__ == "__main__":
    main()