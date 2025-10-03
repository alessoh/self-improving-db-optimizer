"""
Force Phase Reassignment - For when all metrics are stuck in one phase

This script forcibly reassigns phases based on timestamps when all
metrics are incorrectly assigned to a single phase.
"""

import sqlite3
from pathlib import Path
from datetime import datetime


def force_reassign_phases(db_path="data/telemetry.db"):
    """Force reassignment of phases based on timestamps."""
    
    print("\n" + "="*70)
    print("FORCE PHASE REASSIGNMENT")
    print("="*70)
    
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Backup
    backup_path = db_path.with_suffix('.db.backup2')
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get total metrics and time range
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(timestamp) as first,
            MAX(timestamp) as last
        FROM metrics
    """)
    
    total, min_time, max_time = cursor.fetchone()
    
    print(f"\nTotal metrics: {total:,}")
    print(f"Time range: {datetime.fromtimestamp(min_time)} to {datetime.fromtimestamp(max_time)}")
    
    duration = max_time - min_time
    print(f"Duration: {duration/3600:.2f} hours")
    
    # Based on your demo output, you had 4 phases:
    # baseline: ~8,376 queries (first ~43 minutes = 2592 seconds)
    # level0_learning: ~10,472 queries (next ~58 minutes = 3456 seconds) 
    # level1_learning: ~8,966 queries (next ~58 minutes = 3456 seconds)
    # level2_learning: ~6,578 queries (final ~43 minutes = 2592 seconds)
    
    # Calculate phase boundaries based on actual duration
    # From your log: baseline=2592s, level0=3456s, level1=3456s, level2=2592s
    total_planned = 2592 + 3456 + 3456 + 2592  # = 12096 seconds
    
    # Calculate actual phase boundaries
    baseline_end = min_time + 2592
    level0_end = baseline_end + 3456
    level1_end = level0_end + 3456
    # level2_end = max_time (everything else)
    
    print(f"\nPhase boundaries:")
    print(f"  Baseline: {datetime.fromtimestamp(min_time)} to {datetime.fromtimestamp(baseline_end)}")
    print(f"  Level0:   {datetime.fromtimestamp(baseline_end)} to {datetime.fromtimestamp(level0_end)}")
    print(f"  Level1:   {datetime.fromtimestamp(level0_end)} to {datetime.fromtimestamp(level1_end)}")
    print(f"  Level2:   {datetime.fromtimestamp(level1_end)} to {datetime.fromtimestamp(max_time)}")
    
    # Update phases
    print("\nReassigning phases...")
    
    # Baseline
    cursor.execute("""
        UPDATE metrics 
        SET phase = 'baseline'
        WHERE timestamp < ?
    """, (baseline_end,))
    baseline_count = cursor.rowcount
    print(f"  ✓ Assigned {baseline_count:,} metrics to 'baseline'")
    
    # Level 0
    cursor.execute("""
        UPDATE metrics 
        SET phase = 'level0_learning'
        WHERE timestamp >= ? AND timestamp < ?
    """, (baseline_end, level0_end))
    level0_count = cursor.rowcount
    print(f"  ✓ Assigned {level0_count:,} metrics to 'level0_learning'")
    
    # Level 1
    cursor.execute("""
        UPDATE metrics 
        SET phase = 'level1_learning'
        WHERE timestamp >= ? AND timestamp < ?
    """, (level0_end, level1_end))
    level1_count = cursor.rowcount
    print(f"  ✓ Assigned {level1_count:,} metrics to 'level1_learning'")
    
    # Level 2
    cursor.execute("""
        UPDATE metrics 
        SET phase = 'level2_learning'
        WHERE timestamp >= ?
    """, (level1_end,))
    level2_count = cursor.rowcount
    print(f"  ✓ Assigned {level2_count:,} metrics to 'level2_learning'")
    
    conn.commit()
    
    # Verify
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    cursor.execute("""
        SELECT phase, COUNT(*) as count 
        FROM metrics 
        GROUP BY phase 
        ORDER BY MIN(timestamp)
    """)
    
    phases = cursor.fetchall()
    
    print("\nPhase distribution after reassignment:")
    total_verified = 0
    for phase, count in phases:
        print(f"  {phase}: {count:,} metrics")
        total_verified += count
    
    print(f"\nTotal: {total_verified:,} metrics")
    
    if total_verified == total:
        print("✅ All metrics successfully reassigned!")
    else:
        print(f"⚠️  Warning: {total - total_verified} metrics may be missing")
    
    conn.close()
    
    return True


def main():
    """Main entry point."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "FORCE PHASE REASSIGNMENT" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\n⚠️  This will FORCE reassignment of ALL metrics to correct phases.")
    print("    This is safe - a backup will be created.")
    response = input("\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    if force_reassign_phases():
        print("\n" + "="*70)
        print("✅ REASSIGNMENT COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Verify: sqlite3 data/telemetry.db \"SELECT phase, COUNT(*) FROM metrics GROUP BY phase;\"")
        print("  2. Test: python debug_telemetry.py")
        print("  3. View results: python view_results.py")


if __name__ == "__main__":
    main()