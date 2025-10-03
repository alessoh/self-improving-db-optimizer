"""
Telemetry System Fixes

This script fixes common telemetry issues:
1. Missing phase names in metrics
2. Incorrect phase name formatting
3. Phase synchronization between collector and storage
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime


class TelemetryFixer:
    """Fix telemetry issues automatically."""
    
    def __init__(self, db_path="data/telemetry.db"):
        self.db_path = Path(db_path)
        
    def backup_database(self):
        """Create backup before making changes."""
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            return False
        
        backup_path = self.db_path.with_suffix('.db.backup')
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        print(f"✓ Created backup: {backup_path}")
        return True
    
    def fix_missing_phases(self):
        """Fix metrics with missing or incorrect phase names."""
        print("\n" + "="*70)
        print("FIXING PHASE NAMES")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Find metrics with problematic phases
        cursor.execute("""
            SELECT COUNT(*) FROM metrics 
            WHERE phase IS NULL OR phase = '' OR phase = 'unknown'
        """)
        null_count = cursor.fetchone()[0]
        
        if null_count > 0:
            print(f"\nFound {null_count} metrics with missing/unknown phase")
            print("Attempting to infer phases from timestamps...")
            
            # Get time ranges for each phase
            # Assuming 3-phase structure based on timestamps
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM metrics
            """)
            min_time, max_time = cursor.fetchone()
            
            if min_time and max_time:
                duration = max_time - min_time
                
                # Split into phases based on duration
                # Phase 1: baseline (first 30%)
                baseline_end = min_time + (duration * 0.30)
                # Phase 2: level0_learning (next 30%)
                level0_end = baseline_end + (duration * 0.30)
                # Phase 3: level1_learning (next 30%)
                level1_end = level0_end + (duration * 0.30)
                # Phase 4: level2_learning (remaining)
                
                # Update phases
                cursor.execute("""
                    UPDATE metrics 
                    SET phase = 'baseline'
                    WHERE (phase IS NULL OR phase = '' OR phase = 'unknown')
                    AND timestamp < ?
                """, (baseline_end,))
                baseline_fixed = cursor.rowcount
                
                cursor.execute("""
                    UPDATE metrics 
                    SET phase = 'level0_learning'
                    WHERE (phase IS NULL OR phase = '' OR phase = 'unknown')
                    AND timestamp >= ? AND timestamp < ?
                """, (baseline_end, level0_end))
                level0_fixed = cursor.rowcount
                
                cursor.execute("""
                    UPDATE metrics 
                    SET phase = 'level1_learning'
                    WHERE (phase IS NULL OR phase = '' OR phase = 'unknown')
                    AND timestamp >= ? AND timestamp < ?
                """, (level0_end, level1_end))
                level1_fixed = cursor.rowcount
                
                cursor.execute("""
                    UPDATE metrics 
                    SET phase = 'level2_learning'
                    WHERE (phase IS NULL OR phase = '' OR phase = 'unknown')
                    AND timestamp >= ?
                """, (level1_end,))
                level2_fixed = cursor.rowcount
                
                conn.commit()
                
                print(f"  ✓ Fixed {baseline_fixed} baseline metrics")
                print(f"  ✓ Fixed {level0_fixed} level0_learning metrics")
                print(f"  ✓ Fixed {level1_fixed} level1_learning metrics")
                print(f"  ✓ Fixed {level2_fixed} level2_learning metrics")
        else:
            print("✓ No missing phases found")
        
        conn.close()
    
    def normalize_phase_names(self):
        """Normalize phase names (fix capitalization, whitespace)."""
        print("\n" + "="*70)
        print("NORMALIZING PHASE NAMES")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get all distinct phases
        cursor.execute("SELECT DISTINCT phase FROM metrics")
        phases = [row[0] for row in cursor.fetchall()]
        
        print(f"\nCurrent phases: {phases}")
        
        fixes = {
            'BASELINE': 'baseline',
            'Baseline': 'baseline',
            'baseline ': 'baseline',
            ' baseline': 'baseline',
            'LEVEL0_LEARNING': 'level0_learning',
            'Level0_learning': 'level0_learning',
            'level0_learning ': 'level0_learning',
            'LEVEL1_LEARNING': 'level1_learning',
            'Level1_learning': 'level1_learning',
            'level1_learning ': 'level1_learning',
            'LEVEL2_LEARNING': 'level2_learning',
            'Level2_learning': 'level2_learning',
            'level2_learning ': 'level2_learning',
        }
        
        total_fixed = 0
        for wrong, correct in fixes.items():
            cursor.execute("""
                UPDATE metrics 
                SET phase = ?
                WHERE phase = ?
            """, (correct, wrong))
            
            if cursor.rowcount > 0:
                print(f"  ✓ Fixed {cursor.rowcount} '{wrong}' → '{correct}'")
                total_fixed += cursor.rowcount
        
        if total_fixed == 0:
            print("✓ Phase names already normalized")
        else:
            conn.commit()
            print(f"\nTotal metrics fixed: {total_fixed}")
        
        conn.close()
    
    def verify_fixes(self):
        """Verify that fixes worked."""
        print("\n" + "="*70)
        print("VERIFICATION")
        print("="*70)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Check phase distribution
        cursor.execute("""
            SELECT phase, COUNT(*) as count 
            FROM metrics 
            GROUP BY phase 
            ORDER BY MIN(timestamp)
        """)
        
        phases = cursor.fetchall()
        
        print("\nPhase distribution after fixes:")
        for phase, count in phases:
            print(f"  {phase}: {count:,} metrics")
        
        # Check for remaining issues
        cursor.execute("""
            SELECT COUNT(*) FROM metrics 
            WHERE phase IS NULL OR phase = '' OR phase = 'unknown'
        """)
        remaining = cursor.fetchone()[0]
        
        if remaining > 0:
            print(f"\n⚠️  Warning: {remaining} metrics still have issues")
        else:
            print("\n✅ All metrics have valid phases!")
        
        conn.close()
    
    def run_all_fixes(self):
        """Run all fixes."""
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*20 + "TELEMETRY FIXER" + " "*33 + "║")
        print("╚" + "="*68 + "╝")
        
        if not self.backup_database():
            return
        
        self.normalize_phase_names()
        self.fix_missing_phases()
        self.verify_fixes()
        
        print("\n" + "="*70)
        print("✅ FIXES COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run: python debug_telemetry.py")
        print("  2. Verify phase queries return data")
        print("  3. Re-run demo if needed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix telemetry issues"
    )
    parser.add_argument(
        "--db",
        default="data/telemetry.db",
        help="Path to telemetry database"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation"
    )
    
    args = parser.parse_args()
    
    fixer = TelemetryFixer(args.db)
    
    if not args.no_backup:
        print("\n⚠️  This script will modify your database.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()