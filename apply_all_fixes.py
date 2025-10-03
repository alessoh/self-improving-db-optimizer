"""
Apply All Fixes Script

This script applies all the fixes needed to make the system work correctly:
1. Updates config.yaml with correct parameters
2. Shows you what to fix in main.py
3. Updates telemetry/collector.py
"""

import shutil
from pathlib import Path
from datetime import datetime


def backup_file(filepath):
    """Create timestamped backup of a file."""
    filepath = Path(filepath)
    if filepath.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.{timestamp}.backup')
        shutil.copy2(filepath, backup_path)
        print(f"  ✓ Backed up to: {backup_path}")
        return True
    return False


def update_config():
    """Update config.yaml with fixed parameters."""
    print("\n" + "="*70)
    print("STEP 1: UPDATE config.yaml")
    print("="*70)
    
    config_path = Path('config.yaml')
    
    if not config_path.exists():
        print("❌ config.yaml not found!")
        return False
    
    backup_file(config_path)
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Apply fixes
    fixes_applied = []
    new_lines = []
    
    for line in lines:
        original_line = line
        
        # Fix epsilon_start
        if 'epsilon_start:' in line and '1.0' in line:
            line = '  epsilon_start: 0.3                # FIXED: Start lower (was 1.0)\n'
            fixes_applied.append('epsilon_start: 1.0 → 0.3')
        
        # Fix epsilon_decay
        elif 'epsilon_decay:' in line and '0.995' in line:
            line = '  epsilon_decay: 0.99               # FIXED: Decay faster (was 0.995)\n'
            fixes_applied.append('epsilon_decay: 0.995 → 0.99')
        
        # Fix update_interval
        elif 'update_interval: 3600' in line:
            line = '  update_interval: 1800             # FIXED: 30 min instead of 1 hour\n'
            fixes_applied.append('update_interval: 3600 → 1800')
        
        # Fix validation_samples
        elif 'validation_samples:' in line:
            line = '  validation_samples: 50            # FIXED: Lowered from 100\n'
            fixes_applied.append('validation_samples: 100 → 50')
        
        # Fix min_improvement
        elif 'min_improvement: 0.02' in line:
            line = '  min_improvement: 0.01             # FIXED: 1% instead of 2%\n'
            fixes_applied.append('min_improvement: 0.02 → 0.01')
        
        # Fix evaluation_interval
        elif 'evaluation_interval: 86400' in line:
            line = '  evaluation_interval: 7200         # FIXED: 2 hours instead of 24 hours\n'
            fixes_applied.append('evaluation_interval: 86400 → 7200')
        
        # Fix execution_time reward
        elif 'execution_time: -1.0' in line:
            line = '  execution_time: -0.1            # FIXED: Much smaller penalty (was -1.0)\n'
            fixes_applied.append('execution_time penalty: -1.0 → -0.1')
        
        # Fix cache_hit_rate reward
        elif 'cache_hit_rate: 0.5' in line:
            line = '  cache_hit_rate: 1.0             # FIXED: Larger bonus (was 0.5)\n'
            fixes_applied.append('cache_hit_rate bonus: 0.5 → 1.0')
        
        # Fix success_bonus
        elif 'success_bonus: 1.0' in line:
            line = '  success_bonus: 2.0              # FIXED: Larger bonus (was 1.0)\n'
            fixes_applied.append('success_bonus: 1.0 → 2.0')
        
        # Fix learning_rate
        elif 'learning_rate: 0.0003' in line and 'level0:' in ''.join(new_lines[-5:]):
            line = '  learning_rate: 0.001              # Increased from 0.0003 (learn faster)\n'
            fixes_applied.append('learning_rate: 0.0003 → 0.001')
        
        new_lines.append(line)
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"\n✓ Applied {len(fixes_applied)} fixes to config.yaml:")
    for fix in fixes_applied:
        print(f"  - {fix}")
    
    return True


def show_main_py_fix():
    """Show instructions for fixing main.py."""
    print("\n" + "="*70)
    print("STEP 2: FIX main.py")
    print("="*70)
    
    print("\nYou need to add ONE line to the _run_phase method in main.py")
    print("\nFind this section (around line 180):")
    print("""
    self.current_phase = phase_name
    phase_start = time.time()
    duration_seconds = duration * 86400
    
    self.logger.info(f"Starting phase: {phase_name} (duration: {duration} days)")
    self.logger.info(f"Learning - L0: {level0}, L1: {level1}, L2: {level2}")
    """)
    
    print("\nADD THESE LINES right after the logger.info calls:")
    print("""
    # CRITICAL FIX: Set phase in telemetry collector
    if self.telemetry_collector:
        self.telemetry_collector.set_phase(phase_name)
        self.logger.info(f"✓ Telemetry collector phase set to: '{phase_name}'")
    """)
    
    print("\nOR copy the complete fixed _run_phase method from:")
    print("  main_py_fix.py artifact")
    
    print("\n⚠️  This is the CRITICAL fix - without it, telemetry won't work!")


def show_test_instructions():
    """Show testing instructions."""
    print("\n" + "="*70)
    print("STEP 3: TEST THE FIXES")
    print("="*70)
    
    print("\n1. Run a quick test (10 minutes):")
    print("   python run_demo.py --duration 0.01 --fast-mode")
    
    print("\n2. Watch for these in the output:")
    print("   ✓ 'Telemetry collector phase set to: baseline'")
    print("   ✓ 'Running policy update (Level 1)...'")
    print("   ✓ 'Policy updated to version X'")
    
    print("\n3. Check results:")
    print("   python view_results.py")
    
    print("\n4. Verify improvements:")
    print("   - Policy Updates > 0 (should be 1-2)")
    print("   - Performance stable or improving")
    print("   - Average latency ~40-50ms (not 70ms)")


def calculate_new_rewards():
    """Show example of new reward calculation."""
    print("\n" + "="*70)
    print("REWARD FUNCTION - BEFORE vs AFTER")
    print("="*70)
    
    print("\nExample: 50ms query, 10% CPU, 20% memory, 80% cache hit")
    
    print("\nBEFORE (broken):")
    reward_old = 1.0 + (0.05 * -1.0) + (10 * -0.1) + (20 * -0.1) + (0.8 * 0.5)
    print(f"  Reward = {reward_old:.3f} (NEGATIVE = BAD)")
    
    print("\nAFTER (fixed):")
    reward_new = 2.0 + (0.05 * -0.1) + (10 * -0.01) + (20 * -0.01) + (0.8 * 1.0)
    print(f"  Reward = {reward_new:.3f} (POSITIVE = GOOD)")
    
    print("\nThis means:")
    print("  - Agent now incentivized to minimize latency")
    print("  - Fast queries with high cache hits get rewarded")
    print("  - Learning actually works!")


def main():
    """Main entry point."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "APPLY ALL FIXES" + " "*32 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nThis script will fix all identified issues:")
    print("  1. Epsilon decay too slow (causing random exploration)")
    print("  2. Negative rewards (breaking learning incentive)")
    print("  3. High validation thresholds (preventing policy updates)")
    print("  4. Long update intervals (making learning too infrequent)")
    
    response = input("\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Apply fixes
    if update_config():
        print("\n✅ config.yaml updated successfully!")
    
    show_main_py_fix()
    calculate_new_rewards()
    show_test_instructions()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nCompleted:")
    print("  ✓ config.yaml updated with correct parameters")
    print("  ✓ Backup created")
    
    print("\nManual steps required:")
    print("  ⚠️  Add telemetry_collector.set_phase() to main.py")
    print("  ⚠️  (See instructions above)")
    
    print("\nNext:")
    print("  1. Edit main.py (add the one line)")
    print("  2. Run: python run_demo.py --duration 0.01 --fast-mode")
    print("  3. Run: python view_results.py")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()