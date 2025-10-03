"""
Check why learning didn't activate by examining logs and config
"""

import yaml
from pathlib import Path


def check_learning_config():
    """Check learning configuration."""
    print("\n" + "="*70)
    print("LEARNING CONFIGURATION CHECK")
    print("="*70)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nLevel 0 (DQN) Configuration:")
    print(f"  Learning Rate:     {config['level0']['learning_rate']}")
    print(f"  Epsilon Start:     {config['level0']['epsilon_start']}")
    print(f"  Epsilon End:       {config['level0']['epsilon_end']}")
    print(f"  Epsilon Decay:     {config['level0']['epsilon_decay']}")
    print(f"  Batch Size:        {config['level0']['batch_size']}")
    print(f"  Buffer Size:       {config['level0']['buffer_size']}")
    
    print("\nLevel 1 (Policy Learning) Configuration:")
    print(f"  Enabled:           {config['level1']['enabled']}")
    print(f"  Update Interval:   {config['level1']['update_interval']} seconds ({config['level1']['update_interval']/3600:.1f} hours)")
    print(f"  Validation Samples: {config['level1'].get('validation_samples', 100)}")
    print(f"  Min Improvement:   {config['level1'].get('min_improvement', 0.02)*100:.1f}%")
    
    print("\nLevel 2 (Meta-Learning) Configuration:")
    print(f"  Enabled:           {config['level2']['enabled']}")
    print(f"  Evaluation Interval: {config['level2']['evaluation_interval']} seconds ({config['level2']['evaluation_interval']/3600:.1f} hours)")
    
    # Calculate how many times policy should have updated
    demo_duration_seconds = 4.62 * 3600  # 4.62 hours
    update_interval = config['level1']['update_interval']
    expected_updates = int(demo_duration_seconds / update_interval)
    
    print("\n" + "="*70)
    print("EXPECTED vs ACTUAL")
    print("="*70)
    print(f"\nDemo Duration: 4.62 hours")
    print(f"Policy Update Interval: {update_interval/3600:.1f} hours")
    print(f"Expected Policy Updates: {expected_updates}")
    print(f"Actual Policy Updates: 0")
    
    if expected_updates > 0:
        print(f"\n❌ Policy learner should have run {expected_updates} times but didn't!")
    
    # Check if validation_samples is the issue
    queries_per_phase = [9270, 5384, 10766, 20972]
    queries_per_hour = sum(queries_per_phase) / 4.62
    queries_per_update = queries_per_hour * (update_interval / 3600)
    
    validation_samples = config['level1'].get('validation_samples', 100)
    
    print(f"\nQueries per hour: ~{queries_per_hour:.0f}")
    print(f"Queries per update interval: ~{queries_per_update:.0f}")
    print(f"Validation samples required: {validation_samples}")
    
    if queries_per_update < validation_samples:
        print(f"\n⚠️  WARNING: Not enough queries between updates!")
        print(f"   Need {validation_samples} but only getting ~{queries_per_update:.0f}")
        print(f"   Solution: Lower validation_samples to {int(queries_per_update * 0.8)}")
    else:
        print(f"\n✓ Sufficient queries per update interval")


def check_log_for_learning():
    """Check logs for learning attempts."""
    print("\n" + "="*70)
    print("LOG FILE ANALYSIS")
    print("="*70)
    
    log_file = Path('data/logs/optimizer.log')
    
    if not log_file.exists():
        print("❌ Log file not found!")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Search for key phrases
    policy_attempts = [l for l in lines if 'Running policy update' in l]
    policy_insufficient = [l for l in lines if 'Insufficient samples' in l]
    policy_success = [l for l in lines if 'Policy updated to version' in l]
    meta_attempts = [l for l in lines if 'Running meta-learning' in l]
    learning_enabled = [l for l in lines if 'Learning enabled' in l]
    learning_disabled = [l for l in lines if 'Learning disabled' in l]
    
    print(f"\nLog file: {log_file}")
    print(f"Total lines: {len(lines):,}")
    
    print(f"\nLearning Status Messages:")
    print(f"  'Learning enabled':       {len(learning_enabled)}")
    print(f"  'Learning disabled':      {len(learning_disabled)}")
    
    print(f"\nPolicy Update Attempts:")
    print(f"  'Running policy update':  {len(policy_attempts)}")
    print(f"  'Insufficient samples':   {len(policy_insufficient)}")
    print(f"  'Policy updated':         {len(policy_success)}")
    
    print(f"\nMeta-Learning Attempts:")
    print(f"  'Running meta-learning':  {len(meta_attempts)}")
    
    if len(policy_insufficient) > 0:
        print(f"\n⚠️  Found {len(policy_insufficient)} 'Insufficient samples' messages")
        print("   Showing last 3:")
        for line in policy_insufficient[-3:]:
            print(f"   {line.strip()}")
    
    if len(policy_attempts) == 0:
        print("\n❌ CRITICAL: Policy learner was NEVER called!")
        print("   This means the update interval check isn't working")
        print("   or policy_learner.update_policy() isn't being invoked")


def check_rewards():
    """Check reward function values."""
    print("\n" + "="*70)
    print("REWARD FUNCTION CHECK")
    print("="*70)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    rewards = config['rewards']
    
    print("\nReward Function Weights:")
    print(f"  Execution time penalty:  {rewards['execution_time']}")
    print(f"  Memory usage penalty:    {rewards['memory_usage']}")
    print(f"  CPU usage penalty:       {rewards['cpu_usage']}")
    print(f"  Cache hit bonus:         {rewards['cache_hit_rate']}")
    print(f"  Success bonus:           {rewards['success_bonus']}")
    print(f"  Failure penalty:         {rewards['failure_penalty']}")
    
    # Simulate a typical query reward
    print("\nExample Reward Calculation:")
    print("  Typical query: 50ms execution, 10% CPU, 20% memory, 80% cache hit")
    
    exec_time = 0.05  # 50ms
    reward = rewards['success_bonus']
    reward += exec_time * rewards['execution_time']
    reward += 10 * rewards['cpu_usage']
    reward += 20 * rewards['memory_usage']
    reward += 0.8 * rewards['cache_hit_rate']
    
    print(f"  Total reward: {reward:.3f}")
    
    if reward < 0:
        print("  ⚠️  Typical queries get NEGATIVE rewards!")
        print("      This means the agent is punished for every query")
        print("      Consider adjusting reward weights")


def main():
    """Main entry point."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*18 + "LEARNING DIAGNOSTIC" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    
    check_learning_config()
    check_log_for_learning()
    check_rewards()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. Fix epsilon decay:")
    print("   Change config.yaml:")
    print("     epsilon_start: 0.3    # Start with less exploration")
    print("     epsilon_decay: 0.99   # Decay faster")
    
    print("\n2. Lower validation threshold:")
    print("   Change config.yaml:")
    print("     validation_samples: 50   # From 100")
    print("     update_interval: 1800    # From 3600 (30 min)")
    
    print("\n3. Check policy learner is being called:")
    print("   Add debug logging in main.py _run_phase()")
    
    print("\n4. Run shorter test with DEBUG logging:")
    print("   python run_demo.py --duration 0.01 --fast-mode --log-level DEBUG")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()