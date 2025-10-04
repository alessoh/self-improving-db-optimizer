import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telemetry.storage import TelemetryStorage


def test_metrics_deserialization():
    """Test that metrics are properly deserialized."""
    print("\n" + "="*70)
    print("TEST 1: Metrics JSON Deserialization")
    print("="*70)
    
    # Initialize storage
    config = {
        'paths': {
            'telemetry_db': 'data/telemetry_test.db'
        }
    }
    storage = TelemetryStorage(config)
    
    # Create test metric with complex plan_info
    test_metric = {
        'timestamp': time.time(),
        'phase': 'test',
        'query_type': 'SELECT',
        'execution_time': 0.123,
        'cpu_usage': 45.6,
        'memory_usage': 67.8,
        'cache_hit_rate': 0.89,
        'rows_processed': 1000,
        'plan_cost': 123.45,
        'success': True,
        'query_hash': 'abc123',
        'plan_info': {
            'join_type': 'nested_loop',
            'index_used': True,
            'estimated_rows': 5000,
            'nested_data': {
                'level1': 'value1',
                'level2': ['item1', 'item2']
            }
        }
    }
    
    # Store the metric
    print("Storing test metric with complex plan_info...")
    storage.store_metric(test_metric)
    
    # Retrieve it
    print("Retrieving metric...")
    metrics = storage.get_recent_metrics(hours=1)
    
    if not metrics:
        print("❌ FAILED: No metrics retrieved")
        return False
    
    retrieved = metrics[-1]  # Get the last one (our test metric)
    
    # Verify plan_info is a dict, not a string
    print(f"\nType of plan_info: {type(retrieved['plan_info'])}")
    
    if isinstance(retrieved['plan_info'], str):
        print("❌ FAILED: plan_info is still a string!")
        print(f"   Value: {retrieved['plan_info']}")
        return False
    
    if not isinstance(retrieved['plan_info'], dict):
        print(f"❌ FAILED: plan_info is {type(retrieved['plan_info'])}, expected dict")
        return False
    
    # Verify we can access nested data
    try:
        join_type = retrieved['plan_info']['join_type']
        nested_value = retrieved['plan_info']['nested_data']['level1']
        print(f"✓ Successfully accessed join_type: {join_type}")
        print(f"✓ Successfully accessed nested value: {nested_value}")
    except (KeyError, AttributeError, TypeError) as e:
        print(f"❌ FAILED: Cannot access nested data: {e}")
        return False
    
    # Verify we can call .get() on the dict
    try:
        index_used = retrieved['plan_info'].get('index_used')
        print(f"✓ Successfully used .get() method: index_used={index_used}")
    except AttributeError as e:
        print(f"❌ FAILED: Cannot use .get() method: {e}")
        return False
    
    print("\n✓ TEST 1 PASSED: Metrics deserialization works correctly")
    return True


def test_policy_updates_deserialization():
    """Test that policy updates are properly deserialized."""
    print("\n" + "="*70)
    print("TEST 2: Policy Updates JSON Deserialization")
    print("="*70)
    
    config = {
        'paths': {
            'telemetry_db': 'data/telemetry_test.db'
        }
    }
    storage = TelemetryStorage(config)
    
    # Create test policy update
    test_update = {
        'timestamp': time.time(),
        'old_version': 1,
        'new_version': 2,
        'improvement': 0.15,
        'validation_score': 0.92,
        'changes': {
            'prioritize_actions': ['use_index', 'parallel_scan'],
            'threshold_adjustments': {
                'min_cost': 100,
                'max_parallelism': 4
            }
        }
    }
    
    print("Storing test policy update...")
    storage.store_policy_update(test_update)
    
    print("Retrieving policy updates...")
    updates = storage.get_policy_updates(limit=10)
    
    if not updates:
        print("❌ FAILED: No policy updates retrieved")
        return False
    
    retrieved = updates[0]  # Get the most recent
    
    print(f"\nType of changes: {type(retrieved['changes'])}")
    
    if isinstance(retrieved['changes'], str):
        print("❌ FAILED: changes is still a string!")
        return False
    
    if not isinstance(retrieved['changes'], dict):
        print(f"❌ FAILED: changes is {type(retrieved['changes'])}, expected dict")
        return False
    
    try:
        actions = retrieved['changes'].get('prioritize_actions')
        print(f"✓ Successfully accessed prioritize_actions: {actions}")
    except AttributeError as e:
        print(f"❌ FAILED: Cannot use .get() method: {e}")
        return False
    
    print("\n✓ TEST 2 PASSED: Policy updates deserialization works correctly")
    return True


def test_safety_events_deserialization():
    """Test that safety events are properly deserialized."""
    print("\n" + "="*70)
    print("TEST 3: Safety Events JSON Deserialization")
    print("="*70)
    
    config = {
        'paths': {
            'telemetry_db': 'data/telemetry_test.db'
        }
    }
    storage = TelemetryStorage(config)
    
    # Create test safety event
    test_event = {
        'timestamp': time.time(),
        'severity': 'warning',
        'event_type': 'performance_degradation',
        'description': 'Query latency increased',
        'action_taken': 'Monitoring',
        'context': {
            'affected_queries': ['query1', 'query2'],
            'performance_drop': 0.25,
            'metrics': {
                'before': 100,
                'after': 125
            }
        }
    }
    
    print("Storing test safety event...")
    storage.store_safety_event(test_event)
    
    print("Retrieving safety events...")
    events = storage.get_safety_events(limit=10)
    
    if not events:
        print("❌ FAILED: No safety events retrieved")
        return False
    
    retrieved = events[0]
    
    print(f"\nType of context: {type(retrieved['context'])}")
    
    if isinstance(retrieved['context'], str):
        print("❌ FAILED: context is still a string!")
        return False
    
    if not isinstance(retrieved['context'], dict):
        print(f"❌ FAILED: context is {type(retrieved['context'])}, expected dict")
        return False
    
    try:
        drop = retrieved['context'].get('performance_drop')
        print(f"✓ Successfully accessed performance_drop: {drop}")
    except AttributeError as e:
        print(f"❌ FAILED: Cannot use .get() method: {e}")
        return False
    
    print("\n✓ TEST 3 PASSED: Safety events deserialization works correctly")
    return True


def test_meta_learning_deserialization():
    """Test that meta-learning events are properly deserialized."""
    print("\n" + "="*70)
    print("TEST 4: Meta-Learning Events JSON Deserialization")
    print("="*70)
    
    config = {
        'paths': {
            'telemetry_db': 'data/telemetry_test.db'
        }
    }
    storage = TelemetryStorage(config)
    
    # Create test meta-learning event
    test_event = {
        'timestamp': time.time(),
        'generation': 5,
        'best_fitness': 0.875,
        'avg_fitness': 0.723,
        'hyperparameters': {
            'learning_rate': 0.002,
            'batch_size': 128,
            'gamma': 0.95,
            'epsilon_decay': 0.995
        },
        'improvements': {
            'best_fitness': 12.5,
            'avg_fitness': 8.3
        }
    }
    
    print("Storing test meta-learning event...")
    storage.store_meta_learning_event(test_event)
    
    print("Retrieving meta-learning events...")
    events = storage.get_meta_learning_events(limit=10)
    
    if not events:
        print("❌ FAILED: No meta-learning events retrieved")
        return False
    
    retrieved = events[0]
    
    print(f"\nType of hyperparameters: {type(retrieved['hyperparameters'])}")
    print(f"Type of improvements: {type(retrieved['improvements'])}")
    
    if isinstance(retrieved['hyperparameters'], str):
        print("❌ FAILED: hyperparameters is still a string!")
        return False
    
    if isinstance(retrieved['improvements'], str):
        print("❌ FAILED: improvements is still a string!")
        return False
    
    try:
        lr = retrieved['hyperparameters'].get('learning_rate')
        improvement = retrieved['improvements'].get('best_fitness')
        print(f"✓ Successfully accessed learning_rate: {lr}")
        print(f"✓ Successfully accessed best_fitness improvement: {improvement}")
    except AttributeError as e:
        print(f"❌ FAILED: Cannot use .get() method: {e}")
        return False
    
    print("\n✓ TEST 4 PASSED: Meta-learning events deserialization works correctly")
    return True


def test_batch_operations():
    """Test that batch operations also work correctly."""
    print("\n" + "="*70)
    print("TEST 5: Batch Metrics Storage and Retrieval")
    print("="*70)
    
    config = {
        'paths': {
            'telemetry_db': 'data/telemetry_test.db'
        }
    }
    storage = TelemetryStorage(config)
    
    # Create batch of metrics
    batch = []
    for i in range(5):
        batch.append({
            'timestamp': time.time() + i,
            'phase': 'batch_test',
            'query_type': f'SELECT_{i}',
            'execution_time': 0.1 + i * 0.01,
            'success': True,
            'plan_info': {
                'batch_number': i,
                'test_data': f'value_{i}'
            }
        })
    
    print(f"Storing batch of {len(batch)} metrics...")
    storage.store_metrics_batch(batch)
    
    print("Retrieving batch metrics...")
    retrieved_batch = storage.get_phase_metrics('batch_test')
    
    if len(retrieved_batch) < len(batch):
        print(f"❌ FAILED: Expected {len(batch)} metrics, got {len(retrieved_batch)}")
        return False
    
    # Check each metric in the batch
    for metric in retrieved_batch[-5:]:  # Check last 5
        if isinstance(metric['plan_info'], str):
            print("❌ FAILED: plan_info is still a string in batch!")
            return False
        
        if not isinstance(metric['plan_info'], dict):
            print(f"❌ FAILED: plan_info is {type(metric['plan_info'])}, expected dict")
            return False
    
    print(f"✓ All {len(batch)} metrics properly deserialized")
    print("\n✓ TEST 5 PASSED: Batch operations work correctly")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("TELEMETRY STORAGE JSON DESERIALIZATION TEST SUITE")
    print("="*70)
    
    tests = [
        test_metrics_deserialization,
        test_policy_updates_deserialization,
        test_safety_events_deserialization,
        test_meta_learning_deserialization,
        test_batch_operations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ TEST EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nThe JSON deserialization fix is working correctly.")
        print("You can now replace the old telemetry/storage.py with the fixed version.")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("\nPlease review the errors above and fix the issues.")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)