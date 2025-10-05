import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time

from core.models import PolicyNetwork


class PolicyLearner:
    """
    Level 1: Tactical Policy Learning
    Analyzes execution telemetry and updates operational policies.
    """
    
    def __init__(self, config: Dict[str, Any], query_optimizer, telemetry_storage):
        """
        Initialize policy learner.
        
        Args:
            config: System configuration
            query_optimizer: QueryOptimizer instance
            telemetry_storage: TelemetryStorage instance
        """
        self.config = config
        self.query_optimizer = query_optimizer
        self.telemetry = telemetry_storage
        self.logger = logging.getLogger(__name__)
        
        self.level1_config = config['level1']
        self.enabled = self.level1_config['enabled']
        
        # Policy version tracking
        self.policy_version = 0
        self.last_update = time.time()
        
        # Policy network for learning improved decision rules
        self.policy_network = PolicyNetwork(
            input_dim=64,  # Telemetry features
            output_dim=32,  # Policy parameters
            hidden_dims=[256, 128]
        )
        
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.level1_config['learning_rate']
        )
        
        # Performance history
        self.performance_history = []
        
        self.logger.info("Policy Learner (Level 1) initialized")
        
    def set_enabled(self, enabled: bool):
        """Enable or disable policy learning."""
        self.enabled = enabled
        self.logger.info(f"Policy learning {'enabled' if enabled else 'disabled'}")
        
    def update_policy(self) -> bool:
        """
        Analyze telemetry and update operational policy.
        
        Returns:
            True if policy was updated
        """
        if not self.enabled:
            return False
        
        self.logger.info("Analyzing telemetry for policy update...")
        
        # Get recent metrics since last update
        time_since_update = time.time() - self.last_update
        hours = time_since_update / 3600
        
        # Get metrics from ALL phases for cross-phase learning
        all_metrics = self.telemetry.get_recent_metrics(hours=24*7)  # Get all recent data
        baseline_metrics = [m for m in all_metrics if m.get('phase') == 'baseline']
        current_metrics = [m for m in all_metrics if m.get('phase') != 'baseline']
        metrics = current_metrics if current_metrics else all_metrics

        
        if len(metrics) < self.level1_config['validation_samples']:
            self.logger.info(f"Insufficient samples ({len(metrics)}) for update")
            return False
        
        # Analyze performance
        current_performance = self._analyze_performance(metrics)
        
        # Check if improvement is possible
        if not self._should_update(current_performance):
            self.logger.info("No significant improvement opportunity detected")
            return False
        
        # Generate new policy
        new_policy = self._generate_improved_policy(metrics, current_performance)
        
        # Validate new policy
        if self._validate_policy(new_policy, current_performance):
            self._apply_policy(new_policy)
            self.policy_version += 1
            self.last_update = time.time()
            
            # Record update
            self.telemetry.store_policy_update({
                'old_version': self.policy_version - 1,
                'new_version': self.policy_version,
                'improvement': new_policy['expected_improvement'],
                'validation_score': new_policy['validation_score'],
                'changes': new_policy['changes']
            })
            
            self.logger.info(
                f"Policy updated to version {self.policy_version} "
                f"(expected improvement: {new_policy['expected_improvement']:.2%})"
            )
            
            return True
        else:
            self.logger.info("New policy failed validation")
            return False
    
    def _analyze_performance(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze current performance from metrics.
        
        Args:
            metrics: List of execution metrics
            
        Returns:
            Performance summary
        """
        exec_times = [m.get('execution_time', 0) for m in metrics]
        success_count = sum(1 for m in metrics if m.get('success', True))
        
        return {
            'avg_latency': np.mean(exec_times),
            'p95_latency': np.percentile(exec_times, 95),
            'p99_latency': np.percentile(exec_times, 99),
            'success_rate': success_count / len(metrics) if metrics else 0,
            'total_samples': len(metrics)
        }
    
    def _should_update(self, current_performance: Dict[str, float]) -> bool:
        """
        Determine if policy update is warranted.
        
        Args:
            current_performance: Current performance metrics
            
        Returns:
            True if update should proceed
        """
        # Always update if this is the first time
        if not self.performance_history:
            return True
        
        # Check if performance has degraded
        prev_performance = self.performance_history[-1]
        
        latency_change = (
            (current_performance['avg_latency'] - prev_performance['avg_latency']) /
            prev_performance['avg_latency']
        )
        
        # Update if performance degraded or if enough time has passed
        min_improvement = self.level1_config['min_improvement']
        
        return latency_change > min_improvement or \
               time.time() - self.last_update > self.level1_config['update_interval'] * 2
    
    def _generate_improved_policy(
        self,
        metrics: List[Dict[str, Any]],
        current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate an improved policy based on telemetry analysis.
        
        Args:
            metrics: Execution metrics
            current_performance: Current performance summary
            
        Returns:
            New policy specification
        """
        # Analyze which actions performed best
        action_performance = {}
        
        for metric in metrics:
            action = metric.get('plan_info', {}).get('action', 'default')
            exec_time = metric.get('execution_time', 0)
            
            if action not in action_performance:
                action_performance[action] = []
            action_performance[action].append(exec_time)
        
        # Calculate average performance per action
        action_scores = {}
        for action, times in action_performance.items():
            action_scores[action] = {
                'avg_time': np.mean(times),
                'count': len(times)
            }
        
        # Determine policy adjustments
        # Increase probability of better-performing actions
        best_actions = sorted(
            action_scores.items(),
            key=lambda x: x[1]['avg_time']
        )[:3]
        
        policy_changes = {
            'prioritize_actions': [action for action, _ in best_actions],
            'action_scores': action_scores
        }
        
        # Estimate expected improvement
        if action_scores:
            best_avg = best_actions[0][1]['avg_time']
            current_avg = current_performance['avg_latency']
            expected_improvement = max(0, (current_avg - best_avg) / current_avg)
        else:
            expected_improvement = 0
        
        return {
            'changes': policy_changes,
            'expected_improvement': expected_improvement,
            'validation_score': 0.0  # To be filled by validation
        }
    
    def _validate_policy(
        self,
        new_policy: Dict[str, Any],
        current_performance: Dict[str, float]
    ) -> bool:
        """
        Validate new policy before deployment.
        
        Args:
            new_policy: Proposed policy
            current_performance: Current performance baseline
            
        Returns:
            True if policy passes validation
        """
        # Simple validation: check expected improvement threshold
        threshold = self.level1_config['min_improvement']
        
        if new_policy['expected_improvement'] < threshold:
            return False
        
        # In a full implementation, would run A/B test
        # For now, assign validation score based on confidence
        confidence = min(
            new_policy['expected_improvement'] / threshold,
            1.0
        )
        
        new_policy['validation_score'] = confidence
        
        return confidence >= self.level1_config['validation_threshold']
    
    def _apply_policy(self, policy: Dict[str, Any]):
        """
        Apply new policy to query optimizer.
        
        Args:
            policy: Policy to apply
        """
        # Update optimizer's action preferences
        changes = policy['changes']
        prioritized = changes.get('prioritize_actions', [])
        
        # This would modify the query optimizer's decision weights
        # For simplicity, just log the changes
        self.logger.info(f"Applying policy changes: {prioritized}")
        
        # Store in history
        self.performance_history.append({
            'version': self.policy_version,
            'timestamp': time.time(),
            'changes': changes
        })
    
    def rollback_policy(self):
        """Rollback to previous policy version."""
        if self.policy_version > 0:
            self.policy_version -= 1
            self.logger.warning(f"Rolled back to policy version {self.policy_version}")
            
            # Record rollback event
            self.telemetry.store_safety_event({
                'severity': 'warning',
                'event_type': 'policy_rollback',
                'description': 'Policy rolled back due to safety event',
                'action_taken': f'Reverted to version {self.policy_version}'
            })
    
    def save_state(self, path: Optional[Path] = None):
        """Save policy learner state."""
        if path is None:
            path = Path(self.config['paths']['policies_dir']) / 'level1_state.json'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'policy_version': self.policy_version,
            'last_update': self.last_update,
            'performance_history': self.performance_history
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"State saved to {path}")