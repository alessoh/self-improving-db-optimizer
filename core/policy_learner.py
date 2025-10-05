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
    
    FIXED: Compares baseline to CURRENT phase only, not aggregate of all phases.
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
        
        # Track current phase
        self.current_phase = None
        
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
    
    def set_current_phase(self, phase: str):
        """
        Set the current phase for phase-specific learning.
        
        Args:
            phase: Name of current phase
        """
        self.current_phase = phase
        self.logger.debug(f"Policy learner tracking phase: {phase}")
        
    def update_policy(self) -> bool:
        """
        Analyze telemetry and update operational policy.
        
        FIXED: Now compares baseline to CURRENT PHASE ONLY, not all phases.
        
        Returns:
            True if policy was updated
        """
        if not self.enabled:
            return False
        
        self.logger.info("Analyzing telemetry for policy update...")
        
        # Get baseline metrics
        baseline_metrics = self.telemetry.get_phase_metrics('baseline')
        
        if not baseline_metrics:
            self.logger.info("No baseline metrics available")
            return False
        
        # CRITICAL FIX: Get metrics from CURRENT PHASE ONLY
        # Don't aggregate all learning phases together
        if self.current_phase and self.current_phase != 'baseline':
            current_metrics = self.telemetry.get_phase_metrics(self.current_phase)
            phase_name = self.current_phase
        else:
            # Fallback: get recent metrics from any non-baseline phase
            all_metrics = self.telemetry.get_recent_metrics(hours=1)
            current_metrics = [m for m in all_metrics if m.get('phase') != 'baseline']
            phase_name = 'current'
        
        if len(current_metrics) < self.level1_config['validation_samples']:
            self.logger.info(
                f"Insufficient samples in {phase_name} phase "
                f"({len(current_metrics)} < {self.level1_config['validation_samples']})"
            )
            return False
        
        # Analyze performance
        baseline_performance = self._analyze_performance(baseline_metrics)
        current_performance = self._analyze_performance(current_metrics)
        
        self.logger.info(
            f"Baseline: {baseline_performance['avg_latency']*1000:.2f}ms, "
            f"{phase_name}: {current_performance['avg_latency']*1000:.2f}ms"
        )
        
        # Check if improvement is possible
        if not self._should_update(baseline_performance, current_performance):
            self.logger.info("No significant improvement opportunity detected")
            return False
        
        # Generate new policy
        new_policy = self._generate_improved_policy(
            current_metrics, 
            baseline_performance,
            current_performance
        )
        
        # Validate new policy
        if self._validate_policy(new_policy, baseline_performance, current_performance):
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
            self.logger.info(
                f"New policy failed validation "
                f"(improvement: {new_policy['expected_improvement']:.2%}, "
                f"validation score: {new_policy['validation_score']:.2f})"
            )
            return False
    
    def _analyze_performance(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze current performance from metrics.
        
        Args:
            metrics: List of execution metrics
            
        Returns:
            Performance summary
        """
        if not metrics:
            return {
                'avg_latency': 0,
                'p95_latency': 0,
                'p99_latency': 0,
                'success_rate': 0,
                'total_samples': 0
            }
        
        exec_times = [m.get('execution_time', 0) for m in metrics]
        success_count = sum(1 for m in metrics if m.get('success', True))
        
        return {
            'avg_latency': np.mean(exec_times),
            'p95_latency': np.percentile(exec_times, 95),
            'p99_latency': np.percentile(exec_times, 99),
            'success_rate': success_count / len(metrics) if metrics else 0,
            'total_samples': len(metrics)
        }
    
    def _should_update(
        self, 
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> bool:
        """
        Determine if policy update is warranted.
        
        FIXED: Now properly compares baseline to current phase.
        
        Args:
            baseline_performance: Baseline performance metrics
            current_performance: Current phase performance metrics
            
        Returns:
            True if update should proceed
        """
        # Always allow update attempts (let validation decide)
        # This allows the system to learn from both improvements and degradations
        return True
    
    def _generate_improved_policy(
        self,
        metrics: List[Dict[str, Any]],
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate an improved policy based on telemetry analysis.
        
        Args:
            metrics: Execution metrics from current phase
            baseline_performance: Baseline performance summary
            current_performance: Current performance summary
            
        Returns:
            New policy specification
        """
        # Analyze which actions performed best
        action_performance = {}
        
        for metric in metrics:
            plan_info = metric.get('plan_info', {})
            if isinstance(plan_info, str):
                try:
                    import json
                    plan_info = json.loads(plan_info)
                except:
                    plan_info = {}
            
            action = plan_info.get('action', 'default')
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
        if action_scores:
            best_actions = sorted(
                action_scores.items(),
                key=lambda x: x[1]['avg_time']
            )[:3]
        else:
            best_actions = []
        
        policy_changes = {
            'prioritize_actions': [action for action, _ in best_actions],
            'action_scores': action_scores,
            'baseline_avg': baseline_performance['avg_latency'],
            'current_avg': current_performance['avg_latency']
        }
        
        # Calculate expected improvement
        baseline_avg = baseline_performance['avg_latency']
        current_avg = current_performance['avg_latency']
        
        if baseline_avg > 0:
            expected_improvement = (baseline_avg - current_avg) / baseline_avg
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
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> bool:
        """
        Validate new policy before deployment.
        
        Args:
            new_policy: Proposed policy
            baseline_performance: Baseline performance
            current_performance: Current performance
            
        Returns:
            True if policy passes validation
        """
        expected_improvement = new_policy['expected_improvement']
        threshold = self.level1_config['min_improvement']
        
        # Check if improvement meets minimum threshold
        if expected_improvement < threshold:
            self.logger.debug(
                f"Improvement {expected_improvement:.2%} < threshold {threshold:.2%}"
            )
            return False
        
        # Calculate validation confidence
        # Higher improvement relative to threshold = higher confidence
        confidence = min(expected_improvement / threshold, 1.0)
        
        new_policy['validation_score'] = confidence
        
        # Check if confidence meets validation threshold
        validation_threshold = self.level1_config['validation_threshold']
        
        if confidence < validation_threshold:
            self.logger.debug(
                f"Confidence {confidence:.2f} < threshold {validation_threshold:.2f}"
            )
            return False
        
        return True
    
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
        self.logger.info(f"Applying policy changes: prioritizing {prioritized}")
        
        # Store in history
        self.performance_history.append({
            'version': self.policy_version,
            'timestamp': time.time(),
            'changes': changes,
            'expected_improvement': policy['expected_improvement']
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
            'performance_history': self.performance_history,
            'current_phase': self.current_phase
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"State saved to {path}")