import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

from core.models import DQNNetwork, ReplayBuffer, StateEncoder


class QueryOptimizer:
    """
    Level 0: Operational AI Agent
    Uses reinforcement learning to optimize query execution.
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, telemetry_collector):
        """
        Initialize query optimizer.
        
        Args:
            config: System configuration
            db_manager: DatabaseManager instance
            telemetry_collector: TelemetryCollector instance
        """
        self.config = config
        self.db_manager = db_manager
        self.telemetry = telemetry_collector
        self.logger = logging.getLogger(__name__)
        
        # Learning configuration
        self.level0_config = config['level0']
        self.learning_enabled = False
        
        # State and action spaces
        self.state_dim = 32  # Encoded state dimension
        self.action_dim = 8   # Number of optimization actions
        
        # Initialize networks
        self.policy_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.level0_config['hidden_layers'],
            activation=self.level0_config['activation']
        )
        
        self.target_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.level0_config['hidden_layers'],
            activation=self.level0_config['activation']
        )
        
        # Copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.level0_config['learning_rate']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.level0_config['buffer_size'])
        
        # State encoder
        self.state_encoder = StateEncoder(self.state_dim)
        
        # Training state
        self.epsilon = self.level0_config['epsilon_start']
        self.epsilon_end = self.level0_config['epsilon_end']
        self.epsilon_decay = self.level0_config['epsilon_decay']
        self.gamma = self.level0_config['gamma']
        self.batch_size = self.level0_config['batch_size']
        
        self.steps = 0
        self.episodes = 0
        self.target_update_freq = self.level0_config['target_update_freq']
        
        # Action mappings
        self.actions = [
            'use_index',
            'sequential_scan',
            'hash_join',
            'nested_loop_join',
            'merge_join',
            'parallel_execution',
            'increase_work_mem',
            'default'
        ]
        
        self.logger.info("Query Optimizer (Level 0) initialized")
        
    def set_learning_enabled(self, enabled: bool):
        """Enable or disable learning."""
        self.learning_enabled = enabled
        self.logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
        
    def execute_query(
        self,
        query: str,
        query_type: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute query with learned optimization.
        
        Args:
            query: SQL query string
            query_type: Type of query
            state: Current database state
            
        Returns:
            Execution results and metrics
        """
        # Encode state
        state_encoded = self.state_encoder.encode(state)
        
        # Select action
        if self.learning_enabled:
            action = self.policy_net.get_action(state_encoded, self.epsilon)
        else:
            action = self.action_dim - 1  # Default action
        
        # Apply optimization hints
        optimized_query = self._apply_optimization(query, action)
        
        # Execute query
        start_time = time.time()
        try:
            results, exec_time, resources = self.db_manager.execute_query(
                optimized_query,
                fetch=True
            )
            success = True
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            results = None
            exec_time = time.time() - start_time
            resources = {'cpu': 0, 'memory': 0, 'cache_hit_rate': 0}
            success = False
        
        # Get plan info
        plan_info = self.db_manager.get_query_plan(query)
        
        # Calculate reward
        reward = self._calculate_reward(exec_time, resources, success)
        
        # Store experience and learn
        if self.learning_enabled:
            # Get next state (same as current for now)
            next_state_encoded = state_encoded
            done = True  # Each query is an episode
            
            self.replay_buffer.push(
                state_encoded,
                action,
                reward,
                next_state_encoded,
                done
            )
            
            # Train if enough samples
            if len(self.replay_buffer) >= self.batch_size:
                self._train_step()
            
            # Update epsilon
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
            
            self.steps += 1
            
            # Update target network
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return {
            'results': results,
            'execution_time': exec_time,
            'resources': resources,
            'plan_info': plan_info,
            'action': self.actions[action],
            'reward': reward,
            'success': success
        }
    
    def _apply_optimization(self, query: str, action: int) -> str:
        """
        Apply SAFE optimizations based on action.
        
        FIXED: Properly handles SQL comments and query structure.
        
        Args:
            query: Original query
            action: Selected action index
            
        Returns:
            Safely optimized query
        """
        action_name = self.actions[action]
        
        # Build prefix statements (executed before query)
        prefix_statements = []
        
        if action_name == 'increase_work_mem':
            # Small work_mem increase is usually safe
            prefix_statements.append('SET LOCAL work_mem = "16MB";')
        elif action_name == 'parallel_execution':
            # Modest parallelism for larger queries only
            if 'JOIN' in query.upper() or 'GROUP BY' in query.upper():
                prefix_statements.append('SET LOCAL max_parallel_workers_per_gather = 2;')
        
        # Build query comment (must be on separate line)
        comment = None
        if action_name == 'use_index':
            comment = '-- Optimizer hint: prefer index scans'
        elif action_name not in ['increase_work_mem', 'parallel_execution', 'default']:
            comment = f'-- Action: {action_name}'
        
        # Construct final query
        parts = []
        
        # Add prefix statements
        if prefix_statements:
            parts.extend(prefix_statements)
        
        # Add comment on separate line
        if comment:
            parts.append(comment)
        
        # Add the actual query
        parts.append(query)
        
        # Join with newlines to ensure proper SQL structure
        return '\n'.join(parts)
    
    def _calculate_reward(
        self,
        execution_time: float,
        resources: Dict[str, float],
        success: bool
    ) -> float:
        """
        Calculate reward for the action taken.
        
        Args:
            execution_time: Query execution time
            resources: Resource usage metrics
            success: Whether query succeeded
            
        Returns:
            Reward value
        """
        if not success:
            return self.config['rewards']['failure_penalty']
        
        # Reward components
        reward = self.config['rewards']['success_bonus']
        
        # Penalize execution time
        time_penalty = execution_time * self.config['rewards']['execution_time']
        reward += time_penalty
        
        # Penalize resource usage
        cpu_penalty = resources.get('cpu', 0) * self.config['rewards']['cpu_usage']
        memory_penalty = resources.get('memory', 0) * self.config['rewards']['memory_usage']
        reward += cpu_penalty + memory_penalty
        
        # Reward cache hits
        cache_reward = resources.get('cache_hit_rate', 0) * self.config['rewards']['cache_hit_rate']
        reward += cache_reward
        
        return reward
    
    def _train_step(self):
        """Perform one training step."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save model checkpoint."""
        if path is None:
            path = Path(self.config['paths']['policies_dir']) / 'level0_checkpoint.pt'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        if not path.exists():
            self.logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        self.logger.info(f"Checkpoint loaded from {path}")