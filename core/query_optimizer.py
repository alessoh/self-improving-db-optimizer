"""
Enhanced Query Optimizer with Real PostgreSQL Actions

This module implements query optimization using Deep Q-Network (DQN) with
actions that actually affect PostgreSQL query execution through SET commands.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """Deep Q-Network for action-value estimation"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QueryOptimizer:
    """
    DQN-based query optimizer with real PostgreSQL control actions
    
    Actions now use SET commands that PostgreSQL respects:
    - force_index: Disables sequential scans
    - increase_work_mem: Allocates more memory for sorts/hashes
    - prefer_hash_join: Forces hash join over nested loop
    - parallel_query: Enables parallel execution
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, telemetry_collector):
        """
        Initialize query optimizer
        
        Args:
            config: System configuration
            db_manager: DatabaseManager instance
            telemetry_collector: TelemetryCollector instance
        """
        self.config = config
        self.db = db_manager
        self.telemetry = telemetry_collector
        self.logger = logging.getLogger(__name__)
        
        # Get level0 configuration
        self.level0_config = config.get('level0', {})
        self.learning_enabled = False
        
        # State and action spaces
        self.state_dim = 32  # Encoded state dimension
        self.action_dim = 8  # Number of optimization actions
        
        # Action definitions with real SET commands
        self.actions = [
            {
                'name': 'default',
                'description': 'Use PostgreSQL default planner settings',
                'setup_commands': [],
                'cleanup_commands': []
            },
            {
                'name': 'force_index',
                'description': 'Force index usage by disabling sequential scans',
                'setup_commands': ['SET enable_seqscan = off'],
                'cleanup_commands': ['SET enable_seqscan = on']
            },
            {
                'name': 'increase_work_mem_64',
                'description': 'Increase work_mem to 64MB for complex sorts/hashes',
                'setup_commands': ['SET work_mem = \'64MB\''],
                'cleanup_commands': ['SET work_mem = \'4MB\'']
            },
            {
                'name': 'increase_work_mem_128',
                'description': 'Increase work_mem to 128MB for very complex operations',
                'setup_commands': ['SET work_mem = \'128MB\''],
                'cleanup_commands': ['SET work_mem = \'4MB\'']
            },
            {
                'name': 'prefer_hash_join',
                'description': 'Prefer hash joins over nested loops',
                'setup_commands': [
                    'SET enable_nestloop = off',
                    'SET enable_hashjoin = on'
                ],
                'cleanup_commands': [
                    'SET enable_nestloop = on',
                    'SET enable_hashjoin = on'
                ]
            },
            {
                'name': 'prefer_merge_join',
                'description': 'Prefer merge joins when data is sorted',
                'setup_commands': [
                    'SET enable_nestloop = off',
                    'SET enable_mergejoin = on'
                ],
                'cleanup_commands': [
                    'SET enable_nestloop = on',
                    'SET enable_mergejoin = on'
                ]
            },
            {
                'name': 'parallel_query',
                'description': 'Enable parallel query execution',
                'setup_commands': [
                    'SET max_parallel_workers_per_gather = 4',
                    'SET parallel_tuple_cost = 0.05'
                ],
                'cleanup_commands': [
                    'SET max_parallel_workers_per_gather = 2',
                    'SET parallel_tuple_cost = 0.1'
                ]
            },
            {
                'name': 'aggressive_optimization',
                'description': 'Combine multiple optimizations for complex queries',
                'setup_commands': [
                    'SET work_mem = \'128MB\'',
                    'SET enable_seqscan = off',
                    'SET max_parallel_workers_per_gather = 4'
                ],
                'cleanup_commands': [
                    'SET work_mem = \'4MB\'',
                    'SET enable_seqscan = on',
                    'SET max_parallel_workers_per_gather = 2'
                ]
            }
        ]
        
        self.action_size = len(self.actions)
        self.state_size = self.state_dim
        
        # DQN components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get hidden layer sizes from config
        hidden_layers = self.level0_config.get('hidden_layers', [128, 128])
        hidden_size = hidden_layers[0] if hidden_layers else 128
        
        self.policy_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Learning parameters
        self.learning_rate = self.level0_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.gamma = self.level0_config.get('gamma', 0.95)
        self.epsilon = self.level0_config.get('epsilon_start', 1.0)
        self.epsilon_decay = self.level0_config.get('epsilon_decay', 0.990)
        self.epsilon_min = self.level0_config.get('epsilon_end', 0.05)
        
        # Experience replay
        self.memory = deque(maxlen=self.level0_config.get('buffer_size', 10000))
        self.batch_size = self.level0_config.get('batch_size', 64)
        
        # Target network update frequency
        self.target_update_freq = self.level0_config.get('target_update_freq', 100)
        
        # Training state
        self.steps = 0
        self.episodes = 0
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'avg_loss': 0.0,
            'action_counts': {action['name']: 0 for action in self.actions}
        }
        
        self.logger.info("Query Optimizer (Level 0) initialized")
        
    def set_learning_enabled(self, enabled: bool):
        """Enable or disable learning"""
        self.learning_enabled = enabled
        self.logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
        
    def execute_query(self, query: str, query_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query with learned optimization
        
        This method provides compatibility with the existing interface
        
        Args:
            query: SQL query to execute
            query_type: Type of query
            state: Query state information
            
        Returns:
            Dictionary with execution results
        """
        # Use optimize_query for the actual work
        result = self.optimize_query(query, training=self.learning_enabled)
        
        # Format result to match expected interface
        return {
            'execution_time': result['execution_time'],
            'success': result['success'],
            'action': result['action'],
            'reward': result['reward'],
            'error': result.get('error')
        }
        
    def extract_features(self, query: str) -> np.ndarray:
        """
        Extract numerical features from query
        
        Features include:
        - Query complexity indicators
        - Table and join counts
        - Estimated selectivity
        - Current system state
        """
        # Return a fixed-size feature vector
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        query_lower = query.lower()
        
        # Feature 0-3: Query type indicators
        features[0] = 1.0 if 'select' in query_lower else 0.0
        features[1] = 1.0 if 'join' in query_lower else 0.0
        features[2] = 1.0 if 'group by' in query_lower else 0.0
        features[3] = 1.0 if 'order by' in query_lower else 0.0
        
        # Feature 4-6: Complexity indicators
        features[4] = min(float(query_lower.count('join')) / 5.0, 1.0)
        features[5] = min(float(query_lower.count('where')) / 3.0, 1.0)
        features[6] = min(float(query_lower.count('and') + query_lower.count('or')) / 5.0, 1.0)
        
        # Feature 7-9: Aggregation indicators
        features[7] = 1.0 if any(agg in query_lower for agg in ['count(', 'sum(', 'avg(', 'max(', 'min(']) else 0.0
        features[8] = 1.0 if 'distinct' in query_lower else 0.0
        features[9] = 1.0 if '(' in query_lower else 0.0
        
        # Feature 10-14: Additional characteristics
        features[10] = min(float(len(query)) / 1000.0, 1.0)  # Query length (normalized)
        features[11] = min(float(query_lower.count('from')) / 5.0, 1.0)  # Table count
        features[12] = 1.0 if 'limit' in query_lower else 0.0
        features[13] = 1.0 if 'union' in query_lower else 0.0
        features[14] = 1.0 if 'subquery' in query_lower else 0.0
       
        # Remaining features: system state placeholders
        for i in range(15, self.state_dim):
            features[i] = np.random.random() * 0.1
        
        return features
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy
        
        Args:
            state: Feature vector
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            action = random.randrange(self.action_size)
            self.logger.debug(f"Exploration: selected random action {self.actions[action]['name']}")
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
                self.logger.debug(f"Exploitation: selected action {self.actions[action]['name']}")
        
        self.stats['action_counts'][self.actions[action]['name']] += 1
        return action
        
    def execute_with_action(self, query: str, action_idx: int) -> Tuple[float, bool, Optional[str]]:
        """
        Execute query with specified action
        
        Args:
            query: SQL query
            action_idx: Index of action to apply
            
        Returns:
            Tuple of (execution_time, success, error_message)
        """
        action = self.actions[action_idx]
        conn = None
        cursor = None
        error_msg = None
        
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Apply SET commands for this action
            for cmd in action['setup_commands']:
                cursor.execute(cmd)
                self.logger.debug(f"Applied: {cmd}")
            
            # Execute the actual query with timing
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            execution_time = time.time() - start_time
            
            # Cleanup: restore default settings
            for cmd in action['cleanup_commands']:
                cursor.execute(cmd)
                self.logger.debug(f"Cleaned up: {cmd}")
            
            conn.commit()
            
            self.logger.info(f"Query executed with action '{action['name']}' in {execution_time*1000:.2f}ms")
            return execution_time, True, None
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Query execution failed with action '{action['name']}': {error_msg}")
            
            # Attempt cleanup even on error
            if cursor:
                try:
                    for cmd in action['cleanup_commands']:
                        cursor.execute(cmd)
                    if conn:
                        conn.commit()
                except:
                    pass
            
            return 0.0, False, error_msg
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.db.return_connection(conn)
                
    def calculate_reward(self, execution_time: float, success: bool, baseline_time: float = 0.05) -> float:
        """
        Calculate reward for an execution
        
        Args:
            execution_time: Time taken to execute query (seconds)
            success: Whether execution succeeded
            baseline_time: Expected baseline time
            
        Returns:
            Reward value
        """
        if not success:
            return -10.0  # Large penalty for failures
        
        # Time-based reward
        time_reward = -20.0 * (execution_time / baseline_time - 1.0)
        
        # Bonus for fast execution
        if execution_time < baseline_time * 0.8:
            time_reward += 5.0
        
        # Cap rewards
        time_reward = max(-25.0, min(25.0, time_reward))
        
        return time_reward
        
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self) -> float:
        """
        Perform one training step using experience replay
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Updated target network")
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.logger.debug(f"Epsilon decayed to {self.epsilon:.4f}")
            
    def optimize_query(self, query: str, training: bool = True, baseline_time: float = 0.05) -> Dict:
        """
        Main optimization loop for a query
        
        Args:
            query: SQL query to optimize
            training: Whether to train the network
            baseline_time: Expected baseline execution time
            
        Returns:
            Dictionary with execution results
        """
        # Extract features
        state = self.extract_features(query)
        
        # Select action
        action = self.select_action(state, training=training)
        
        # Execute query with action
        execution_time, success, error_msg = self.execute_with_action(query, action)
        
        # Calculate reward
        reward = self.calculate_reward(execution_time, success, baseline_time)
        
        # Get next state
        next_state = state
        done = True
        
        # Store experience if training
        if training:
            self.store_experience(state, action, reward, next_state, done)
            
            # Train if enough experiences
            loss = self.train_step()
            
            # Update statistics
            self.stats['episodes'] += 1
            self.stats['total_reward'] += reward
            self.stats['avg_loss'] = loss
            self.steps += 1
            self.episodes += 1
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Periodically update target network
            if self.episodes % self.target_update_freq == 0:
                self.update_target_network()
        
        return {
            'execution_time': execution_time,
            'success': success,
            'error': error_msg,
            'action': self.actions[action]['name'],
            'reward': reward,
            'epsilon': self.epsilon
        }
        
    def get_stats(self) -> Dict:
        """Return current statistics"""
        return self.stats.copy()
        
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': self.stats
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.stats = checkpoint['stats']
        self.logger.info(f"Model loaded from {path}")