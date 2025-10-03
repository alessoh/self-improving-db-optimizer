import os
import sys
import time
import signal
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.query_optimizer import QueryOptimizer
from core.policy_learner import PolicyLearner
from core.meta_learner import MetaLearner
from core.safety_monitor import SafetyMonitor
from database.database_manager import DatabaseManager
from database.workload_generator import WorkloadGenerator
from telemetry.collector import TelemetryCollector
from telemetry.storage import TelemetryStorage
from utils.logger import setup_logger, get_logger
from utils.metrics import MetricsCalculator


class SystemOrchestrator:
    """Main orchestrator coordinating all system components."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the system orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False
        self.paused = False
        
        # Setup logging
        setup_logger(self.config)
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.db_manager = None
        self.workload_generator = None
        self.query_optimizer = None
        self.policy_learner = None
        self.meta_learner = None
        self.safety_monitor = None
        self.telemetry_collector = None
        self.telemetry_storage = None
        self.metrics_calculator = None
        
        # State tracking
        self.start_time = None
        self.current_phase = "initialization"
        self.stats = {
            "queries_executed": 0,
            "policies_updated": 0,
            "meta_learner_runs": 0,
            "safety_events": 0
        }
        
        self.logger.info("System Orchestrator initialized")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {config_path}")
            print("Please copy config.yaml.example to config.yaml and configure it.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error loading config: {e}")
            sys.exit(1)
            
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing system components...")
            
            # Create necessary directories
            self._create_directories()
            
            # Initialize database manager
            self.logger.info("Initializing database manager...")
            self.db_manager = DatabaseManager(self.config['database'])
            
            # Verify database connection
            if not self._verify_database_connection():
                raise Exception("Failed to connect to database")
            
            # Initialize telemetry storage
            self.logger.info("Initializing telemetry storage...")
            self.telemetry_storage = TelemetryStorage(self.config)
            
            # Initialize telemetry collector
            self.logger.info("Initializing telemetry collector...")
            self.telemetry_collector = TelemetryCollector(
                self.config,
                self.telemetry_storage
            )
            
            # Initialize workload generator
            self.logger.info("Initializing workload generator...")
            self.workload_generator = WorkloadGenerator(
                self.config['workload'],
                self.db_manager
            )
            
            # Initialize query optimizer (Level 0)
            self.logger.info("Initializing query optimizer (Level 0)...")
            self.query_optimizer = QueryOptimizer(
                self.config['level0'],
                self.db_manager,
                self.telemetry_collector
            )
            
            # Initialize policy learner (Level 1)
            if self.config['level1']['enabled']:
                self.logger.info("Initializing policy learner (Level 1)...")
                self.policy_learner = PolicyLearner(
                    self.config['level1'],
                    self.query_optimizer,
                    self.telemetry_storage
                )
            
            # Initialize meta-learner (Level 2)
            if self.config['level2']['enabled']:
                self.logger.info("Initializing meta-learner (Level 2)...")
                self.meta_learner = MetaLearner(
                    self.config['level2'],
                    self.policy_learner,
                    self.telemetry_storage
                )
            
            # Initialize safety monitor
            if self.config['safety']['enabled']:
                self.logger.info("Initializing safety monitor...")
                self.safety_monitor = SafetyMonitor(
                    self.config['safety'],
                    self.db_manager,
                    self.telemetry_collector
                )
            
            # Initialize metrics calculator
            self.logger.info("Initializing metrics calculator...")
            self.metrics_calculator = MetricsCalculator()
            
            # Register signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            raise
    
    def _verify_database_connection(self) -> bool:
        """Verify database connection is working."""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            self.db_manager.return_connection(conn)
            
            if result and result[0] == 1:
                self.logger.info("Database connection verified successfully")
                return True
            else:
                self.logger.error("Database connection test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Database connection verification failed: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories."""
        paths = self.config['paths']
        for key, path in paths.items():
            if key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created/verified directory: {path}")
    
    def start(self, duration_days: float = 14.0, fast_mode: bool = False):
        """
        Start the system and run for specified duration.
        
        Args:
            duration_days: Duration in days (can be fractional)
            fast_mode: Accelerate time for testing
        """
        try:
            self.running = True
            self.start_time = datetime.now()
            
            self.logger.info("="*70)
            self.logger.info(f"Starting system - Duration: {duration_days} days")
            self.logger.info(f"Fast mode: {fast_mode}")
            self.logger.info("="*70)
            
            # Calculate time scaling
            time_scale = 100 if fast_mode else 1
            duration_seconds = duration_days * 24 * 3600 / time_scale
            end_time = time.time() + duration_seconds
            
            # Main execution loop
            iteration = 0
            while self.running and time.time() < end_time:
                try:
                    if not self.paused:
                        self._execute_iteration(iteration)
                        iteration += 1
                    
                    # Sleep briefly to prevent CPU spinning
                    time.sleep(0.1 if fast_mode else 1.0)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
                    if self.safety_monitor:
                        self.safety_monitor.record_error(e)
            
            # Shutdown
            self.shutdown()
            
            self.logger.info("="*70)
            self.logger.info("System execution completed")
            self.logger.info(f"Total iterations: {iteration}")
            self.logger.info(f"Total queries: {self.stats['queries_executed']}")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Fatal error in system execution: {e}", exc_info=True)
            self.shutdown()
            raise
    
    def _execute_iteration(self, iteration: int):
        """Execute a single iteration of the system."""
        # Generate and execute queries
        queries = self.workload_generator.generate_batch(10)
        
        for query in queries:
            try:
                # Execute query through optimizer
                result = self.query_optimizer.execute_query(query)
                
                # Record metrics
                self.telemetry_collector.record_query_execution(
                    query=query,
                    latency=result['execution_time'],
                    success=result['success'],
                    phase=self.current_phase
                )
                
                self.stats['queries_executed'] += 1
                
                # Safety monitoring
                if self.safety_monitor:
                    self.safety_monitor.check_query_safety(result)
                
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                self.telemetry_collector.record_query_execution(
                    query=query,
                    latency=0,
                    success=False,
                    phase=self.current_phase
                )
        
        # Periodic policy updates (Level 1)
        if self.policy_learner and iteration % 100 == 0:
            try:
                self.policy_learner.update_policy()
                self.stats['policies_updated'] += 1
            except Exception as e:
                self.logger.error(f"Policy update failed: {e}")
        
        # Periodic meta-learning (Level 2)
        if self.meta_learner and iteration % 1000 == 0:
            try:
                self.meta_learner.optimize_hyperparameters()
                self.stats['meta_learner_runs'] += 1
            except Exception as e:
                self.logger.error(f"Meta-learning failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def pause(self):
        """Pause system execution."""
        self.paused = True
        self.logger.info("System paused")
    
    def resume(self):
        """Resume system execution."""
        self.paused = False
        self.logger.info("System resumed")
    
    def shutdown(self):
        """Shutdown all components gracefully."""
        self.logger.info("Shutting down system...")
        self.running = False
        
        try:
            # Save states
            if self.query_optimizer:
                self.logger.info("Saving query optimizer state...")
                self.query_optimizer.save_checkpoint()
                
            if self.policy_learner:
                self.logger.info("Saving policy learner state...")
                self.policy_learner.save_state()
                
            if self.meta_learner:
                self.logger.info("Saving meta-learner state...")
                self.meta_learner.save_state()
            
            # Flush telemetry
            if self.telemetry_collector:
                self.logger.info("Flushing telemetry data...")
                self.telemetry_collector.flush()
            
            # Disconnect database
            if self.db_manager:
                self.logger.info("Closing database connections...")
                self.db_manager.disconnect()
            
            self.logger.info("System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Self-Improving Database Query Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 2-week simulation
  python main.py --duration 14
  
  # Quick test (1 hour)
  python main.py --duration 0.04 --fast-mode
  
  # Custom configuration
  python main.py --config custom_config.yaml --duration 7
        """
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=14.0,
        help="Duration to run in days (can be fractional, default: 14.0)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run in fast mode for testing (100x time acceleration)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and start orchestrator
        print("\nInitializing System Orchestrator...")
        orchestrator = SystemOrchestrator(args.config)
        
        print("Initializing components...")
        orchestrator.initialize_components()
        
        print("Starting system execution...")
        orchestrator.start(
            duration_days=args.duration,
            fast_mode=args.fast_mode
        )
        
        print("\nSystem execution completed successfully")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()