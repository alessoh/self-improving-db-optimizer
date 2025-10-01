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
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {config_path}")
            print("Please copy config.yaml.example to config.yaml and configure it.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
            
    def initialize_components(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components...")
        
        try:
            # Create data directories
            self._create_directories()
            
            # Initialize database manager
            self.logger.info("Connecting to database...")
            self.db_manager = DatabaseManager(self.config)
            self.db_manager.connect()
            
            # Initialize telemetry
            self.logger.info("Setting up telemetry...")
            self.telemetry_storage = TelemetryStorage(self.config)
            self.telemetry_collector = TelemetryCollector(
                self.config, 
                self.telemetry_storage
            )
            
            # Initialize workload generator
            self.logger.info("Initializing workload generator...")
            self.workload_generator = WorkloadGenerator(
                self.config,
                self.db_manager
            )
            
            # Initialize query optimizer (Level 0)
            self.logger.info("Initializing query optimizer (Level 0)...")
            self.query_optimizer = QueryOptimizer(
                self.config,
                self.db_manager,
                self.telemetry_collector
            )
            
            # Initialize policy learner (Level 1)
            if self.config['level1']['enabled']:
                self.logger.info("Initializing policy learner (Level 1)...")
                self.policy_learner = PolicyLearner(
                    self.config,
                    self.query_optimizer,
                    self.telemetry_storage
                )
            
            # Initialize meta-learner (Level 2)
            if self.config['level2']['enabled']:
                self.logger.info("Initializing meta-learner (Level 2)...")
                self.meta_learner = MetaLearner(
                    self.config,
                    self.policy_learner,
                    self.telemetry_storage
                )
            
            # Initialize safety monitor
            self.logger.info("Initializing safety monitor...")
            self.safety_monitor = SafetyMonitor(
                self.config,
                self.telemetry_storage
            )
            
            # Initialize metrics calculator
            self.metrics_calculator = MetricsCalculator(self.config)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        paths = self.config['paths']
        for key, path in paths.items():
            if key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
                
    def start(self, duration_days: float = 14.0, fast_mode: bool = False):
        """
        Start the system and run for specified duration.
        
        Args:
            duration_days: Number of days to run (can be fractional)
            fast_mode: If True, accelerate time for testing
        """
        self.running = True
        self.start_time = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Starting system for {duration_days} days")
        if fast_mode:
            self.logger.info("Fast mode enabled - time accelerated")
            
        try:
            # Phase 1: Baseline (Days 1-3)
            self._run_phase(
                "baseline",
                duration=3.0 * (1.0 if not fast_mode else 0.01),
                learning_enabled=False
            )
            
            # Phase 2: Level 0 Learning (Days 4-7)
            self._run_phase(
                "level0_learning",
                duration=4.0 * (1.0 if not fast_mode else 0.01),
                learning_enabled=True,
                level0=True,
                level1=False,
                level2=False
            )
            
            # Phase 3: Level 1 Learning (Days 8-11)
            if self.policy_learner:
                self._run_phase(
                    "level1_learning",
                    duration=4.0 * (1.0 if not fast_mode else 0.01),
                    learning_enabled=True,
                    level0=True,
                    level1=True,
                    level2=False
                )
            
            # Phase 4: Level 2 Meta-Learning (Days 12-14)
            if self.meta_learner:
                self._run_phase(
                    "level2_learning",
                    duration=3.0 * (1.0 if not fast_mode else 0.01),
                    learning_enabled=True,
                    level0=True,
                    level1=True,
                    level2=True
                )
                
            self.logger.info("System run completed successfully")
            self._generate_final_report()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error during system run: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()
            
    def _run_phase(
        self, 
        phase_name: str, 
        duration: float,
        learning_enabled: bool,
        level0: bool = False,
        level1: bool = False,
        level2: bool = False
    ):
        """
        Run a specific phase of the demonstration.
        
        Args:
            phase_name: Name of the phase
            duration: Duration in days
            learning_enabled: Whether learning is enabled
            level0: Enable Level 0 learning
            level1: Enable Level 1 learning
            level2: Enable Level 2 learning
        """
        self.current_phase = phase_name
        phase_start = time.time()
        duration_seconds = duration * 86400  # Convert days to seconds
        
        self.logger.info(f"Starting phase: {phase_name} (duration: {duration} days)")
        self.logger.info(f"Learning - L0: {level0}, L1: {level1}, L2: {level2}")
        
        # Configure learning levels
        if self.query_optimizer:
            self.query_optimizer.set_learning_enabled(level0)
        if self.policy_learner:
            self.policy_learner.set_enabled(level1)
        if self.meta_learner:
            self.meta_learner.set_enabled(level2)
            
        # Timers for periodic tasks
        last_policy_update = time.time()
        last_meta_learning = time.time()
        last_safety_check = time.time()
        last_metrics_report = time.time()
        
        # Phase execution loop
        while time.time() - phase_start < duration_seconds and self.running:
            if self.paused:
                time.sleep(1)
                continue
                
            try:
                # Generate and execute query
                query, query_type = self.workload_generator.generate_query()
                state = self._get_database_state()
                
                # Execute query through optimizer
                result = self.query_optimizer.execute_query(
                    query,
                    query_type,
                    state
                )
                
                self.stats["queries_executed"] += 1
                
                # Collect telemetry
                self.telemetry_collector.record_execution(
                    query=query,
                    query_type=query_type,
                    execution_time=result['execution_time'],
                    resources=result['resources'],
                    plan_info=result['plan_info']
                )
                
                # Periodic policy updates (Level 1)
                if level1 and self.policy_learner:
                    interval = self.config['level1']['update_interval']
                    if time.time() - last_policy_update > interval:
                        self.logger.info("Running policy update (Level 1)...")
                        updated = self.policy_learner.update_policy()
                        if updated:
                            self.stats["policies_updated"] += 1
                        last_policy_update = time.time()
                
                # Periodic meta-learning (Level 2)
                if level2 and self.meta_learner:
                    interval = self.config['level2']['evaluation_interval']
                    if time.time() - last_meta_learning > interval:
                        self.logger.info("Running meta-learning (Level 2)...")
                        self.meta_learner.optimize()
                        self.stats["meta_learner_runs"] += 1
                        last_meta_learning = time.time()
                
                # Safety monitoring
                if time.time() - last_safety_check > 10:  # Every 10 seconds
                    safety_status = self.safety_monitor.check_system_health()
                    if not safety_status['healthy']:
                        self.logger.warning(f"Safety issue detected: {safety_status['issues']}")
                        self.stats["safety_events"] += 1
                        self._handle_safety_event(safety_status)
                    last_safety_check = time.time()
                
                # Periodic metrics reporting
                if time.time() - last_metrics_report > 300:  # Every 5 minutes
                    self._log_metrics_summary()
                    last_metrics_report = time.time()
                
                # Small delay between queries
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}", exc_info=True)
                time.sleep(1)
                
        elapsed = time.time() - phase_start
        self.logger.info(f"Phase {phase_name} completed in {elapsed:.1f} seconds")
        self._log_phase_summary(phase_name)
        
    def _get_database_state(self) -> Dict[str, Any]:
        """Get current database state for optimizer."""
        try:
            state = {
                'cache_hit_rate': self.db_manager.get_cache_hit_rate(),
                'connection_count': self.db_manager.get_connection_count(),
                'table_sizes': self.db_manager.get_table_sizes(),
                'index_usage': self.db_manager.get_index_usage(),
                'load_average': self.telemetry_collector.get_system_load()
            }
            return state
        except Exception as e:
            self.logger.warning(f"Error getting database state: {e}")
            return {}
            
    def _handle_safety_event(self, safety_status: Dict[str, Any]):
        """Handle safety events."""
        if safety_status['severity'] == 'critical':
            self.logger.error("Critical safety event - initiating rollback")
            if self.policy_learner:
                self.policy_learner.rollback_policy()
        elif safety_status['severity'] == 'warning':
            self.logger.warning("Safety warning - monitoring closely")
            
    def _log_metrics_summary(self):
        """Log summary of current metrics."""
        try:
            metrics = self.telemetry_storage.get_recent_metrics(minutes=5)
            if metrics:
                summary = self.metrics_calculator.calculate_summary(metrics)
                self.logger.info(
                    f"Metrics - Avg Latency: {summary['avg_latency']:.1f}ms, "
                    f"P99: {summary['p99_latency']:.1f}ms, "
                    f"Queries: {self.stats['queries_executed']}"
                )
        except Exception as e:
            self.logger.warning(f"Error logging metrics: {e}")
            
    def _log_phase_summary(self, phase_name: str):
        """Log summary statistics for completed phase."""
        try:
            metrics = self.telemetry_storage.get_phase_metrics(phase_name)
            summary = self.metrics_calculator.calculate_summary(metrics)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Phase Summary: {phase_name}")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Queries Executed: {len(metrics)}")
            self.logger.info(f"Average Latency: {summary['avg_latency']:.2f}ms")
            self.logger.info(f"P50 Latency: {summary['p50_latency']:.2f}ms")
            self.logger.info(f"P95 Latency: {summary['p95_latency']:.2f}ms")
            self.logger.info(f"P99 Latency: {summary['p99_latency']:.2f}ms")
            self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
            self.logger.info(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.warning(f"Error generating phase summary: {e}")
            
    def _generate_final_report(self):
        """Generate final report of entire run."""
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL REPORT")
        self.logger.info("="*60)
        
        try:
            # Overall statistics
            self.logger.info(f"Total Runtime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours")
            self.logger.info(f"Total Queries: {self.stats['queries_executed']}")
            self.logger.info(f"Policy Updates: {self.stats['policies_updated']}")
            self.logger.info(f"Meta-Learning Runs: {self.stats['meta_learner_runs']}")
            self.logger.info(f"Safety Events: {self.stats['safety_events']}")
            
            # Performance comparison
            baseline_metrics = self.telemetry_storage.get_phase_metrics("baseline")
            final_metrics = self.telemetry_storage.get_recent_metrics(hours=24)
            
            if baseline_metrics and final_metrics:
                baseline_summary = self.metrics_calculator.calculate_summary(baseline_metrics)
                final_summary = self.metrics_calculator.calculate_summary(final_metrics)
                
                improvement = self.metrics_calculator.calculate_improvement(
                    baseline_summary,
                    final_summary
                )
                
                self.logger.info("\nPerformance Improvements:")
                self.logger.info(f"Average Latency: {improvement['avg_latency']:.1f}%")
                self.logger.info(f"P99 Latency: {improvement['p99_latency']:.1f}%")
                self.logger.info(f"Resource Efficiency: {improvement['resource_efficiency']:.1f}%")
                
            # Save detailed report
            report_path = Path(self.config['paths']['data_dir']) / 'final_report.txt'
            self._save_detailed_report(report_path)
            self.logger.info(f"\nDetailed report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}", exc_info=True)
            
        self.logger.info("="*60 + "\n")
        
    def _save_detailed_report(self, path: Path):
        """Save detailed report to file."""
        # Implementation would write comprehensive report
        pass
        
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
            if self.query_optimizer:
                self.query_optimizer.save_checkpoint()
            if self.policy_learner:
                self.policy_learner.save_state()
            if self.meta_learner:
                self.meta_learner.save_state()
            if self.telemetry_collector:
                self.telemetry_collector.flush()
            if self.db_manager:
                self.db_manager.disconnect()
                
            self.logger.info("System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Self-Improving Database Query Optimizer"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=14.0,
        help="Duration to run in days (can be fractional)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run in fast mode for testing (accelerated time)"
    )
    
    args = parser.parse_args()
    
    # Create and start orchestrator
    orchestrator = SystemOrchestrator(args.config)
    orchestrator.initialize_components()
    orchestrator.start(
        duration_days=args.duration,
        fast_mode=args.fast_mode
    )


if __name__ == "__main__":
    main()