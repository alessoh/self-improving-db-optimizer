import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import yaml
import time
from datetime import datetime, timedelta
import sqlite3
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telemetry.storage import TelemetryStorage
from utils.metrics import MetricsCalculator


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
config = None
telemetry_storage = None
metrics_calculator = None


def load_config(config_path: str = "config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def init_dashboard():
    """Initialize dashboard components."""
    global config, telemetry_storage, metrics_calculator
    
    config = load_config()
    telemetry_storage = TelemetryStorage(config)
    metrics_calculator = MetricsCalculator(config)


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current system status."""
    try:
        # Get recent metrics
        recent_metrics = telemetry_storage.get_recent_metrics(minutes=5)
        
        if recent_metrics:
            latest = recent_metrics[-1]
            summary = metrics_calculator.calculate_summary(recent_metrics)
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'running': True,
                'current_phase': latest.get('phase', 'unknown'),
                'queries_executed': len(recent_metrics),
                'avg_latency': summary['avg_latency'],
                'p99_latency': summary['p99_latency'],
                'success_rate': summary['success_rate'],
                'cache_hit_rate': summary.get('cache_hit_rate', 0)
            }
        else:
            status = {
                'timestamp': datetime.now().isoformat(),
                'running': False,
                'message': 'No recent data'
            }
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/timeseries')
def get_metrics_timeseries():
    """Get time series metrics for charting."""
    try:
        hours = float(request.args.get('hours', 24))
        metrics = telemetry_storage.get_recent_metrics(hours=hours)
        
        if not metrics:
            return jsonify({'data': []})
        
        # Aggregate by time windows
        window_size = 60  # seconds
        aggregated = []
        current_window = []
        current_time = metrics[0]['timestamp']
        
        for metric in metrics:
            if metric['timestamp'] - current_time > window_size:
                if current_window:
                    summary = metrics_calculator.calculate_summary(current_window)
                    aggregated.append({
                        'timestamp': current_time,
                        'avg_latency': summary['avg_latency'],
                        'p50_latency': summary['p50_latency'],
                        'p95_latency': summary['p95_latency'],
                        'p99_latency': summary['p99_latency'],
                        'success_rate': summary['success_rate']
                    })
                current_window = []
                current_time = metric['timestamp']
            current_window.append(metric)
        
        # Add final window
        if current_window:
            summary = metrics_calculator.calculate_summary(current_window)
            aggregated.append({
                'timestamp': current_time,
                'avg_latency': summary['avg_latency'],
                'p50_latency': summary['p50_latency'],
                'p95_latency': summary['p95_latency'],
                'p99_latency': summary['p99_latency'],
                'success_rate': summary['success_rate']
            })
        
        return jsonify({'data': aggregated})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/phase_comparison')
def get_phase_comparison():
    """Get performance comparison across phases."""
    try:
        phases = ['baseline', 'level0_learning', 'level1_learning', 'level2_learning']
        comparison = []
        
        for phase in phases:
            metrics = telemetry_storage.get_phase_metrics(phase)
            if metrics:
                summary = metrics_calculator.calculate_summary(metrics)
                comparison.append({
                    'phase': phase,
                    'avg_latency': summary['avg_latency'],
                    'p99_latency': summary['p99_latency'],
                    'queries': len(metrics),
                    'success_rate': summary['success_rate']
                })
        
        return jsonify({'data': comparison})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/policy_updates')
def get_policy_updates():
    """Get policy update history."""
    try:
        updates = telemetry_storage.get_policy_updates(limit=50)
        return jsonify({'data': updates})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/meta_decisions')
def get_meta_decisions():
    """Get meta-learner decisions."""
    try:
        # This would be populated by the meta-learner
        # For now, return sample data
        decisions = []
        return jsonify({'data': decisions})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/events')
def get_safety_events():
    """Get safety monitor events."""
    try:
        events = telemetry_storage.get_safety_events(limit=100)
        return jsonify({'data': events})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query_types/distribution')
def get_query_type_distribution():
    """Get distribution of query types."""
    try:
        metrics = telemetry_storage.get_recent_metrics(hours=1)
        
        distribution = {}
        for metric in metrics:
            query_type = metric.get('query_type', 'unknown')
            distribution[query_type] = distribution.get(query_type, 0) + 1
            
        data = [{'type': k, 'count': v} for k, v in distribution.items()]
        return jsonify({'data': data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/resources/usage')
def get_resource_usage():
    """Get resource usage statistics."""
    try:
        metrics = telemetry_storage.get_recent_metrics(minutes=5)
        
        if not metrics:
            return jsonify({'data': {}})
        
        recent = metrics[-10:]  # Last 10 queries
        
        avg_cpu = sum(m.get('cpu_usage', 0) for m in recent) / len(recent)
        avg_memory = sum(m.get('memory_usage', 0) for m in recent) / len(recent)
        cache_hit_rate = sum(m.get('cache_hit_rate', 0) for m in recent) / len(recent)
        
        data = {
            'cpu_usage': avg_cpu,
            'memory_usage': avg_memory,
            'cache_hit_rate': cache_hit_rate
        }
        
        return jsonify({'data': data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/improvement/summary')
def get_improvement_summary():
    """Get overall improvement summary."""
    try:
        baseline_metrics = telemetry_storage.get_phase_metrics('baseline')
        current_metrics = telemetry_storage.get_recent_metrics(hours=1)
        
        if not baseline_metrics or not current_metrics:
            return jsonify({'data': None})
        
        baseline_summary = metrics_calculator.calculate_summary(baseline_metrics)
        current_summary = metrics_calculator.calculate_summary(current_metrics)
        
        improvement = metrics_calculator.calculate_improvement(
            baseline_summary,
            current_summary
        )
        
        return jsonify({'data': improvement})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Run the dashboard server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query Optimizer Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize dashboard
    print("Initializing dashboard...")
    init_dashboard()
    
    print(f"\nDashboard starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()