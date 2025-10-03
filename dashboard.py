#!/usr/bin/env python
"""
Dashboard Server for Database Query Optimizer
Windows-Compatible Version with All Fixes Applied
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, jsonify
from datetime import datetime
import sqlite3
from typing import Dict, List, Optional

# FIXED: Ensure UTF-8 encoding for Windows
import locale
locale.setlocale(locale.LC_ALL, '')

app = Flask(__name__, 
            template_folder='dashboard/templates',
            static_folder='dashboard/static')


class DashboardData:
    """Fetches and processes data for the dashboard."""
    
    def __init__(self, db_path: str = "data/telemetry.db"):
        # FIXED: Use Path object for Windows compatibility
        self.db_path = Path(db_path).resolve()
    
    def get_connection(self):
        """Get database connection."""
        if not self.db_path.exists():
            return None
        
        # FIXED: Explicit timeout and check_same_thread for Windows
        return sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False
        )
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Get recent query metrics."""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, latency_ms, success, phase
                FROM query_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                # FIXED: Proper boolean conversion for Windows
                success_value = bool(row[2]) if row[2] is not None else False
                results.append({
                    'timestamp': row[0],
                    'latency_ms': float(row[1]) if row[1] is not None else 0.0,
                    'success': success_value,
                    'phase': str(row[3]) if row[3] else 'unknown'
                })
            
            return results
        except Exception as e:
            print(f"Error fetching recent metrics: {e}", file=sys.stderr)
            return []
        finally:
            conn.close()
    
    def get_phase_summary(self) -> Dict:
        """Get summary statistics by phase."""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    phase,
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM query_metrics
                WHERE phase IS NOT NULL
                GROUP BY phase
                ORDER BY phase
            """)
            
            summary = {}
            for row in cursor.fetchall():
                phase = str(row[0])
                summary[phase] = {
                    'count': int(row[1]),
                    'avg_latency': round(float(row[2]), 2) if row[2] else 0.0,
                    'min_latency': round(float(row[3]), 2) if row[3] else 0.0,
                    'max_latency': round(float(row[4]), 2) if row[4] else 0.0,
                    'success_rate': round(float(row[5]), 2) if row[5] else 0.0
                }
            
            return summary
        except Exception as e:
            print(f"Error fetching phase summary: {e}", file=sys.stderr)
            return {}
        finally:
            conn.close()
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Get latest learning metrics
            cursor.execute("""
                SELECT 
                    epsilon,
                    learning_rate,
                    loss
                FROM learning_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'latest_epsilon': round(float(row[0]), 4) if row[0] else 1.0,
                    'latest_lr': float(row[1]) if row[1] else 0.0,
                    'latest_loss': round(float(row[2]), 6) if row[2] else 0.0
                }
            
            return {
                'latest_epsilon': 1.0,
                'latest_lr': 0.0,
                'latest_loss': 0.0
            }
        except Exception as e:
            print(f"Error fetching learning stats: {e}", file=sys.stderr)
            return {
                'latest_epsilon': 1.0,
                'latest_lr': 0.0,
                'latest_loss': 0.0
            }
        finally:
            conn.close()
    
    def get_latency_distribution(self) -> Dict:
        """Get latency percentile distribution."""
        conn = self.get_connection()
        if not conn:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT latency_ms
                FROM query_metrics
                WHERE latency_ms IS NOT NULL
                ORDER BY latency_ms
            """)
            
            latencies = [float(row[0]) for row in cursor.fetchall() if row[0] is not None]
            if not latencies:
                return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
            
            # FIXED: Safe percentile calculation for Windows
            try:
                import numpy as np
                return {
                    'p50': round(float(np.percentile(latencies, 50)), 2),
                    'p95': round(float(np.percentile(latencies, 95)), 2),
                    'p99': round(float(np.percentile(latencies, 99)), 2),
                    'p999': round(float(np.percentile(latencies, 99.9)), 2)
                }
            except ImportError:
                # Fallback if numpy not available
                n = len(latencies)
                return {
                    'p50': round(latencies[int(n * 0.50)], 2),
                    'p95': round(latencies[int(n * 0.95)], 2),
                    'p99': round(latencies[int(n * 0.99)], 2),
                    'p999': round(latencies[int(n * 0.999)], 2)
                }
        except Exception as e:
            print(f"Error fetching latency distribution: {e}", file=sys.stderr)
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
        finally:
            conn.close()


dashboard_data = DashboardData()


@app.route('/')
def index():
    """Render main dashboard page."""
    return render_template('index.html')


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for recent metrics."""
    try:
        metrics = dashboard_data.get_recent_metrics(limit=200)
        return jsonify(metrics)
    except Exception as e:
        print(f"Error in /api/metrics: {e}", file=sys.stderr)
        return jsonify([]), 500


@app.route('/api/phase-summary')
def api_phase_summary():
    """API endpoint for phase summary."""
    try:
        summary = dashboard_data.get_phase_summary()
        return jsonify(summary)
    except Exception as e:
        print(f"Error in /api/phase-summary: {e}", file=sys.stderr)
        return jsonify({}), 500


@app.route('/api/learning-stats')
def api_learning_stats():
    """API endpoint for learning statistics."""
    try:
        stats = dashboard_data.get_learning_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in /api/learning-stats: {e}", file=sys.stderr)
        return jsonify({}), 500


@app.route('/api/latency-distribution')
def api_latency_distribution():
    """API endpoint for latency distribution."""
    try:
        distribution = dashboard_data.get_latency_distribution()
        return jsonify(distribution)
    except Exception as e:
        print(f"Error in /api/latency-distribution: {e}", file=sys.stderr)
        return jsonify({}), 500


@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        db_exists = dashboard_data.db_path.exists()
        return jsonify({
            'status': 'running' if db_exists else 'no_database',
            'timestamp': datetime.now().isoformat(),
            'db_exists': db_exists,
            'db_path': str(dashboard_data.db_path)
        })
    except Exception as e:
        print(f"Error in /api/status: {e}", file=sys.stderr)
        return jsonify({'status': 'error', 'message': str(e)}), 500


def check_prerequisites():
    """Check if prerequisites are met."""
    # FIXED: Use Path objects
    data_dir = Path('data')
    
    if not data_dir.exists():
        print("\nError: 'data' directory not found.")
        print("Please run 'python setup_database.py' first to initialize the database.\n")
        return False
    
    if not dashboard_data.db_path.exists():
        print(f"\nWarning: Telemetry database not found at {dashboard_data.db_path}")
        print("The dashboard will start but no data will be displayed until you:")
        print("  1. Run 'python setup_database.py' to initialize the database")
        print("  2. Run 'python run_demo.py' to generate data\n")
    
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Database Query Optimizer - Dashboard")
    print("="*70)
    
    if not check_prerequisites():
        sys.exit(1)
    
    print(f"  Dashboard URLs:")
    print(f"    - http://localhost:5000")
    print(f"    - http://127.0.0.1:5000")
    print(f"  Database: {dashboard_data.db_path}")
    print(f"  Database exists: {dashboard_data.db_path.exists()}")
    print("="*70)
    print("\nStarting Flask server...")
    print("Press CTRL+C to stop the server\n")
    
    # FIXED: Use 127.0.0.1 for Windows compatibility, disable reloader
    try:
        app.run(
            debug=False,  # FIXED: Disabled debug for stability
            host='127.0.0.1',  # FIXED: localhost only
            port=5000,
            use_reloader=False  # FIXED: Disabled reloader for Windows
        )
    except Exception as e:
        print(f"\nError starting dashboard: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Check if port 5000 is already in use")
        print("  2. Try running as administrator")
        print("  3. Check Windows Firewall settings")
        print("\nTo find process using port 5000:")
        print("  netstat -ano | findstr :5000")
        sys.exit(1)