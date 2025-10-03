import os
import sys
import json
from flask import Flask, render_template, jsonify
from datetime import datetime
import sqlite3
from typing import Dict, List, Optional

app = Flask(__name__, 
            template_folder='dashboard/templates',
            static_folder='dashboard/static')

class DashboardData:
    """Fetches and processes data for the dashboard."""
    
    def __init__(self, db_path: str = "data/telemetry.db"):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection."""
        if not os.path.exists(self.db_path):
            return None
        return sqlite3.connect(self.db_path)
    
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
                # Convert success to boolean properly for Windows
                success_value = bool(row[2]) if row[2] is not None else False
                results.append({
                    'timestamp': row[0],
                    'latency_ms': float(row[1]) if row[1] is not None else 0.0,
                    'success': success_value,
                    'phase': str(row[3]) if row[3] else 'unknown'
                })
            
            return results
        except Exception as e:
            print(f"Error fetching recent metrics: {e}")
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
                    COUNT(*) as query_count,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM query_metrics
                GROUP BY phase
                ORDER BY MIN(timestamp)
            """)
            
            phases = {}
            for row in cursor.fetchall():
                phases[row[0]] = {
                    'query_count': int(row[1]) if row[1] else 0,
                    'avg_latency': round(float(row[2]), 2) if row[2] else 0,
                    'min_latency': round(float(row[3]), 2) if row[3] else 0,
                    'max_latency': round(float(row[4]), 2) if row[4] else 0,
                    'success_rate': round(float(row[5]), 2) if row[5] else 0
                }
            
            return phases
        except Exception as e:
            print(f"Error fetching phase summary: {e}")
            return {}
        finally:
            conn.close()
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        conn = self.get_connection()
        if not conn:
            return {
                'policy_updates': 0,
                'meta_learning_runs': 0,
                'latest_loss': None,
                'latest_loss_time': None
            }
        
        try:
            cursor = conn.cursor()
            
            # Get policy updates
            cursor.execute("SELECT COUNT(*) FROM policy_updates")
            result = cursor.fetchone()
            policy_updates = int(result[0]) if result and result[0] else 0
            
            # Get meta-learning runs
            cursor.execute("SELECT COUNT(*) FROM meta_learning_runs")
            result = cursor.fetchone()
            meta_runs = int(result[0]) if result and result[0] else 0
            
            # Get latest losses
            cursor.execute("""
                SELECT loss, timestamp 
                FROM policy_updates 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            latest_loss = cursor.fetchone()
            
            return {
                'policy_updates': policy_updates,
                'meta_learning_runs': meta_runs,
                'latest_loss': round(float(latest_loss[0]), 6) if latest_loss and latest_loss[0] else None,
                'latest_loss_time': latest_loss[1] if latest_loss else None
            }
        except Exception as e:
            print(f"Error fetching learning stats: {e}")
            return {
                'policy_updates': 0,
                'meta_learning_runs': 0,
                'latest_loss': None,
                'latest_loss_time': None
            }
        finally:
            conn.close()
    
    def get_latency_distribution(self) -> Dict:
        """Get latency percentiles."""
        conn = self.get_connection()
        if not conn:
            return {}
        
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
                return {
                    'p50': 0,
                    'p95': 0,
                    'p99': 0,
                    'p999': 0
                }
            
            import numpy as np
            return {
                'p50': round(float(np.percentile(latencies, 50)), 2),
                'p95': round(float(np.percentile(latencies, 95)), 2),
                'p99': round(float(np.percentile(latencies, 99)), 2),
                'p999': round(float(np.percentile(latencies, 99.9)), 2)
            }
        except Exception as e:
            print(f"Error fetching latency distribution: {e}")
            return {
                'p50': 0,
                'p95': 0,
                'p99': 0,
                'p999': 0
            }
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
        print(f"Error in /api/metrics: {e}")
        return jsonify([]), 500

@app.route('/api/phase-summary')
def api_phase_summary():
    """API endpoint for phase summary."""
    try:
        summary = dashboard_data.get_phase_summary()
        return jsonify(summary)
    except Exception as e:
        print(f"Error in /api/phase-summary: {e}")
        return jsonify({}), 500

@app.route('/api/learning-stats')
def api_learning_stats():
    """API endpoint for learning statistics."""
    try:
        stats = dashboard_data.get_learning_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in /api/learning-stats: {e}")
        return jsonify({}), 500

@app.route('/api/latency-distribution')
def api_latency_distribution():
    """API endpoint for latency distribution."""
    try:
        distribution = dashboard_data.get_latency_distribution()
        return jsonify(distribution)
    except Exception as e:
        print(f"Error in /api/latency-distribution: {e}")
        return jsonify({}), 500

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        db_exists = os.path.exists(dashboard_data.db_path)
        return jsonify({
            'status': 'running' if db_exists else 'no_database',
            'timestamp': datetime.now().isoformat(),
            'db_exists': db_exists,
            'db_path': dashboard_data.db_path
        })
    except Exception as e:
        print(f"Error in /api/status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def check_prerequisites():
    """Check if prerequisites are met."""
    # Check if data directory exists
    if not os.path.exists('data'):
        print("\nError: 'data' directory not found.")
        print("Please run 'python setup_database.py' first to initialize the database.\n")
        return False
    
    # Check if telemetry database exists
    if not os.path.exists(dashboard_data.db_path):
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
    print(f"  Database exists: {os.path.exists(dashboard_data.db_path)}")
    print("="*70)
    print("\nStarting Flask server...")
    print("Press CTRL+C to stop the server\n")
    
    # Use 127.0.0.1 instead of 0.0.0.0 for better Windows compatibility
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if port 5000 is already in use")
        print("  2. Try running as administrator")
        print("  3. Check Windows Firewall settings")
        sys.exit(1)