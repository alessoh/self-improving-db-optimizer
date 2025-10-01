import os
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
                results.append({
                    'timestamp': row[0],
                    'latency_ms': row[1],
                    'success': row[2],
                    'phase': row[3]
                })
            
            return results
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
                    'query_count': row[1],
                    'avg_latency': round(row[2], 2) if row[2] else 0,
                    'min_latency': round(row[3], 2) if row[3] else 0,
                    'max_latency': round(row[4], 2) if row[4] else 0,
                    'success_rate': round(row[5], 2) if row[5] else 0
                }
            
            return phases
        finally:
            conn.close()
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Get policy updates
            cursor.execute("SELECT COUNT(*) FROM policy_updates")
            policy_updates = cursor.fetchone()[0]
            
            # Get meta-learning runs
            cursor.execute("SELECT COUNT(*) FROM meta_learning_runs")
            meta_runs = cursor.fetchone()[0]
            
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
                'latest_loss': round(latest_loss[0], 6) if latest_loss else None,
                'latest_loss_time': latest_loss[1] if latest_loss else None
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
                ORDER BY latency_ms
            """)
            
            latencies = [row[0] for row in cursor.fetchall()]
            if not latencies:
                return {}
            
            import numpy as np
            return {
                'p50': round(np.percentile(latencies, 50), 2),
                'p95': round(np.percentile(latencies, 95), 2),
                'p99': round(np.percentile(latencies, 99), 2),
                'p999': round(np.percentile(latencies, 99.9), 2)
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
    metrics = dashboard_data.get_recent_metrics(limit=200)
    return jsonify(metrics)

@app.route('/api/phase-summary')
def api_phase_summary():
    """API endpoint for phase summary."""
    summary = dashboard_data.get_phase_summary()
    return jsonify(summary)

@app.route('/api/learning-stats')
def api_learning_stats():
    """API endpoint for learning statistics."""
    stats = dashboard_data.get_learning_stats()
    return jsonify(stats)

@app.route('/api/latency-distribution')
def api_latency_distribution():
    """API endpoint for latency distribution."""
    distribution = dashboard_data.get_latency_distribution()
    return jsonify(distribution)

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'db_exists': os.path.exists(dashboard_data.db_path)
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Database Query Optimizer - Dashboard")
    print("="*70)
    print(f"  Dashboard URL: http://localhost:5000")
    print(f"  Database: {dashboard_data.db_path}")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)