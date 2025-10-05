# monitor_learning.py
"""Real-time monitoring of learning progress"""

import time
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

class LearningMonitor:
    def __init__(self, db_path="data/telemetry.db", refresh_interval=5):
        self.db_path = Path(db_path)
        self.refresh_interval = refresh_interval
        
    def monitor(self):
        """Continuous monitoring loop."""
        print("\n" + "="*70)
        print("LEARNING PROGRESS MONITOR")
        print("="*70)
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.display_status()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def display_status(self):
        """Display current learning status."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get recent metrics
        cursor.execute("""
            SELECT COUNT(*), AVG(execution_time), MAX(timestamp), phase
            FROM metrics
            WHERE timestamp > (SELECT MAX(timestamp) - 300 FROM metrics)
            GROUP BY phase
        """)
        
        recent = cursor.fetchall()
        
        # Get policy updates
        cursor.execute("SELECT COUNT(*) FROM policy_updates")
        policy_count = cursor.fetchone()[0]
        
        # Get latest improvement
        cursor.execute("""
            SELECT improvement FROM policy_updates 
            ORDER BY timestamp DESC LIMIT 1
        """)
        latest_improvement = cursor.fetchone()
        
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        print("\nRecent Activity (last 5 minutes):")
        for count, avg_time, last_time, phase in recent:
            print(f"  {phase}: {count} queries, avg {avg_time*1000:.1f}ms")
        
        print(f"\nLearning Progress:")
        print(f"  Policy updates: {policy_count}")
        if latest_improvement:
            print(f"  Latest improvement: {latest_improvement[0]*100:.2f}%")
        
        conn.close()

if __name__ == "__main__":
    monitor = LearningMonitor()
    monitor.monitor()
