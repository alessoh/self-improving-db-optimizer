# database/database_manager.py

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
from typing import Dict, Any, List, Optional, Tuple
import logging


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.db_config = config['database']
        self.logger = logging.getLogger(__name__)
        
        self.connection_pool = None
        self.conn = None
        
    def connect(self):
        """Establish database connection and create connection pool."""
        try:
            # Create connection pool
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self.db_config.get('max_connections', 10),
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['dbname'],
                connect_timeout=self.db_config.get('connection_timeout', 30)
            )
            
            # Test connection
            self.conn = self.get_connection()
            cursor = self.conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            cursor.close()
            self.return_connection(self.conn)
            
            self.logger.info(f"Connected to PostgreSQL: {version[0]}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool."""
        if self.connection_pool:
            return self.connection_pool.getconn()
        raise Exception("Connection pool not initialized")
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True
    ) -> Tuple[Optional[List], float, Dict[str, Any]]:
        """
        Execute a query and collect metrics.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Tuple of (results, execution_time, resource_metrics)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        start_time = time.time()
        results = None
        
        try:
            # Enable timing
            cursor.execute("SET track_io_timing = ON")
            
            # Execute query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results if needed
            if fetch and cursor.description:
                results = cursor.fetchall()
            
            conn.commit()
            execution_time = time.time() - start_time
            
            # Get execution metrics
            metrics = self._collect_query_metrics(cursor)
            
            cursor.close()
            self.return_connection(conn)
            
            return results, execution_time, metrics
            
        except Exception as e:
            conn.rollback()
            cursor.close()
            self.return_connection(conn)
            self.logger.error(f"Query execution error: {e}")
            raise
    
    def _collect_query_metrics(self, cursor) -> Dict[str, Any]:
        """Collect query execution metrics."""
        metrics = {
            'cpu': 0,
            'memory': 0,
            'cache_hit_rate': 0,
            'rows': 0,
            'cost': 0
        }
        
        try:
            # Get row count
            if cursor.rowcount >= 0:
                metrics['rows'] = cursor.rowcount
                
        except:
            pass
        
        return metrics
    
    def get_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Get query execution plan.
        
        Args:
            query: SQL query
            
        Returns:
            Dictionary containing plan information
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get EXPLAIN output
            cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
            plan = cursor.fetchone()[0][0]
            
            cursor.close()
            self.return_connection(conn)
            
            return {
                'plan': plan,
                'cost': plan.get('Plan', {}).get('Total Cost', 0),
                'rows': plan.get('Plan', {}).get('Plan Rows', 0)
            }
            
        except Exception as e:
            cursor.close()
            self.return_connection(conn)
            self.logger.warning(f"Failed to get query plan: {e}")
            return {'plan': {}, 'cost': 0, 'rows': 0}
    
    def get_cache_hit_rate(self) -> float:
        """Get database cache hit rate."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    sum(heap_blks_hit) / nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0) as cache_hit_rate
                FROM pg_statio_user_tables
            """)
            result = cursor.fetchone()
            cache_hit_rate = result[0] if result and result[0] else 0.0
            
            cursor.close()
            self.return_connection(conn)
            
            return float(cache_hit_rate)
            
        except:
            cursor.close()
            self.return_connection(conn)
            return 0.0
    
    def get_connection_count(self) -> int:
        """Get current connection count."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT count(*) 
                FROM pg_stat_activity 
                WHERE datname = %s
            """, (self.db_config['dbname'],))
            count = cursor.fetchone()[0]
            
            cursor.close()
            self.return_connection(conn)
            
            return count
            
        except:
            cursor.close()
            self.return_connection(conn)
            return 0
    
    def get_table_sizes(self) -> Dict[str, int]:
        """Get sizes of all tables in bytes."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    tablename,
                    pg_total_relation_size(schemaname||'.'||tablename) as size
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY size DESC
            """)
            
            sizes = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.close()
            self.return_connection(conn)
            
            return sizes
            
        except:
            cursor.close()
            self.return_connection(conn)
            return {}
    
    def get_index_usage(self) -> Dict[str, float]:
        """Get index usage statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    indexrelname as index_name,
                    idx_scan as scans
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
            """)
            
            usage = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.close()
            self.return_connection(conn)
            
            return usage
            
        except:
            cursor.close()
            self.return_connection(conn)
            return {}
    
    def disconnect(self):
        """Close all connections."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Database connections closed")