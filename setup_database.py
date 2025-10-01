import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import random
from datetime import datetime, timedelta
from tqdm import tqdm


class DatabaseSetup:
    """Handles database initialization and data generation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize database setup."""
        self.config = self._load_config(config_path)
        self.conn = None
        self.cursor = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def connect_postgres(self):
        """Connect to PostgreSQL server (not specific database)."""
        db_config = self.config['database']
        self.conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database='postgres'  # Connect to default database
        )
        self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.conn.cursor()
        
    def connect_demo_db(self):
        """Connect to the demo database."""
        db_config = self.config['database']
        self.conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['dbname']
        )
        self.cursor = self.conn.cursor()
        
    def create_database(self):
        """Create the demo database if it doesn't exist."""
        print("Creating database...")
        db_name = self.config['database']['dbname']
        
        try:
            # Check if database exists
            self.cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (db_name,)
            )
            exists = self.cursor.fetchone()
            
            if exists:
                print(f"Database {db_name} already exists. Dropping and recreating...")
                # Terminate existing connections
                self.cursor.execute(f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{db_name}'
                    AND pid <> pg_backend_pid();
                """)
                # Drop database
                self.cursor.execute(sql.SQL("DROP DATABASE {}").format(
                    sql.Identifier(db_name)
                ))
            
            # Create database
            self.cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(db_name)
            ))
            print(f"Database {db_name} created successfully")
            
        except Exception as e:
            print(f"Error creating database: {e}")
            raise
            
    def create_schema(self):
        """Create database schema."""
        print("Creating schema...")
        
        schema_sql = """
        -- Regions table
        CREATE TABLE regions (
            region_id SERIAL PRIMARY KEY,
            region_name VARCHAR(100) NOT NULL,
            country VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Suppliers table
        CREATE TABLE suppliers (
            supplier_id SERIAL PRIMARY KEY,
            supplier_name VARCHAR(200) NOT NULL,
            region_id INTEGER REFERENCES regions(region_id),
            contact_email VARCHAR(200),
            phone VARCHAR(50),
            rating DECIMAL(3,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Products table
        CREATE TABLE products (
            product_id SERIAL PRIMARY KEY,
            product_name VARCHAR(200) NOT NULL,
            category VARCHAR(100) NOT NULL,
            supplier_id INTEGER REFERENCES suppliers(supplier_id),
            price DECIMAL(10,2) NOT NULL,
            stock_quantity INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Customers table
        CREATE TABLE customers (
            customer_id SERIAL PRIMARY KEY,
            customer_name VARCHAR(200) NOT NULL,
            email VARCHAR(200) NOT NULL,
            region_id INTEGER REFERENCES regions(region_id),
            loyalty_tier VARCHAR(50),
            registration_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Orders table
        CREATE TABLE orders (
            order_id SERIAL PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            order_date DATE NOT NULL,
            order_status VARCHAR(50) NOT NULL,
            total_amount DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Transactions table (order details)
        CREATE TABLE transactions (
            transaction_id SERIAL PRIMARY KEY,
            order_id INTEGER REFERENCES orders(order_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            discount DECIMAL(5,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes
        CREATE INDEX idx_products_category ON products(category);
        CREATE INDEX idx_products_supplier ON products(supplier_id);
        CREATE INDEX idx_customers_region ON customers(region_id);
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_date ON orders(order_date);
        CREATE INDEX idx_transactions_order ON transactions(order_id);
        CREATE INDEX idx_transactions_product ON transactions(product_id);
        """
        
        try:
            self.cursor.execute(schema_sql)
            self.conn.commit()
            print("Schema created successfully")
        except Exception as e:
            print(f"Error creating schema: {e}")
            self.conn.rollback()
            raise
            
    def generate_data(self, scale_factor: float = 1.0):
        """
        Generate sample data.
        
        Args:
            scale_factor: Multiplier for data volume (1.0 = ~10M rows)
        """
        print("\nGenerating sample data...")
        print(f"Scale factor: {scale_factor}x")
        
        # Calculate row counts based on scale factor
        n_regions = int(10 * scale_factor)
        n_suppliers = int(1000 * scale_factor)
        n_products = int(10000 * scale_factor)
        n_customers = int(100000 * scale_factor)
        n_orders = int(500000 * scale_factor)
        n_transactions = int(2000000 * scale_factor)
        
        # Generate data
        self._generate_regions(n_regions)
        self._generate_suppliers(n_suppliers, n_regions)
        self._generate_products(n_products, n_suppliers)
        self._generate_customers(n_customers, n_regions)
        self._generate_orders(n_orders, n_customers)
        self._generate_transactions(n_transactions, n_orders, n_products)
        
        print("\nData generation complete!")
        
    def _generate_regions(self, n: int):
        """Generate region data."""
        print("Generating regions...")
        
        regions = [
            ('North America', 'USA'),
            ('North America', 'Canada'),
            ('Europe', 'UK'),
            ('Europe', 'Germany'),
            ('Europe', 'France'),
            ('Asia', 'Japan'),
            ('Asia', 'China'),
            ('Asia', 'India'),
            ('South America', 'Brazil'),
            ('Oceania', 'Australia'),
        ]
        
        data = regions[:n]
        if len(data) < n:
            # Repeat if needed
            data = (regions * (n // len(regions) + 1))[:n]
        
        self.cursor.executemany(
            "INSERT INTO regions (region_name, country) VALUES (%s, %s)",
            data
        )
        self.conn.commit()
        
    def _generate_suppliers(self, n: int, n_regions: int):
        """Generate supplier data."""
        print("Generating suppliers...")
        
        batch_size = 1000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                data.append((
                    f"Supplier {i + j}",
                    random.randint(1, n_regions),
                    f"contact{i+j}@supplier.com",
                    f"+1-555-{random.randint(1000, 9999)}",
                    round(random.uniform(1.0, 5.0), 2)
                ))
            
            self.cursor.executemany(
                """INSERT INTO suppliers (supplier_name, region_id, contact_email, phone, rating)
                   VALUES (%s, %s, %s, %s, %s)""",
                data
            )
            self.conn.commit()
            
    def _generate_products(self, n: int, n_suppliers: int):
        """Generate product data."""
        print("Generating products...")
        
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys', 'Sports', 'Home', 'Garden', 'Auto', 'Health']
        
        batch_size = 1000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                data.append((
                    f"Product {i + j}",
                    random.choice(categories),
                    random.randint(1, n_suppliers),
                    round(random.uniform(10, 1000), 2),
                    random.randint(0, 10000)
                ))
            
            self.cursor.executemany(
                """INSERT INTO products (product_name, category, supplier_id, price, stock_quantity)
                   VALUES (%s, %s, %s, %s, %s)""",
                data
            )
            self.conn.commit()
            
    def _generate_customers(self, n: int, n_regions: int):
        """Generate customer data."""
        print("Generating customers...")
        
        tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
        base_date = datetime.now() - timedelta(days=365*3)
        
        batch_size = 1000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                days_ago = random.randint(0, 365*3)
                reg_date = base_date + timedelta(days=days_ago)
                
                data.append((
                    f"Customer {i + j}",
                    f"customer{i+j}@email.com",
                    random.randint(1, n_regions),
                    random.choice(tiers),
                    reg_date.date()
                ))
            
            self.cursor.executemany(
                """INSERT INTO customers (customer_name, email, region_id, loyalty_tier, registration_date)
                   VALUES (%s, %s, %s, %s, %s)""",
                data
            )
            self.conn.commit()
            
    def _generate_orders(self, n: int, n_customers: int):
        """Generate order data."""
        print("Generating orders...")
        
        statuses = ['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled']
        base_date = datetime.now() - timedelta(days=365)
        
        batch_size = 1000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                days_ago = random.randint(0, 365)
                order_date = base_date + timedelta(days=days_ago)
                
                data.append((
                    random.randint(1, n_customers),
                    order_date.date(),
                    random.choice(statuses),
                    round(random.uniform(50, 5000), 2)
                ))
            
            self.cursor.executemany(
                """INSERT INTO orders (customer_id, order_date, order_status, total_amount)
                   VALUES (%s, %s, %s, %s)""",
                data
            )
            self.conn.commit()
            
    def _generate_transactions(self, n: int, n_orders: int, n_products: int):
        """Generate transaction data."""
        print("Generating transactions...")
        
        batch_size = 1000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                data.append((
                    random.randint(1, n_orders),
                    random.randint(1, n_products),
                    random.randint(1, 10),
                    round(random.uniform(10, 1000), 2),
                    round(random.uniform(0, 20), 2)
                ))
            
            self.cursor.executemany(
                """INSERT INTO transactions (order_id, product_id, quantity, unit_price, discount)
                   VALUES (%s, %s, %s, %s, %s)""",
                data
            )
            self.conn.commit()
            
    def create_telemetry_db(self):
        """Create SQLite telemetry database with correct schema."""
        print("Creating telemetry database...")
        
        import sqlite3
        
        telemetry_path = self.config['paths']['telemetry_db']
        Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(telemetry_path)
        cursor = conn.cursor()
        
        # Create tables with complete schema matching telemetry/storage.py
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                phase TEXT NOT NULL,
                query_type TEXT,
                execution_time REAL,
                cpu_usage REAL,
                memory_usage REAL,
                cache_hit_rate REAL,
                rows_processed INTEGER,
                plan_cost REAL,
                success INTEGER,
                query_hash TEXT,
                plan_info TEXT
            );
            
            CREATE TABLE IF NOT EXISTS policy_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                old_version INTEGER,
                new_version INTEGER,
                improvement REAL,
                validation_score REAL,
                changes TEXT
            );
            
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                severity TEXT NOT NULL,
                event_type TEXT,
                description TEXT,
                action_taken TEXT,
                context TEXT
            );
            
            CREATE TABLE IF NOT EXISTS meta_learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                generation INTEGER,
                best_fitness REAL,
                avg_fitness REAL,
                hyperparameters TEXT,
                improvements TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_phase ON metrics(phase);
            CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(query_type);
            CREATE INDEX IF NOT EXISTS idx_policy_timestamp ON policy_updates(timestamp);
            CREATE INDEX IF NOT EXISTS idx_safety_timestamp ON safety_events(timestamp);
        """)
        
        conn.commit()
        conn.close()
        
        print("Telemetry database created")
        
    def verify_setup(self):
        """Verify database setup is correct."""
        print("\nVerifying setup...")
        
        try:
            # Check tables exist and have data
            tables = ['regions', 'suppliers', 'products', 'customers', 'orders', 'transactions']
            
            for table in tables:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                print(f"  {table}: {count:,} rows")
                
            print("\nSetup verification complete!")
            return True
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
            
    def cleanup(self):
        """Close database connections."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    """Main entry point for database setup."""
    parser = argparse.ArgumentParser(description="Setup database for query optimizer demo")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--scale", type=float, default=1.0, help="Data scale factor")
    parser.add_argument("--verify", action="store_true", help="Only verify setup")
    
    args = parser.parse_args()
    
    setup = DatabaseSetup(args.config)
    
    try:
        if args.verify:
            setup.connect_demo_db()
            setup.verify_setup()
        else:
            print("Starting database setup...")
            print("This will take 10-15 minutes...")
            print()
            
            # Connect and create database
            setup.connect_postgres()
            setup.create_database()
            setup.cleanup()
            
            # Connect to new database and create schema
            setup.connect_demo_db()
            setup.create_schema()
            
            # Generate data
            setup.generate_data(args.scale)
            
            # Create telemetry database
            setup.create_telemetry_db()
            
            # Verify
            setup.verify_setup()
            
            print("\nDatabase setup complete!")
            print("You can now run the demo with: python run_demo.py")
            
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
    finally:
        setup.cleanup()


if __name__ == "__main__":
    main()