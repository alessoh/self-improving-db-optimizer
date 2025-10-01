import random
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List


class WorkloadGenerator:
    """Generates synthetic database workload for testing."""
    
    def __init__(self, config: Dict[str, Any], db_manager):
        """
        Initialize workload generator.
        
        Args:
            config: System configuration
            db_manager: DatabaseManager instance
        """
        self.config = config
        self.db_manager = db_manager
        self.workload_config = config['workload']
        
        # Query templates
        self.query_templates = {
            'select_simple': [
                "SELECT * FROM customers WHERE customer_id = {id}",
                "SELECT * FROM products WHERE product_id = {id}",
                "SELECT * FROM orders WHERE order_id = {id}",
            ],
            'join_two_tables': [
                """SELECT c.customer_name, o.order_date, o.total_amount
                   FROM customers c
                   JOIN orders o ON c.customer_id = o.customer_id
                   WHERE c.customer_id = {id}""",
                """SELECT p.product_name, s.supplier_name, p.price
                   FROM products p
                   JOIN suppliers s ON p.supplier_id = s.supplier_id
                   WHERE p.category = '{category}'""",
            ],
            'join_multiple': [
                """SELECT c.customer_name, o.order_date, p.product_name, t.quantity
                   FROM customers c
                   JOIN orders o ON c.customer_id = o.customer_id
                   JOIN transactions t ON o.order_id = t.order_id
                   JOIN products p ON t.product_id = p.product_id
                   WHERE o.order_date >= '{date}'
                   LIMIT {limit}""",
            ],
            'aggregation': [
                """SELECT category, COUNT(*) as count, AVG(price) as avg_price
                   FROM products
                   GROUP BY category
                   ORDER BY count DESC""",
                """SELECT customer_id, SUM(total_amount) as total_spent
                   FROM orders
                   WHERE order_date >= '{date}'
                   GROUP BY customer_id
                   ORDER BY total_spent DESC
                   LIMIT {limit}""",
            ],
            'analytical': [
                """SELECT 
                       DATE_TRUNC('month', order_date) as month,
                       COUNT(*) as order_count,
                       SUM(total_amount) as revenue
                   FROM orders
                   WHERE order_date >= '{date}'
                   GROUP BY month
                   ORDER BY month""",
                """SELECT 
                       r.region_name,
                       COUNT(DISTINCT c.customer_id) as customer_count,
                       SUM(o.total_amount) as total_revenue
                   FROM regions r
                   JOIN customers c ON r.region_id = c.region_id
                   JOIN orders o ON c.customer_id = o.customer_id
                   WHERE o.order_date >= '{date}'
                   GROUP BY r.region_name
                   ORDER BY total_revenue DESC""",
            ]
        }
        
    def generate_query(self) -> Tuple[str, str]:
        """
        Generate a random query based on distribution.
        
        Returns:
            Tuple of (query_string, query_type)
        """
        # Select query type based on distribution
        distribution = self.workload_config['query_distribution']
        query_type = random.choices(
            list(distribution.keys()),
            weights=list(distribution.values())
        )[0]
        
        # Select template
        template = random.choice(self.query_templates[query_type])
        
        # Fill in parameters
        params = self._generate_parameters()
        query = template.format(**params)
        
        return query, query_type
    
    def _generate_parameters(self) -> Dict[str, Any]:
        """Generate random query parameters."""
        param_ranges = self.workload_config['parameter_ranges']
        
        # Random date in the past year
        days_ago = random.randint(1, param_ranges.get('date_range_days', 365))
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Random categories
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys', 
                     'Sports', 'Home', 'Garden', 'Auto', 'Health']
        
        return {
            'id': random.randint(1, 100000),
            'limit': random.randint(*param_ranges.get('limit_rows', [10, 1000])),
            'date': date,
            'category': random.choice(categories),
            'price_min': random.randint(1, 500),
            'price_max': random.randint(500, 10000)
        }