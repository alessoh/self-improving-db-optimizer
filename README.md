# Self-Improving Database Query Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 14+](https://img.shields.io/badge/postgresql-14+-blue.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready demonstration of autonomous database optimization through hierarchical machine learning, featuring three distinct levels of self-improvement without manual intervention.

## Overview

This system implements a novel three-tier architecture for autonomous database query optimization:

- **Level 0 (Operational)**: Deep reinforcement learning agent that executes and optimizes individual queries in real-time
- **Level 1 (Tactical)**: Policy learner that analyzes execution telemetry and continuously refines operational strategies
- **Level 2 (Strategic)**: Meta-learner that optimizes the learning process itself through evolutionary algorithms

Unlike traditional rule-based query optimizers, this system learns from experience, adapts to workload patterns, and improves its own learning mechanisms autonomously.

## Key Features

- **Autonomous Learning**: Zero-configuration optimization that improves without human intervention
- **Multi-Level Architecture**: Hierarchical learning from query execution to meta-optimization
- **Safety Mechanisms**: Comprehensive monitoring with automatic rollback capabilities
- **Real-Time Dashboard**: Web-based visualization of performance metrics and learning progress
- **Production-Ready**: Built-in telemetry, logging, and state management
- **Extensive Telemetry**: Detailed tracking of all system decisions and performance metrics

## Architecture

### Three-Tier Learning Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  Level 2: Strategic Meta-Learning                           │
│  • Optimizes learning hyperparameters                       │
│  • Evolutionary algorithm for architecture search           │
│  • Adapts to changing workload characteristics             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Level 1: Tactical Policy Learning                          │
│  • Analyzes execution patterns                              │
│  • Updates operational policies                             │
│  • Validates changes before deployment                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Level 0: Operational Query Execution                       │
│  • Deep Q-Network for query optimization                    │
│  • Real-time execution decisions                            │
│  • Experience replay and target networks                    │
└─────────────────────────────────────────────────────────────┘
```

## System Requirements

### Hardware
- **OS**: Windows 10/11 (64-bit), Linux, or macOS
- **RAM**: 16 GB minimum (32 GB recommended)
- **Storage**: 50 GB available space
- **CPU**: 4+ cores recommended
- **Network**: Internet connection for initial setup

### Software Dependencies
- Python 3.10 or later
- PostgreSQL 14 or later
- Git

## Quick Start

### 1. Install PostgreSQL

**Windows:**
```bash
# Download from https://www.postgresql.org/download/windows/
# Run installer and note your postgres password

# Add to PATH
setx PATH "%PATH%;C:\Program Files\PostgreSQL\14\bin"
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql-14 postgresql-contrib
sudo systemctl start postgresql
```

**macOS:**
```bash
brew install postgresql@14
brew services start postgresql@14
```

### 2. Clone and Setup Python Environment

```bash
# Clone repository
git clone https://github.com/alessoh/self-improving-db-optimizer.git
cd self-improving-db-optimizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure System

```bash
# Copy example configuration
cp config.yaml.example config.yaml

# Edit config.yaml and set your PostgreSQL password
# Replace "CHANGE_ME" with your actual password
```

### 4. Initialize Database

```bash
# This creates tables and loads ~10M rows of synthetic data
# Expected time: 10-15 minutes
python setup_database.py

# Verify setup
python setup_database.py --verify
```

### 5. Run Demonstration

**Quick Test (5 minutes):**
```bash
python run_demo.py --duration 0.1 --fast-mode
```

**Full Simulation (2 weeks compressed):**
```bash
python run_demo.py --duration 14
```

**Custom Configuration:**
```bash
python run_demo.py --duration 7 --queries 500 --workload analytical
```

### 6. Monitor with Dashboard

Open a new terminal window:
```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Start dashboard
python dashboard.py
```

Access at: http://localhost:5000

## Project Structure

```
self-improving-db-optimizer/
│
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── README.md               # This file
│
├── main.py                 # System orchestrator
├── setup_database.py       # Database initialization
├── run_demo.py            # Demonstration runner
├── dashboard.py           # Web dashboard server
│
├── core/                  # Core learning components
│   ├── query_optimizer.py    # Level 0: RL agent
│   ├── policy_learner.py     # Level 1: Policy optimization
│   ├── meta_learner.py       # Level 2: Meta-learning
│   ├── models.py             # Neural network architectures
│   └── safety_monitor.py     # Safety & monitoring
│
├── database/              # Database management
│   ├── database_manager.py   # Connection & query execution
│   ├── workload_generator.py # Synthetic workload
│   └── schema.sql           # Database schema
│
├── telemetry/            # Metrics & monitoring
│   ├── collector.py         # Real-time collection
│   └── storage.py          # Persistent storage
│
├── dashboard/            # Web interface
│   ├── templates/
│   │   └── index.html      # Dashboard UI
│   └── static/
│       ├── style.css       # Styling
│       └── app.js         # Frontend logic
│
├── utils/                # Utilities
│   ├── logger.py           # Logging configuration
│   └── metrics.py         # Performance calculations
│
└── data/                 # Generated data (created on setup)
    ├── telemetry.db        # SQLite telemetry database
    ├── policies/          # Saved policy checkpoints
    └── logs/             # System logs
```

## Configuration

The `config.yaml` file controls all system parameters:

### Database Configuration
```yaml
database:
  host: localhost
  port: 5432
  user: postgres
  password: YOUR_PASSWORD
  dbname: query_optimizer_demo
```

### Learning Parameters
```yaml
level0:  # Reinforcement Learning
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 64
  buffer_size: 10000

level1:  # Policy Learning
  enabled: true
  update_interval: 3600
  min_improvement: 0.02

level2:  # Meta-Learning
  enabled: true
  evaluation_interval: 86400
  population_size: 5
```

### Safety Configuration
```yaml
safety:
  enabled: true
  query_timeout: 30
  memory_limit_gb: 2
  rollback_threshold: 0.15
```

## Usage Examples

### Basic Demonstration
```bash
# Run full 2-week simulation
python run_demo.py --duration 14
```

### Fast Testing
```bash
# Quick 1-hour test with 100x time acceleration
python run_demo.py --duration 0.04 --fast-mode
```

### Custom Workload
```bash
# 7-day analytical workload with 500 queries/hour
python run_demo.py --duration 7 --queries 500 --workload analytical
```

### Programmatic Usage
```python
from main import SystemOrchestrator

# Initialize system
orchestrator = SystemOrchestrator('config.yaml')
orchestrator.initialize_components()

# Run for specified duration
orchestrator.start(duration_days=14, fast_mode=False)
```

## Performance Monitoring

### Command-Line Metrics
```bash
# View telemetry summary
python -c "from telemetry.storage import TelemetryStorage; \
          ts = TelemetryStorage({'paths': {'telemetry_db': 'data/telemetry.db'}}); \
          ts.print_summary()"
```

### Web Dashboard
The dashboard provides real-time visualization of:
- Query latency trends
- Learning progress metrics
- Phase-by-phase comparison
- Latency distribution percentiles
- Policy update history
- Safety monitor status

### Log Files
- `data/logs/optimizer.log` - Detailed system logs
- `data/logs/errors.log` - Error tracking
- `data/final_report.txt` - Performance summary

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Verify PostgreSQL is running
psql --version
pg_isready

# Test connection
psql -h localhost -U postgres -d postgres
```

### Python Package Installation Errors
```bash
# Windows: Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch for Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Reduce `workload.queries_per_hour` in `config.yaml`
- Decrease `level0.buffer_size` for smaller memory footprint
- Lower `database.max_connections`

### Database Initialization Hangs
- Reduce scale factor: `python setup_database.py --scale 0.5`
- Check available disk space
- Verify PostgreSQL settings allow sufficient memory

## Development

### Running Tests
```bash
# Install development dependencies
pip install pytest pytest-cov pytest-mock

# Run test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Style
```bash
# Install formatters
pip install black isort flake8

# Format code
black .
isort .

# Check style
flake8 .
```

## Technical Details

### Reinforcement Learning (Level 0)
- **Algorithm**: Deep Q-Network (DQN) with experience replay
- **State Space**: Database metrics, cache statistics, query characteristics
- **Action Space**: Query execution strategies (index usage, join methods, parallelization)
- **Reward Function**: Weighted combination of latency, resource usage, and success rate

### Policy Learning (Level 1)
- **Method**: Gradient-based policy optimization
- **Update Frequency**: Configurable (default: hourly)
- **Validation**: Statistical testing before deployment
- **Rollback**: Automatic reversion on performance degradation

### Meta-Learning (Level 2)
- **Algorithm**: Genetic algorithm for hyperparameter optimization
- **Population**: Multiple hyperparameter configurations
- **Fitness**: Multi-objective optimization (speed, stability, resource efficiency)
- **Evolution**: Selection, crossover, mutation with elitism

### Safety Mechanisms
- **Resource Monitoring**: CPU, memory, and connection limits
- **Performance Tracking**: Baseline comparison with automatic rollback
- **Query Timeout**: Prevents runaway queries
- **Validation Sandbox**: Tests changes before production deployment

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- New features include tests
- Documentation is updated

## Citation

If you use this work in your research, please cite:

```bibtex
@software{self_improving_db_optimizer,
  author = {Alesso, Harry Peter},
  title = {Self-Improving Database Query Optimizer},
  year = {2025},
  url = {https://github.com/alessoh/self-improving-db-optimizer}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PostgreSQL community for the excellent database system
- PyTorch team for the deep learning framework
- Research in reinforcement learning and meta-learning that inspired this architecture

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/alessoh/self-improving-db-optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alessoh/self-improving-db-optimizer/discussions)
- **Author**: Harry Peter Alesso

## Roadmap

- [ ] Multi-database support (MySQL, SQLite)
- [ ] Distributed training capabilities
- [ ] Advanced visualization tools
- [ ] Cloud deployment templates
- [ ] REST API for external integration
- [ ] Real-world workload adapters

---

**Note**: This is a research demonstration system. For production use, conduct thorough testing and validation in your specific environment.