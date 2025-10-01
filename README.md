# Self-Improving Database Query Optimizer

## A 3-level autonomous learning system that optimizes database query performance without manual tuning.

## Overview
This system demonstrates three levels of autonomous learning:

Level 0 (Operational): AI executes database operations using reinforcement learning

Level 1 (Tactical): System continuously improves operational policies based on performance

Level 2 (Strategic): Meta-learner optimizes the learning process itself

## The opportunity lies in building a database architecture
with three levels of autonomous learning. 

### The first level
 is the operational level where AI models would manage core database functions—query optimization through reinforcement learning agents trained on actual execution patterns, index management through deep neural networks that learn from workload history, and resource allocation through multi-agent systems. This represents an evolution beyond current rule-based approaches toward systems that learn from experience.
### The second level 
is the policy learning level where the system would analyze execution telemetry to continuously improve its operational policies. Rather than waiting for human administrators to identify and fix performance problems, the database would update its decision models based on observed outcomes, validating changes through statistical testing before deployment. This continuous improvement loop would be governed by strict safety mechanisms including execution sandboxes, cost estimation, and automatic rollback capabilities to prevent self-damage.
### The third level 
is the meta-learning level where the system would evaluate its own learning effectiveness and redesign its neural architecture when learning plateaus. Using techniques like neural architecture search documented in academic research, the system could discover improved model structures. It would optimize reward functions to align with business objectives and auto-tune hyperparameters through population-based training approaches established in machine learning literature. This represents genuine self-improvement rather than parameter adjustment within fixed architectures.
## This pilot project system runs entirely on a Windows laptop using PostgreSQL and learns to optimize query execution without human intervention.
System Requirements
Hardware Requirements
Windows 10 or 11 (64-bit)
16 GB RAM minimum (32 GB recommended)
50 GB free disk space
4+ CPU cores recommended
Internet connection for initial setup
Software Prerequisites
Python 3.10 or later
PostgreSQL 14 or later
Git (for cloning the repository)
Visual Studio C++ Build Tools (for some Python packages)
## Installation Guide
Step 1: Install PostgreSQL
Download PostgreSQL from https://www.postgresql.org/download/windows/
Run the installer (postgresql-14.x-windows-x64.exe or later)
During installation:
## Note your superuser password (you'll need this)
Accept default port 5432
Accept default locale
## Install Stack Builder components when prompted
Add PostgreSQL to your PATH:
Open System Properties → Environment Variables
Add C:\Program Files\PostgreSQL\14\bin to PATH
Verify installation:
## Quick Start
# 1. Install PostgreSQL 14+ and set password
# 2. Clone repo and setup
git clone https://github.com/alessoh/self-improving-db-optimizer
cd self-improving-db-optimizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure
copy config.yaml.example config.yaml
# Edit config.yaml - set your PostgreSQL password

# 4. Initialize database (10-15 minutes)
python setup_database.py

# 5. Run demo
python run_demo.py --duration 0.1 --fast-mode  # 5-minute test
python run_demo.py --duration 14                # Full 2-week simulation

Level 1 (Tactical)**: System continuously improves operational policies based on performance
- **Level 2 (Strategic)**: Meta-learner optimizes the learning process itself

The system runs entirely on a Windows laptop using PostgreSQL and learns to optimize query execution without human intervention.

## System Requirements

### Hardware Requirements
- Windows 10 or 11 (64-bit)
- 16 GB RAM minimum (32 GB recommended)
- 50 GB free disk space
- 4+ CPU cores recommended
- Internet connection for initial setup

### Software Prerequisites
- Python 3.10 or later
- PostgreSQL 14 or later
- Git (for cloning the repository)
- Visual Studio C++ Build Tools (for some Python packages)

## Installation Guide

### Step 1: Install PostgreSQL

1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Run the installer (postgresql-14.x-windows-x64.exe or later)
3. During installation:
   - Note your superuser password (you'll need this)
   - Accept default port 5432
   - Accept default locale
   - Install Stack Builder components when prompted
4. Add PostgreSQL to your PATH:
   - Open System Properties → Environment Variables
   - Add `C:\Program Files\PostgreSQL\14\bin` to PATH
5. Verify installation:
   psql --version
Step 2: Install Python Dependencies

Clone this repository:

git clone https://github.com/yourusername/self-improving-db-optimizer.git
   cd self-improving-db-optimizer

Create a virtual environment:

 python -m venv venv
   venv\Scripts\activate

Install required packages:

pip install --upgrade pip
   pip install -r requirements.txt
If you encounter issues with PyTorch on Windows:
bash   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Step 3: Configure the System

Copy the example configuration:

bash   copy config.yaml.example config.yaml

Edit config.yaml with your PostgreSQL credentials:

yaml   database:
     host: localhost
     port: 5432
     user: postgres
     password: YOUR_POSTGRES_PASSWORD  # Change this!
     dbname: query_optimizer_demo
Step 4: Initialize the Database
Run the database setup script:
bashpython setup_database.py
This will:

### Create the database query_optimizer_demo
Generate sample tables (customers, orders, products, suppliers, regions, transactions)
Load approximately 10 million rows of synthetic data
Create necessary indexes
Initialize the telemetry database

### Note: Initial setup takes 10-15 minutes depending on your system.
Verify the setup:
bashpython setup_database.py --verify
Step 5: Run the Demonstration
Start the complete 2-week simulation:
bashpython run_demo.py --duration 14
For a quick test (2-hour simulation):
bashpython run_demo.py --duration 0.1 --fast-mode
The system will:

### Establish baseline performance (no learning)
Activate Level 0 operational learning
Enable Level 1 policy optimization
Engage Level 2 meta-learning
Display results and performance improvements

## Step 6: Monitor the Dashboard
While the system is running, access the web dashboard:

Open another terminal and activate the virtual environment:

venv\Scripts\activate

Start the dashboard:

python dashboard.py

Open your browser to: http://localhost:5000

The dashboard shows:

Real-time performance metrics
Learning curves
Policy update history
Safety monitor status
Meta-learner decisions

Project Structure
self-improving-db-optimizer/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── config.yaml.example      # Example configuration
│
├── main.py                  # Main orchestrator
├── setup_database.py        # Database initialization
├── run_demo.py             # Demonstration script
│
├── core/
│   ├── __init__.py
│   ├── query_optimizer.py   # Level 0: RL agent
│   ├── policy_learner.py    # Level 1: Policy optimization
│   ├── meta_learner.py      # Level 2: Meta-learning
│   ├── models.py            # Neural network definitions
│   └── safety_monitor.py    # Safety mechanisms
│
├── database/
│   ├── __init__.py
│   ├── database_manager.py  # Database operations
│   ├── workload_generator.py # Synthetic workload
│   └── schema.sql           # Database schema
│
├── telemetry/
│   ├── __init__.py
│   ├── collector.py         # Metrics collection
│   └── storage.py          # Telemetry storage
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py              # Flask application
│   ├── templates/          # HTML templates
│   │   └── index.html
│   └── static/            # CSS, JS, images
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── charts.js
│
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Logging utilities
│   └── metrics.py         # Performance metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_optimizer.py
│   ├── test_learner.py
│   └── test_safety.py
│
└── data/                   # Generated data directory
    ├── telemetry.db       # SQLite telemetry database
    ├── policies/          # Saved policy checkpoints
    └── logs/             # System logs
Running Tests
Run the test suite to verify installation: