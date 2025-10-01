# self-improving-db-optimizer
AI self-improving db optimizer
# Self-Improving Database Query Optimizer

A demonstration of autonomous, self-improving database intelligence using multi-level machine learning on a single Windows laptop.

## Overview

This system demonstrates three levels of autonomous learning:
- **Level 0 (Operational)**: AI executes database operations using reinforcement learning
- **Level 1 (Tactical)**: System continuously improves operational policies based on performance
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
```bash
   psql --version