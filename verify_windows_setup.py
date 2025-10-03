#!/usr/bin/env python
"""
Windows Setup Verification and Diagnostic Tool
Checks all requirements and identifies configuration issues
"""

import os
import sys
import subprocess
from pathlib import Path
import platform

# Color output for Windows
try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN
    RED = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    RESET = colorama.Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = RESET = ""


def print_header(text):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_result(check_name, passed, details=""):
    """Print check result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {check_name}")
    if details:
        print(f"       {details}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 10)
    passed = version >= required
    details = f"Found: {version.major}.{version.minor}.{version.micro}, Required: 3.10+"
    return passed, details


def check_os():
    """Check operating system."""
    system = platform.system()
    passed = system == "Windows"
    details = f"OS: {system} {platform.release()}"
    return passed, details


def check_command_exists(command):
    """Check if a command exists in PATH."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return True, result.stdout.split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False, "Not found in PATH"


def check_postgresql():
    """Check PostgreSQL installation."""
    passed, details = check_command_exists("psql")
    
    if passed:
        # Try to check service status
        try:
            result = subprocess.run(
                ["sc", "query", "postgresql-x64-14"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "RUNNING" in result.stdout:
                details += " (Service: RUNNING)"
            else:
                details += f" {YELLOW}(Service: NOT RUNNING){RESET}"
                passed = False
        except:
            details += " (Service status unknown)"
    
    return passed, details


def check_port_available(port):
    """Check if a port is available."""
    try:
        result = subprocess.run(
            ["netstat", "-an"],
            capture_output=True,
            text=True,
            timeout=5
        )
        listening = f":{port}" in result.stdout
        if listening:
            return False, f"Port {port} is already in use"
        return True, f"Port {port} is available"
    except:
        return None, f"Could not check port {port}"


def check_directory_structure():
    """Check project directory structure."""
    project_root = Path.cwd()
    
    required_files = [
        "config.yaml.example",
        "requirements.txt",
        "main.py",
        "setup_database.py",
        "run_demo.py",
        "dashboard.py"
    ]
    
    required_dirs = [
        "core",
        "database",
        "telemetry",
        "utils",
        "dashboard"
    ]
    
    issues = []
    
    # Check files
    for file in required_files:
        if not (project_root / file).exists():
            issues.append(f"Missing file: {file}")
    
    # Check directories
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            issues.append(f"Missing directory: {dir_name}")
    
    if issues:
        return False, " | ".join(issues)
    return True, "All required files and directories present"


def check_config_file():
    """Check if config.yaml exists and is valid."""
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        return False, "config.yaml not found (copy from config.yaml.example)"
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check for placeholder password
        if config.get('database', {}).get('password') == 'CHANGE_ME':
            return False, "Database password still set to CHANGE_ME"
        
        # Check dashboard host
        if config.get('dashboard', {}).get('host') not in ['127.0.0.1', 'localhost']:
            return None, f"{YELLOW}Warning: dashboard.host should be 127.0.0.1 for Windows{RESET}"
        
        return True, "Configuration file valid"
    except Exception as e:
        return False, f"Error reading config: {str(e)}"


def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        ('psycopg2', 'psycopg2-binary'),
        ('yaml', 'pyyaml'),
        ('flask', 'flask'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm')
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All required packages installed"


def check_virtual_environment():
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        return True, f"Virtual environment: {sys.prefix}"
    return False, "Not running in virtual environment (recommended)"


def check_database_connection():
    """Check if can connect to PostgreSQL database."""
    try:
        import yaml
        import psycopg2
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            return None, "Cannot check: config.yaml not found"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        db_config = config.get('database', {})
        
        # Try to connect
        conn = psycopg2.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password'),
            database='postgres',  # Connect to default database
            connect_timeout=10
        )
        conn.close()
        
        # Check if demo database exists
        conn = psycopg2.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password'),
            database='postgres',
            connect_timeout=10
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_config.get('dbname', 'query_optimizer_demo'),)
        )
        db_exists = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        
        if db_exists:
            return True, "Database connection successful and demo database exists"
        return None, f"{YELLOW}Connection OK, but database not initialized (run setup_database.py){RESET}"
        
    except ImportError as e:
        return None, f"Cannot check: {str(e)}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def check_data_directories():
    """Check if data directories exist."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        return False, "data/ directory does not exist (will be created on first run)"
    
    subdirs = ["logs", "policies"]
    existing = [d for d in subdirs if (data_dir / d).exists()]
    
    if existing:
        return True, f"data/ directory exists with: {', '.join(existing)}"
    return None, f"{YELLOW}data/ exists but subdirectories not created yet{RESET}"


def main():
    """Run all checks."""
    print_header("Windows Setup Verification for Self-Improving DB Optimizer")
    
    print("Platform Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version}")
    print(f"  Working Directory: {Path.cwd()}")
    
    checks = [
        ("Operating System (Windows)", check_os),
        ("Python Version (3.10+)", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("PostgreSQL Installation", check_postgresql),
        ("Python Packages", check_python_packages),
        ("Project Directory Structure", check_directory_structure),
        ("Configuration File", check_config_file),
        ("Port 5432 (PostgreSQL)", lambda: check_port_available(5432)),
        ("Port 5000 (Dashboard)", lambda: check_port_available(5000)),
        ("Database Connection", check_database_connection),
        ("Data Directories", check_data_directories),
    ]
    
    print_header("Running Checks")
    
    results = []
    for check_name, check_func in checks:
        try:
            passed, details = check_func()
            print_result(check_name, passed, details)
            results.append((check_name, passed, details))
        except Exception as e:
            print_result(check_name, False, f"Error: {str(e)}")
            results.append((check_name, False, str(e)))
    
    # Summary
    print_header("Summary")
    
    passed_count = sum(1 for _, passed, _ in results if passed is True)
    failed_count = sum(1 for _, passed, _ in results if passed is False)
    warning_count = sum(1 for _, passed, _ in results if passed is None)
    
    print(f"Checks Passed:  {GREEN}{passed_count}{RESET}")
    print(f"Checks Failed:  {RED}{failed_count}{RESET}")
    print(f"Warnings:       {YELLOW}{warning_count}{RESET}")
    
    # Recommendations
    if failed_count > 0:
        print_header("Recommended Actions")
        
        for check_name, passed, details in results:
            if passed is False:
                print(f"\n{RED}✗{RESET} {check_name}")
                print(f"   Issue: {details}")
                
                # Provide specific recommendations
                if "Python Version" in check_name:
                    print("   Action: Install Python 3.10 or higher from python.org")
                elif "PostgreSQL" in check_name:
                    print("   Action: Install PostgreSQL 14+ from postgresql.org/download/windows/")
                    print("   Action: Ensure PostgreSQL service is running: net start postgresql-x64-14")
                elif "Python Packages" in check_name:
                    print("   Action: Install packages: pip install -r requirements.txt")
                elif "config.yaml" in check_name:
                    if "not found" in details:
                        print("   Action: Copy config: copy config.yaml.example config.yaml")
                    if "CHANGE_ME" in details:
                        print("   Action: Edit config.yaml and set your PostgreSQL password")
                elif "Database Connection" in check_name:
                    print("   Action: Verify PostgreSQL is running and credentials are correct")
                    print("   Action: Test manually: psql -U postgres -h localhost")
                elif "Port" in check_name and "in use" in details:
                    print(f"   Action: Find process: netstat -ano | findstr :{5432 if '5432' in check_name else 5000}")
                    print("   Action: Stop the process or choose a different port")
    
    print()
    
    if failed_count == 0:
        print(f"{GREEN}All critical checks passed! System is ready to run.{RESET}")
        print("\nNext steps:")
        print("  1. If not done: python setup_database.py")
        print("  2. Run quick test: python run_demo.py --duration 0.1 --fast-mode")
        print("  3. Start dashboard: python dashboard.py")
        return 0
    else:
        print(f"{RED}Please fix the failed checks above before proceeding.{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())