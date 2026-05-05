# SYNCO Installation Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation from Source](#installation-from-source)
3. [Verify Installation](#verify-installation)
4. [Troubleshooting](#troubleshooting)
5. [Python Version Support](#python-version-support)

---

## Prerequisites

Before installing SYNCO, ensure you have the following:

- **Python 3.10 or higher** (recommended: Python 3.11)
- **pip** (Python package manager, usually included with Python)
- **git** (for cloning the repository)
- **~200 MB disk space** for the package and dependencies

### Check Your Python Version

```bash
python --version
```

If you need to install Python, visit [python.org](https://www.python.org/downloads/).

---

## Installation from Source

This is the recommended installation method while SYNCO is in early development.

### Step 1: Clone the Repository

```bash
git clone https://github.com/ViviamSB/SYNCO
cd synco
```

### Step 2: Create a Virtual Environment

A virtual environment isolates SYNCO's dependencies from your system Python. Choose **one** of the following options:

#### Option A: Using `conda` (Recommended for data science workflows)

```bash
# Create a new conda environment with Python 3.11
conda create -n synco python=3.11 -y

# Activate the environment
conda activate synco
```

#### Option B: Using Python's built-in `venv`

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip and Install SYNCO

```bash
# Upgrade pip to the latest version
pip install -U pip

# Install SYNCO and all dependencies in editable mode
pip install -e .
```

The `-e` (editable) flag allows you to use the latest code directly from the repository. Any changes to the source code will be immediately available without reinstalling.

### Step 4: Verify Installation

```bash
# Check if SYNCO is installed
python -c "import synco; print(synco.__version__)"

# Check if the CLI is available
synco --help
```

---

## Verify Installation

### Run a Quick Test

1. **Check module imports:**
   ```bash
   python -c "from synco.dashboard import create_app; print('Dashboard ready!')"
   ```

2. **Check CLI availability:**
   ```bash
   synco --help
   ```
   You should see the help message for the SYNCO command-line interface.

3. **Launch the dashboard (optional):**
   ```bash
   python -m synco.dashboard
   ```
   The dashboard should start on `http://127.0.0.1:8050/` (open this URL in your browser).

---

## Troubleshooting

### Problem: Python version not supported

**Error:** `Python X.X is not supported. Please use Python 3.10 or higher.`

**Solution:**
- Install Python 3.11: https://www.python.org/downloads/
- Or use conda: `conda install python=3.11 -y`

### Problem: Virtual environment not activating

**Error:** `command not found: synco` or `ModuleNotFoundError: No module named 'synco'`

**Solution:**
- Ensure your virtual environment is activated. You should see `(synco)` in your terminal prompt.
- Try reactivating:
  ```bash
  # macOS/Linux:
  source .venv/bin/activate

  # Windows:
  .venv\Scripts\activate
  ```

### Problem: Dependency conflict

**Error:** `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Solution:**
- This is typically a warning and can be safely ignored.
- If you encounter module import errors, try a clean install:
  ```bash
  pip install --force-reinstall -e .
  ```

### Problem: Permission denied on macOS/Linux

**Error:** `Permission denied` when running `pip install`

**Solution:**
- Ensure you're using a virtual environment (don't use system Python with `sudo`).
- Activate your virtual environment and try again:
  ```bash
  source .venv/bin/activate
  pip install -e .
  ```

### Problem: Dashboard fails to start

**Error:** `Address already in use` or port 8050 is not accessible

**Solution:**
- Use a different port:
  ```bash
  python -m synco.dashboard --port 8080
  ```
- Or kill the process occupying the port:
  ```bash
  # macOS/Linux:
  lsof -ti:8050 | xargs kill -9

  # Windows (PowerShell):
  Get-Process | Where-Object {$_.Handles -match "8050"} | Stop-Process
  ```

---

## Python Version Support

SYNCO requires **Python 3.10 or higher**. The following versions are tested and supported:

- ✅ Python 3.11 (Recommended)
- ✅ Python 3.10
- ✅ Python 3.12
- ❌ Python 3.9 and below

### Check Current Python Version

```bash
python --version
python3 --version
```

---

## Next Steps

After successful installation, proceed to:

1. **[QUICKSTART.md](QUICKSTART.md)** – Get started with the dashboard and workflows
2. **[config_guide.md](config_guide.md)** – Learn configuration options
3. **[roc_pr_metrics_guide.md](roc_pr_metrics_guide.md)** – Understand metrics and analysis modes

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review the example configuration in `examples/synco_example_config.json`
3. Contact the maintainer: viviamsb@ntnu.no

---

**Status:** SYNCO is in early pre-release. If you find installation issues, please report them to viviamsb@ntnu.no.
