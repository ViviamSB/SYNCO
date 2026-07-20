# SYNCO module for DrugLogics pipeline

## Description
SYNCO (SYNergy COnvergency) is a Python package for the analysis of synergistic drug responses, predicted using the DrugLogics software (See: https://github.com/druglogics) and compared to experimental observations. It integrates pipeline predictions (DrugLogics drabme outputs) and the experimental results (synergy scores) and harmonises both results, enabling the calculation of different decision-analytic metrics, such as accuracy, precision, and recall, in addition to Receiver-operating characteristic curves (ROC), Precision-Recall curves (PR), and ring plots summarising the confusion matrix results at a given threshold.

SYNCO now features an **interactive web-based dashboard** for comprehensive data exploration, real-time visualization, and dynamic analysis of results across multiple tissues and cell lines.

Status: early pre-release. Interfaces may evolve.

---

## Features
- Extract and unify predictions and experimental observations
- Harmonise drug profiles for predictions and experimental combinations
- Converge results in clean data frames
- Compare results in terms of accuracy, recall and precision
- Calculate metrics for ROC and Precision-Recall curves
- **Interactive Dashboard** with:
  - Multi-tissue analysis across cell lines
  - Real-time classification tables
  - Performance metrics (sensitivity, specificity, accuracy)
  - ROC/PR curve visualization
  - Distribution analysis of predictions and ground truth
  - Drug profile comparisons
  - Cross-tissue synchronization analysis

---

## Installation

### Option A — from source (recommended for now)
```bash
# 1) Clone the repository
git clone https://github.com/ViviamSB/SYNCO
cd synco

# 2) Create a fresh environment (conda or venv)
# Using conda:
conda create -n synco python=3.11 -y
conda activate synco

# Or using venv:
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install in editable mode
pip install -U pip
pip install -e .
```

For detailed setup instructions, see [INSTALLATION.md](INSTALLATION.md).

### Option B (Only once the package is published to PyPI.)
```bash
pip install synco
```

---

## 🚀 Quick Start

For detailed workflows and examples, see [QUICKSTART.md](QUICKSTART.md).

### Dashboard (Interactive Web Interface) — Recommended for exploration
```bash
# 1. Create and activate the environment (see Installation).

# 2. Launch the dashboard:
python -m synco.dashboard

# 3. Open your browser to http://127.0.0.1:8050/
```

The dashboard provides:
- **Setup Page**: Configure paths and load your pipeline results
- **Data Page**: Inspect and manage loaded datasets
- **Explorer**: Dynamic visualization with tissue/cell-line filtering

### Notebook workflow (exploratory)

1. Create and activate the environment (see Installation).

2. Open one of the notebooks in the exmaple folder

3. Prepare the CONFIG to read your data and options:
    - paths: base, pipeline_runs, input, output
    - general: cell_lines, run_date, verbose
    - compare: prediction_method (DrugLogics or BooLEVARD), threshold, synergy_column, analysis_mode (inhibitor_combination or cell_line)

4. Run next cells to build and extract results, make ring plots and ROC or PR curves

### CLI — run from terminal

You can run SYNCO either with a configuration file (JSON or YAML) or using a lightweight direct-arguments mode.

- Config-file mode (recommended for reproducibility):
```powershell
python -m synco -c examples/synco_example_config.json --plan
python -m synco -c examples/synco_example_config.json
```

- Direct-args mode (no config file):
```powershell
python -m synco --base data/DrugLogics \
    --pipeline-runs data/sample_raw/20250804/drabme_out \
    --input data/synco_input \
    --output results \
    --cell-lines C2BBE1,CAR1,T84 \
    --prediction-method DrugLogics \
    --plan
```

Optional override: specify the exact experimental synergies file to use with `--synergies_filename` (accepts a filename relative to the `--input` folder or an absolute path). This sets `steps.data_loading.synergy_filename` in the merged config and takes precedence over the default pattern (`synergies_observed*.csv`).

Example (override):
```powershell
python -m synco -c examples/synco_example_config.json --synergies_filename data/synco_input/labdata_nochemo_hsa.csv --plan
```


## Authors
Developed by Viviam Solangeli Bermudez Paiva under the FLobak Lab, https://github.com/druglogics, Norwegian University of Science and Technology - NTNU.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please contact Viviam Bermudez at viviam.bermudez@ntnu.no or viviambermudez@gmail.com