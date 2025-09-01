# SYNCO module for DrugLogics pipeline

## Description
SYNCO (SYNergy COnvergency) is a Python package for the analysis of synergistic drug responses, predicted using the DrugLogics software (See: https://github.com/druglogics) and compared to experimental observations. It integrates pipeline predictions (DrugLogics drabme outputs) and the experimental results (synergy scores) and harmonises both results, enabling the calculation of different decision-analytic metrics, such as accuracy, precision, and recall, in addition to Receiver-operating characteristic curves (ROC), Precision-Recall curves (PR), and ring plots summarising the confusion matrix results at a given threshold.

Status: early pre-release. Interfaces may evolve.

---

## Features
- Extract and unify of predictions and experimental observations
- Harmonise drug profiles for predictions and experimental combinations
- Converge results in clean data frames
- Compare results in terms of accuracy, recall and precision
- Calculate metrics for ROC and PR

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

### Option B (Only once the package is published to PyPI.)
```bash
pip install synco
```

---

## 🚀 Quick Start
### Notebook workflow (exploratory)

1. Create and activate the environment (see Installation).

2. Open the notebook "synco_plots.ipynb":

3. Prepare the CONFIG to read your data and options:
    - paths: base, pipeline_runs, input, output
    - general: cell_lines, run_date, verbose
    - compare: prediction_method (DrugLogics or BooLEVARD), threshold, synergy_column, analysis_mode (inhibitor_combination or cell_line)

4. Run next cells to build and extract results, make ring plots and ROC or PR curves

---

## Authors
Developed by Viviam Solangeli Bermudez Paiva under the FLobak Lab, https://github.com/druglogics, Norwegian University of Science and Technology - NTNU.

## License
This project is licensed under the [License Name]. See the `LICENSE` file for details.

## Contact
For questions or feedback, please contact Viviam Bermudez at viviamsb@ntnu.no.