# SYNCO Quick Start Guide

## Table of Contents

1. [Dashboard Workflow (Recommended)](#dashboard-workflow-recommended)
2. [Command-Line Interface (CLI)](#command-line-interface-cli)
3. [Notebook Workflow](#notebook-workflow)
4. [Understanding Dashboard Tabs](#understanding-dashboard-tabs)
5. [Tips and Tricks](#tips-and-tricks)
6. [Common Use Cases](#common-use-cases)

---

## Dashboard Workflow (Recommended)

The interactive dashboard is the easiest way to explore SYNCO results without writing code.

### Launch the Dashboard

```bash
# Ensure your virtual environment is activated
conda activate synco
# or: source .venv/bin/activate

# Start the dashboard
python -m synco.dashboard
```

A message will appear:
```
  SYNCO Dashboard running on http://127.0.0.1:8050/
  Press Ctrl+C to stop.
```

Open your browser to **http://127.0.0.1:8050/**

### Dashboard Navigation

The dashboard has 3 main sections:

#### 1. **Setup Page** (⚙️ Setup tab)

Configure and load your data:

1. **Results Directory**: Path to your SYNCO output folder (e.g., `results/` or `synco_output/`)
2. **Cell Fate Directory**: Path to additional cell fate data (optional, for advanced analysis)
3. **Configuration**: Either:
   - Upload a JSON/YAML config file (`examples/synco_example_config.json`)
   - Use the form to specify paths and settings directly

**Form Fields:**
- **Base Path**: Root directory containing pipeline data
- **Pipeline Runs**: Folder with drabme output files
- **Input**: Folder with synergy experimental data
- **Output**: Where SYNCO results will be saved
- **Cell Lines**: Comma-separated list (e.g., `C2BBE1,CAR1,T84`)
- **Prediction Method**: `DrugLogics` or `BooLEVARD`
- **Threshold**: Synergy threshold value (default: 0.1)

**Actions:**
- Click **Plan** to preview what will be loaded
- Click **Run** to execute the analysis pipeline

#### 2. **Data Page** (📊 Data tab)

Inspect loaded datasets:

- View summary statistics
- Check loaded cell lines and combinations
- Review prediction vs. experimental data
- Export data tables (PNG, HTML, CSV)

#### 3. **Explorer Page** (📈 Explorer tab)

Dynamically visualize results with filtering:

**Tissue Selection:**
Select which tissue/cell-line to analyze. All plots update dynamically.

**Filter Controls** (changes based on active tab):
- Tissue selection
- Individual drug filters
- Combination filters
- Profile-based filters

**Five Analysis Tabs:**

##### **Classification Tab**
- interactive table showing predictions vs. ground truth
- Display class distributions
- Useful for understanding individual predictions

##### **Performance Tab**
- Sensitivity (recall)
- Specificity
- Accuracy
- Pie charts and bar charts

##### **ROC / PR Tab**
- Receiver Operating Characteristic (ROC) curves
- Precision-Recall (PR) curves
- AUC scores
- Threshold visualization

##### **Distributions Tab**
- Distribution of prediction scores
- Distribution of experimental synergy scores
- Histogram comparisons
- Support for multiple analysis modes

##### **Profiles Tab**
- Drug profile comparisons across tissues
- Individual drug pair analysis
- Cross-tissue consistency metrics

### Example Dashboard Workflow

1. **Open Dashboard**
   ```bash
   python -m synco.dashboard
   ```

2. **Go to Setup Page**
   - Load config: `examples/synco_example_config.json`
   - Click "Plan" to preview
   - Click "Run" to process data

3. **Go to Data Page**
   - Verify data loaded correctly
   - Check summary statistics

4. **Go to Explorer**
   - Select tissue from dropdown
   - Choose a tab (e.g., "ROC / PR")
   - Apply filters as needed
   - Visualizations update in real-time

---

## Command-Line Interface (CLI)

For reproducible, automated workflows, use the CLI.

### Basic Usage

```bash
synco --help
```

### Configuration-File Mode (Recommended)

Use a configuration file (JSON or YAML) for reproducibility:

```bash
# Preview what will be processed
python -m synco -c examples/synco_example_config.json --plan

# Run the analysis
python -m synco -c examples/synco_example_config.json
```

### Direct-Arguments Mode

Specify options directly without a config file:

```bash
python -m synco \
    --base data/DrugLogics \
    --pipeline-runs data/sample_raw/20250804/drabme_out \
    --input data/synco_input \
    --output results \
    --cell-lines C2BBE1,CAR1,T84 \
    --prediction-method DrugLogics \
    --threshold 0.1 \
    --plan
```

Then run without `--plan` to execute:

```bash
python -m synco \
    --base data/DrugLogics \
    --pipeline-runs data/sample_raw/20250804/drabme_out \
    --input data/synco_input \
    --output results \
    --cell-lines C2BBE1,CAR1,T84 \
    --prediction-method DrugLogics
```

### Common CLI Flags

- `--plan`: Dry-run mode. Shows what will be loaded without processing
- `-c / --config`: Path to JSON/YAML configuration file
- `--base`: Base directory path
- `--pipeline-runs`: Path to drabme output folder
- `--input`: Path to synergy input data
- `--output`: Path to output directory
- `--cell-lines`: Comma-separated cell line names
- `--prediction-method`: `DrugLogics` or `BooLEVARD`
- `--threshold`: Synergy threshold (float)
- `--synergies_filename`: Override synergy filename to load

### Override Synergy File

Specify a custom synergy input file:

```bash
python -m synco -c examples/synco_example_config.json \
    --synergies_filename data/synco_input/labdata_specific.csv \
    --plan
```

---

## Notebook Workflow

For exploratory analysis and custom plotting in Jupyter:

### Setup

1. Activate environment and open Jupyter:
   ```bash
   conda activate synco
   jupyter notebook
   ```

2. Open `synco_plots.ipynb`

### Configuration in Notebook

Create a CONFIG dictionary:

```python
CONFIG = {
    "paths": {
        "base": "examples/consensus/ovary/",
        "pipeline_runs": "examples/consensus/ovary/BL_output",
        "input": "examples/consensus/synco_input/",
        "output": "examples/consensus/ovary/synco_output/"
    },
    "general": {
        "cell_lines": ["AsPC-1", "BxPC-3", "CAPAN-1"],
        "run_date": None,
        "verbose": True
    },
    "compare": {
        "prediction_method": "BooLEVARD",
        "threshold": 0.1,
        "synergy_column": "synergy",
        "analysis_mode": "inhibitor_combination",
        "duplicate_strategy": "mean"
    }
}
```

### Workflow Steps

1. Create and activate environment
2. Define CONFIG (as above)
3. Build convergence: `conv = Convergence.from_config(CONFIG)`
4. Create comparison: `comp = Comparison(convergence=conv, **CONFIG["compare"])`
5. Generate plots:
   ```python
   from synco.plots import plot_roc, plot_ring

   # ROC/PR curves
   plot_roc(comp, show=True)

   # Ring plots (confusion matrix)
   plot_ring(comp, show=True)
   ```

---

## Understanding Dashboard Tabs

### Key Metrics Explained

**Sensitivity (Recall):** Proportion of true synergies correctly identified
- Formula: TP / (TP + FN)
- High sensitivity = few missed real synergies

**Specificity:** Proportion of true negatives correctly identified
- Formula: TN / (FP + TN)
- High specificity = few false alarms

**Accuracy:** Overall correctness
- Formula: (TP + TN) / Total

**ROC Curve:** Shows trade-off between sensitivity and false-positive rate
- Closer to top-left = better performance
- AUC = area under curve (1.0 = perfect, 0.5 = random)

**PR Curve:** Shows trade-off between precision and recall
- Useful when classes are imbalanced
- Higher curves = better performance

### Filtering in Explorer

- **Tissue Selection**: Changes which cell line/tissue is analyzed
- **Combination Filter**: Show results for specific drug pairs only
- **Drug Filter**: Filter by a single drug across all combinations
- **Profile Filter**: Filter by prediction/experimental profile type

Combined filters update all 5 tabs simultaneously.

---

## Tips and Tricks

### Speed Up Analysis

- For large datasets, specify fewer cell lines in config
- Use `--plan` mode first to verify configuration
- Cache results in output folder (reuse with same config)

### Reproducibility

- Always use config-file mode (`-c` flag)
- Document your config file in version control
- Include `run_date` in config for tracking

### Debugging

- Enable verbose mode: `"verbose": true` in config
- Check output folder for intermediate files
- Use `--plan` mode to see what's being loaded

### Export Results

From the dashboard:
1. Go to **Data page**
2. Find the table of interest
3. Click the camera icon to export as PNG
4. Or right-click for other format options

---

## Common Use Cases

### Use Case 1: Quick Visual Exploration

```bash
# 1. Start dashboard
python -m synco.dashboard

# 2. Upload config on Setup page
# 3. Click "Run"
# 4. Go to Explorer and filter interactively
```

### Use Case 2: Automated Batch Analysis

```bash
# Run multiple configurations automatically
for config in configs/*.json; do
    python -m synco -c "$config"
done
```

### Use Case 3: Compare Two Methods

```bash
# Run with DrugLogics
python -m synco -c config.json

# In the dashboard, compare against BooLEVARD results
# by loading a different config on Setup page
```

### Use Case 4: Fine-Tune Threshold

```bash
# Use dashboard to visually test different thresholds
# 1. Setup page: Load config
# 2. Modify threshold in form
# 3. Click "Run"
# 4. Explorer shows updated metrics
```

### Use Case 5: Publication-Ready Plots

```bash
# 1. Explore in dashboard to find best view
# 2. Export via Explorer's export tools
# 3. Or use notebook workflow for custom styling
```

---

## Troubleshooting

**Dashboard won't start?**
- Check firewall (port 8050 might be blocked)
- Try a different port: `python -m synco.dashboard --port 8080`

**Data not loading?**
- Verify paths in config file exist
- Check file naming conventions in `config_guide.md`
- Use `--plan` mode to see what's being searched for

**Plots not appearing?**
- Ensure data loaded successfully (check Data page)
- Verify cell lines are in the loaded data
- Try resetting filters: click "Reset" button

**Need more help?**
- See [INSTALLATION.md](INSTALLATION.md) for setup issues
- See [config_guide.md](config_guide.md) for configuration help
- See [roc_pr_metrics_guide.md](roc_pr_metrics_guide.md) for metrics details

---

**Next Steps:**
- Explore the [config_guide.md](config_guide.md) for advanced configuration options
- Read [roc_pr_metrics_guide.md](roc_pr_metrics_guide.md) for detailed metric definitions
- Review example configs in `examples/` folder

---

**Status:** SYNCO is in early pre-release. Interfaces may evolve. Feedback welcome: viviamsb@ntnu.no
