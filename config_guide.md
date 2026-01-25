# SYNCO Configuration Examples

## Table of Contents
1. [Cell Line Specification Options](#cell-line-specification-options)
2. [Comparison Configuration](#comparison-configuration)
3. [ROC Metrics Configuration](#roc-metrics-configuration)
4. [Complete Configuration Examples](#complete-configuration-examples)

---

## Cell Line Specification Options

The `cell_lines` field in the `general` section supports three flexible input methods:

### 1. Manual List (Explicit Cell Lines)

Specify cell lines directly as a list:

```json
{
  "paths": {
    "base": "examples/consensus/ovary/",
    "pipeline_runs": "examples/consensus/ovary/BL_output",
    "input": "examples/consensus/synco_input/",
    "output": "examples/consensus/ovary/synco_output/"
  },
  "general": {
    "cell_lines": ["AsPC-1", "BxPC-3", "CAPAN-1", "MIA-PaCa-2"],
    "run_date": null,
    "verbose": true
  },
  "compare": {
    "prediction_method": "BooLEVARD",
    "threshold": 0.1e-20,
    "synergy_column": "synergy",
    "analysis_mode": "inhibitor_combination",
    "duplicate_strategy": "mean"
  }
}
```

**Use case:** When you want to analyze a specific subset of cell lines.

---

### 2. CSV File (Load from File)

Specify a CSV filename or path:

```json
{
  "paths": {
    "base": "examples/consensus/ovary/",
    "pipeline_runs": "examples/consensus/ovary/BL_output",
    "input": "examples/consensus/synco_input/",
    "output": "examples/consensus/ovary/synco_output/"
  },
  "general": {
    "cell_lines": "cell_line_list.csv",
    "run_date": null,
    "verbose": true
  },
  "compare": {
    "prediction_method": "BooLEVARD",
    "threshold": 0.1e-20,
    "synergy_column": "synergy",
    "analysis_mode": "inhibitor_combination",
    "duplicate_strategy": "mean"
  }
}
```

**CSV File Format:**
The CSV file should have a column with cell line names. Common column names are automatically detected:
- `cell_line_name`
- `cell_line`
- `cellline`
- `name`
- Or the first column will be used

Example `cell_line_list.csv`:
```csv
cell_line_name,SIDM
AsPC-1,SIDM00899
BxPC-3,SIDM00132
CAPAN-1,SIDM00934
CAPAN-2,SIDM00943
MIA-PaCa-2,SIDM00505
```

**File Location:**
- If the filename is provided (e.g., `"cell_line_list.csv"`), it will be searched in the `input` directory
- You can also provide an absolute path (e.g., `"/full/path/to/cell_lines.csv"`)

**Use case:** When you have a standardized list of cell lines in a file, especially useful for reproducibility and sharing configurations.

---

### 3. Auto-Discovery (Automatic Detection)

Leave empty or set to null to automatically detect cell lines:

```json
{
  "paths": {
    "base": "examples/consensus/ovary/",
    "pipeline_runs": "examples/consensus/ovary/BL_output",
    "input": "examples/consensus/synco_input/",
    "output": "examples/consensus/ovary/synco_output/"
  },
  "general": {
    "cell_lines": [],
    "run_date": null,
    "verbose": true
  },
  "compare": {
    "prediction_method": "BooLEVARD",
    "threshold": 0.1e-20,
    "synergy_column": "synergy",
    "analysis_mode": "inhibitor_combination",
    "duplicate_strategy": "mean"
  }
}
```

Or:

```json
{
  "general": {
    "cell_lines": null,
    ...
  }
}
```

**How it works:**
- Automatically scans the `pipeline_runs` directory for subdirectories
- Each subdirectory (excluding common system folders) is treated as a cell line
- Also checks nested `drabme_out` folder (common in DrugLogics pipelines)

**Use case:** When processing all available cell lines in your pipeline results directory, or for exploratory analysis.

---

### Example 2: Production Analysis with Bootstrap CIs

```json
{
  "paths": {
    "base": "examples/oncologics/20251013/",
    "pipeline_runs": "examples/oncologics/20251013/drabme_out",
    "input": "examples/oncologics/synco_input/",
    "output": "examples/oncologics/20251013/synco_output/"
  },
  "general": {
    "cell_lines": "cell_lines.csv",
    "run_date": "20251013",
    "verbose": true
  },
  "compare": {
    "prediction_method": "DrugLogics",
    "threshold": 0.00001,
    "synergy_column": "synergy",
    "analysis_mode": "cell_line",
    "duplicate_strategy": "mean",
    "return_pair_details": true
  },
  "roc_metrics": {
    "threshold_offsets": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "n_bootstrap": 10000,
    "ci_level": 0.95
  }
}
```

### Example 3: Debugging Specific Cell Lines

```json
{
  "paths": {
    "base": "examples/consensus/colon/",
    "pipeline_runs": "examples/consensus/colon/BL_output",
    "input": "examples/consensus/synco_input/",
    "output": "examples/consensus/colon/synco_output/"
  },
  "general": {
    "cell_lines": ["HT115", "C2BBE1", "T84"],
    "run_date": null,
    "verbose": true
  },
  "compare": {
    "prediction_method": "BooLEVARD",
    "threshold": 0.1e-20,
    "synergy_column": "synergy",
    "analysis_mode": "cell_line",
    "duplicate_strategy": "mean",
    "debug_items": ["HT115", "C2BBE1"],
    "return_pair_details": true
  },
  "roc_metrics": {
    "threshold_offsets": [-5.0, -2.0, 0.0, 2.0, 5.0],
    "n_bootstrap": null,
    "ci_level": 0.95
  }
}
```

---

## Implementation Details

The cell line resolution happens in two stages:

1. **Configuration validation** (`build_pipeline_config`): Validates that `cell_lines` is either a list, string, or None
2. **Cell line resolution** (`resolve_cell_lines`): Resolves the actual cell line names based on the input type

The resolution is handled by the `resolve_cell_lines()` function in `synco/features/loader.py`, which:
- Returns manual lists as-is (if non-empty)
- Loads and parses CSV files (searching in the input directory)
- Discovers cell lines from pipeline_runs directory structure
- Provides clear error messages if resolution fails

---

## Error Handling

If cell lines cannot be resolved, you'll receive a helpful error message:

```
ValueError: Could not resolve cell lines. Please provide one of:
  1. A manual list in config['general']['cell_lines']
  2. A CSV filename/path in config['general']['cell_lines']
  3. Valid pipeline_runs path for auto-discovery
```

---

## Recommended Approach

### Cell Line Specification:
- **Development/Exploration:** Use auto-discovery (`[]`) to process all available data
- **Production/Specific Analysis:** Use CSV file for reproducibility and documentation
- **Quick Testing:** Use manual list for analyzing specific cell lines

### Comparison Strategy:
- **Standard Analysis:** Use `"mean"` for duplicate_strategy to handle technical replicates
- **Quick Exploration:** Set `return_pair_details: false` to skip detailed output
- **Troubleshooting:** Add specific cell lines to `debug_items` when investigating issues

### ROC Metrics:
- **First Run:** Skip bootstrap CIs (`n_bootstrap: null`) for faster results
- **Publication:** Enable bootstrap with `n_bootstrap: 10000` for robust confidence intervals
- **Sensitivity Analysis:** Use default `threshold_offsets` to understand threshold impact
- **Performance:** Disable threshold sweeps if not needed for faster execution

### Cell Line Name Handling:
- Cell line names are automatically cleaned (uppercase, hyphens removed)
- `"HT-115"` and `"HT115"` are treated as the same cell line
- `"C2-BBE1"` and `"C2BBE1"` are treated as the same cell line
- This ensures consistent matching between experimental and predicted data

---

## Output Files

With the recommended configuration, SYNCO generates a comprehensive set of outputs organized by pipeline step:

### Step 1: Drug Profile Mappings
Generated by the `create_drug_profiles()` step:

- `PD_inhibitors_dict.json` - Maps PD profile pairs to inhibitor group names
- `Drugnames_PD_dict.json` - Maps drug names to PD profile IDs
- `PD_drugnames_dict.json` - Maps PD profile IDs to drug names
- `inhibitorgroups_dict.json` - Maps inhibitor groups to their member drugs
- `Drugnames_inhibitor_dict.json` - Maps drug names to inhibitor groups
- `PD_targets_dict.json` - Maps PD profiles to target proteins/pathways
- `PD_mechanism_dict.json` - Maps PD profiles to mechanism of action (if available)

### Step 2: Synergy Predictions
Generated by the `create_synergy_predictions()` step:

- `synergy_predictions.csv` - Raw synergy predictions from the model

### Step 3: Synergy Convergence
Generated by the `converge_synergies()` step for both experimental and predicted data:

**Experimental:**
- `experimental_full_df.csv` - Complete experimental data with mappings
- `experimental_drug_names_synergies_df.csv` - Drug-level synergies (long format)
- `experimental_inhibitor_group_synergies_df.csv` - Inhibitor group-level synergies (long format)

**Predicted:**
- `predictions_full_df.csv` - Complete predictions with mappings
- `predictions_drug_names_synergies_df.csv` - Drug-level predictions (long format)
- `predictions_inhibitor_group_synergies_df.csv` - Inhibitor group-level predictions (long format)

### Step 4: Synergy Comparison
Generated by the `compare_synergies()` step:

- `{analysis_mode}_comparison_results.csv` - Confusion matrix per cell line/combination
  - Columns: True Positive, True Negative, False Positive, False Negative, Total, Match, Mismatch, Match %, Mismatch %, Accuracy, Recall, Precision
- `{analysis_mode}_pair_details.csv` - Long-form pair-level details (if enabled)
  - Columns: inhibitor_combination, cell_line, tissue, pd_combination, exp_synergy, pred_synergy, exp_binary, pred_binary, confusion_matrix_value
- `{analysis_mode}_comparison_summary.json` - Global metrics summary (JSON format)
- `{analysis_mode}_comparison_summary.txt` - Global metrics summary (human-readable text)

**Note:** `{analysis_mode}` will be either `cell_line` or `inhibitor_combination` depending on your configuration.

### Step 5: ROC & PR Metrics
Generated by the `calculate_roc_metrics()` step:

- `roc_metrics_df.csv` - Per-cell-line ROC/PR/F1/Balanced Accuracy metrics
  - Columns: cell_line, threshold, roc_auc, pr_auc, f1_score, balanced_accuracy, roc_auc_ci_low, roc_auc_ci_high, pr_auc_ci_low, pr_auc_ci_high, n_positive, n_negative, pred_min
- `roc_pr_curves.json` - Complete curve data including:
  - ROC curves (FPR, TPR points per cell line)
  - PR curves (recall, precision points per cell line)
  - Threshold sweep results (if enabled)
  - Metadata (thresholds, sample counts, confidence intervals)

### Step 6: Visualizations

**Performance Plots:**
- `aggregate_ring.html` - Overall performance ring plot
- `aggregate_ring.svg` - Static version
- `{analysis_mode}_rings_grid.html` - Grid of performance rings per item
- `{analysis_mode}_rings_grid.svg` - Static version

**Classification Plots:**
- `{analysis_mode}_classification_bars.html` - Classification metrics bar chart
- `{analysis_mode}_classification_bars.svg` - Static version
- `{analysis_mode}_confusion_heatmap.html` - Confusion matrix heatmap
- `{analysis_mode}_confusion_heatmap.svg` - Static version

**ROC/PR Curves:**
- `ROC_curve_{tissue}.html` - Interactive ROC curve with hover details
- `ROC_curve_{tissue}.svg` - Publication-quality static ROC curve
- `PR_curve_{tissue}.html` - Interactive PR curve with hover details
- `PR_curve_{tissue}.svg` - Publication-quality static PR curve
- `threshold_sweep_roc_auc.html` - ROC-AUC vs threshold offset plot (if enabled)
- `threshold_sweep_pr_auc.html` - PR-AUC vs threshold offset plot (if enabled)
- `threshold_sweep_f1_score.html` - F1-score vs threshold offset plot (if enabled)
- `threshold_sweep_balanced_accuracy.html` - Balanced accuracy vs threshold offset plot (if enabled)

**Distribution Plots:**
- `pred_distribution_*.html/svg` - Predicted synergy score distributions
- `exp_distribution_*.html/svg` - Experimental synergy score distributions
- `profile_categories_*.html/svg` - Drug profile category analysis

### Directory Structure Example

```
synco_output/
├── PD_inhibitors_dict.json
├── Drugnames_PD_dict.json
├── PD_drugnames_dict.json
├── inhibitorgroups_dict.json
├── Drugnames_inhibitor_dict.json
├── PD_targets_dict.json
├── synergy_predictions.csv
├── experimental_full_df.csv
├── experimental_drug_names_synergies_df.csv
├── experimental_inhibitor_group_synergies_df.csv
├── predictions_full_df.csv
├── predictions_drug_names_synergies_df.csv
├── predictions_inhibitor_group_synergies_df.csv
├── cell_line_comparison_results.csv
├── cell_line_pair_details.csv
├── cell_line_comparison_summary.json
├── cell_line_comparison_summary.txt
├── roc_metrics_df.csv
├── roc_pr_curves.json
└── plots/
    ├── aggregate_ring.html
    ├── aggregate_ring.svg
    ├── cell_line_rings_grid.html
    ├── cell_line_rings_grid.svg
    ├── cell_line_classification_bars.html
    ├── cell_line_confusion_heatmap.html
    ├── ROC_curve_tissue.html
    ├── ROC_curve_tissue.svg
    ├── PR_curve_tissue.html
    ├── PR_curve_tissue.svg
    ├── threshold_sweep_roc_auc.html
    ├── threshold_sweep_pr_auc.html
    └── ... (additional distribution and profile plots)
```
