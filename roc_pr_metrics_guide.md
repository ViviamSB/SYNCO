# ROC-AUC and PR-AUC Curve Calculation in SYNCO

## Overview

SYNCO calculates Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves to evaluate synergy predictions across cell lines. This guide explains the complete workflow: from input data preparation through metric calculation to visualization.

---

## Table of Contents

1. [High-Level Flow](#high-level-flow)
2. [Input Data Structure](#input-data-structure)
3. [Data Preparation](#data-preparation)
4. [Metric Calculation](#metric-calculation)
5. [Curve Generation](#curve-generation)
6. [Threshold Configuration](#threshold-configuration)
7. [Output Files](#output-files)
8. [Key Concepts](#key-concepts)

---

## High-Level Flow

```
calculate_roc_metrics()
  ↓
_collect_true_scores()  → Matches experimental vs predicted synergies
  ↓
_collect_roc_metrics()  → Calculates metrics for each cell line
  ↓
_calculate_roc_metrics()  → Computes ROC/PR curves + traces
  ↓
_save_curves_json()  → Saves data for plotting
  ↓
make_roc_plots()  → Displays final curves
```

---

## Input Data Structure

### Configuration Parameters

The ROC/PR calculation is configured via the `compare` section of `synco_example_config.json`:

```json
{
  "compare": {
    "prediction_method": "DrugLogics",
    "threshold": 0.00001,
    "synergy_column": "synergy",
    "analysis_mode": "cell_line",
    "duplicate_strategy": "mean"
  }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prediction_method` | str | Model used for predictions (e.g., "DrugLogics") |
| `threshold` | float | Binary classification threshold (see [Threshold Configuration](#threshold-configuration)) |
| `synergy_column` | str | Column name containing synergy scores |
| `analysis_mode` | str | Grouping level ("cell_line" or "tissue") |
| `duplicate_strategy` | str | How to aggregate duplicates ("mean", "median", etc.) |

### Input DataFrames

**`df_experiment`** (Experimental synergies):
- **Structure**: One row per drug combination, columns per cell line
- **Columns**: `['Perturbation', 'cell_line_1', 'cell_line_2', ...]`
- **Values**: Continuous synergy scores (e.g., 0.5, 0.8, -0.3)
- **Example**:
  ```
  Perturbation    C2BBE1    CAR1    T84
  Drug_A+Drug_B   0.7       0.5     -0.2
  Drug_C+Drug_D   0.3       0.8     0.6
  ```

**`df_predictions`** (Predicted synergies):
- Same structure as experimental data
- Values: Predicted synergy scores from the model

**`cell_line_list`** (from config):
- List of cell lines to analyze
- Example: `['C2BBE1', 'CAR1', 'T84']`

---

## Data Preparation

### Step 1: Reshape Predictions

```python
df_pred = _melt_cell_lines(df_predictions, cell_line_list)
```

The prediction DataFrame is melted from wide format (columns per cell line) to long format:

**Before (wide format)**:
```
Perturbation    C2BBE1    CAR1    T84
Drug_A+Drug_B   0.3       0.6     -0.1
```

**After (long format)**:
```
Perturbation    cell_line    synergy
Drug_A+Drug_B   C2BBE1       0.3
Drug_A+Drug_B   CAR1         0.6
Drug_A+Drug_B   T84          -0.1
```

### Step 2: Normalize Cell Line Names

Cell line names are normalized for robust matching:

```python
def _normalize_name(val):
    # Convert to uppercase and strip whitespace
    s = str(val).strip().upper()
    
    # Treat common placeholders as missing
    if s in {'', '-', 'NA', 'N/A', 'NONE'}:
        return pandas.NA
    
    # Remove non-alphanumeric characters
    s = re.sub(r'[^A-Z0-9]', '', s)
    
    return s if s != '' else pandas.NA
```

**Examples**:
- `"c2bbe1"` → `"C2BBE1"`
- `"CAR-1"` → `"CAR1"`
- `"T84 "` → `"T84"`
- `"NA"` → `pandas.NA`

### Step 3: Match Experimental and Predicted Scores

For each cell line and perturbation, experimental and predicted scores are paired:

```python
y_true_list = []  # Experimental synergy values
y_score_list = []  # Predicted synergy values

for perturbation in df_exp_cl['Perturbation'].unique():
    exp_rows = df_exp_cl[df_exp_cl['Perturbation'] == perturbation]
    true_values = exp_rows['synergy'].tolist()
    
    pred_match = df_pred_cl[df_pred_cl['Perturbation'] == perturbation]
    pred_value = pred_match['synergy'].values[0]
    
    # Repeat prediction for each experimental entry
    for true_val in true_values:
        y_true_list.append(true_val)
        y_score_list.append(pred_value)
```

### Output: Score Dictionaries

```python
y_true_dict = {
    'C2BBE1': np.array([0.7, 0.5, -0.2, 0.3, ...]),
    'CAR1': np.array([0.5, 0.8, 0.6, ...]),
    'T84': np.array([...])
}

y_score_dict = {
    'C2BBE1': np.array([0.3, 0.6, -0.1, 0.2, ...]),
    'CAR1': np.array([0.4, 0.5, 0.7, ...]),
    'T84': np.array([...])
}
```

---

## Metric Calculation

### Binary Classification

The continuous synergy scores are converted to binary classifications:

```python
threshold = 0.00001  # From config

# Experimental: synergy if score > threshold
y_true_binary = (y_true > threshold).astype(int)

# Predictions: synergy if score < threshold (inverted logic)
y_score_binary = (y_score < threshold).astype(int)

# For ROC/PR curves: flip sign
y_score_continuous = -y_score
```

### Why Invert Predictions?

In SYNCO's model, **lower scores indicate stronger synergy**. However, sklearn's ROC/PR curve functions expect **higher scores to indicate the positive class**. We flip the sign so the curve calculation works correctly:

$$y_{\text{score\_continuous}} = -y_{\text{score}}$$

### Calculated Metrics

For each cell line, core metrics are computed (plus optional CIs):

#### 1. ROC-AUC (Receiver Operating Characteristic Area Under Curve)

```python
roc_auc = roc_auc_score(y_true_binary, y_score_continuous)
```

- **Range**: [0, 1]
- **Interpretation**: 
  - 0.5 = random classifier
  - 1.0 = perfect classifier
  - >0.7 = good performance
- **Formula**: Area under the ROC curve across all classification thresholds

#### 2. PR-AUC (Precision-Recall Area Under Curve)

```python
pr_auc = average_precision_score(y_true_binary, y_score_continuous)
```

- **Range**: [0, 1]
- **Interpretation**: Similar to ROC-AUC but better for imbalanced datasets
- **Formula**: Area under precision-recall curve

#### 3. F1-Score

```python
f1 = f1_score(y_true_binary, y_score_binary)
```

- **Range**: [0, 1]
- **Interpretation**: Harmonic mean of precision and recall
- **Formula**: $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Note**: Computed at the specific threshold value (not across all thresholds)

#### 4. Balanced Accuracy

```python
bal_acc = balanced_accuracy_score(y_true_binary, y_score_binary)
```

- **Range**: [0, 1]; average of TPR and TNR
- **Use**: Simple imbalance-aware alternative to accuracy

#### 5. Bootstrap Confidence Intervals (optional)

```python
roc_auc_ci = bootstrap(roc_auc_score, n_bootstrap, ci_level)
pr_auc_ci = bootstrap(average_precision_score, n_bootstrap, ci_level)
```

- **Purpose**: Quantify uncertainty of ROC-AUC/PR-AUC
- **Behavior**: Skips resamples with a single class; returns lower/upper quantiles

#### 6. Sample Counts

```python
n_positive = int(np.sum(y_true_binary))      # Synergistic pairs
n_negative = int(len(y_true_binary) - np.sum(y_true_binary))  # Non-synergistic
```

### ROC and PR Curve Points

The complete curves are computed using multiple threshold values:

```python
# ROC curve: False Positive Rate vs True Positive Rate
fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_score_continuous)

# PR curve: Recall vs Precision
precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_score_continuous)
```

**ROC Curve Definitions**:
- $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (True Positive Rate / Sensitivity)
- $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ (False Positive Rate)

**PR Curve Definitions**:
- $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ (Positive Predictive Value)
- $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (True Positive Rate)

---

## Curve Generation

### Plotly Trace Creation

For each cell line, two Plotly traces are created:

#### ROC Trace
```python
roc_trace = (
    roc_auc,  # AUC value (used for sorting)
    go.Scatter(
        x=fpr,                              # False Positive Rate (x-axis)
        y=tpr,                              # True Positive Rate (y-axis)
        name=f'C2BBE1 (AUC={roc_auc:.3f})',  # Legend label
        mode='lines'
    )
)
```

#### PR Trace
```python
pr_trace = (
    pr_auc,  # AUC value (used for sorting)
    go.Scatter(
        x=recall,                           # Recall (x-axis)
        y=precision,                        # Precision (y-axis)
        name=f'C2BBE1 (PR AUC={pr_auc:.3f})', # Legend label
        mode='lines'
    )
)
```

### Curve Interpretation

**ROC Curve**: Shows the trade-off between:
- Catching synergies (True Positive Rate on y-axis)
- False alarms (False Positive Rate on x-axis)

**PR Curve**: Shows the trade-off between:
- Correctly identified synergies (Precision on y-axis)
- Coverage of all synergies (Recall on x-axis)

**Visual Comparison**:
- A curve close to the top-left corner (high TPR, low FPR) indicates good performance
- A curve above the diagonal line indicates better-than-random performance
- Multiple overlaid curves allow comparison across cell lines

---

## Threshold Configuration

### Purpose

The `threshold` parameter controls when a synergy score is considered "synergistic":

```json
"threshold": 0.00001
```

### Two Uses of Threshold

#### 1. Binary Classification (Evaluation)
```python
y_true_binary = (y_true > threshold).astype(int)
y_score_binary = (y_score < threshold).astype(int)
```

Scores are converted to 0/1 for metric calculation.

#### 2. Plotting Reference (Optional)
In visualization functions, a horizontal/vertical line is drawn at the threshold value for reference.

#### 3. Micro-Sweep (Offsets)
SYNCO runs a small sensitivity sweep around the provided threshold using multiplicative offsets (default [-2, -1, 0, 1, 2] × |threshold| added to the base threshold). This shows how ROC/PR/F1/balanced accuracy move when the decision boundary shifts near zero without changing the primary threshold.

### Typical Values

| Threshold | Interpretation |
|-----------|-----------------|
| Very small (< 0.0001) | Lenient: only strongest synergies labeled as positive |
| Small (0.1 - 0.5) | Moderate: balanced positive/negative split |
| Larger (> 0.5) | Strict: even weak synergies labeled positive |

### Configuration Example

```json
{
  "compare": {
    "threshold": 0.00001,  // Very lenient: only very strong synergies
  }
}
```

---

## Output Files

### Primary Output: Metrics DataFrame

**File**: `roc_metrics_df.csv`

```
cell_line,threshold,roc_auc,pr_auc,f1_score,balanced_accuracy,roc_auc_ci_low,roc_auc_ci_high,pr_auc_ci_low,pr_auc_ci_high,n_positive,n_negative,pred_min
C2BBE1,0.00001,0.823,0.756,0.684,0.71,0.781,0.861,0.702,0.791,45,105,0.00012
CAR1,0.00001,0.791,0.718,0.621,0.68,0.742,0.829,0.664,0.761,38,112,0.00008
T84,0.00001,0.856,0.802,0.729,0.75,0.821,0.892,0.755,0.842,52,98,0.00015
```

**Columns**:
- `cell_line`: Cell line identifier
- `threshold`: Threshold used for binarization
- `roc_auc`: Area Under ROC Curve [0, 1]
- `pr_auc`: Area Under PR Curve [0, 1]
- `f1_score`: F1-Score at threshold [0, 1]
- `balanced_accuracy`: Average of TPR and TNR
- `roc_auc_ci_low` / `roc_auc_ci_high`: Bootstrap CI bounds for ROC-AUC (if enabled)
- `pr_auc_ci_low` / `pr_auc_ci_high`: Bootstrap CI bounds for PR-AUC (if enabled)
- `n_positive`: Count of synergistic pairs
- `n_negative`: Count of non-synergistic pairs
- `pred_min`: Minimum predicted score for the cell line

### Secondary Output: Curve Data (JSON)

**File**: `roc_pr_curves.json`

```json
{
  "roc_curves": [
    {
      "cell_line": "C2BBE1",
      "auc": 0.823,
      "fpr": [0.0, 0.0095, 0.019, ..., 1.0],
      "tpr": [0.0, 0.044, 0.089, ..., 1.0]
    },
    {
      "cell_line": "CAR1",
      "auc": 0.791,
      "fpr": [0.0, 0.0089, 0.018, ..., 1.0],
      "tpr": [0.0, 0.053, 0.105, ..., 1.0]
    }
  ],
  "pr_curves": [
    {
      "cell_line": "C2BBE1",
      "auc": 0.756,
      "recall": [1.0, 0.978, 0.956, ..., 0.0],
      "precision": [0.756, 0.760, 0.765, ..., 1.0]
    },
    {
      "cell_line": "CAR1",
      "auc": 0.718,
      "recall": [1.0, 0.974, 0.947, ..., 0.0],
      "precision": [0.718, 0.722, 0.727, ..., 1.0]
    }
  ],
  "rocauc_scores": [0.823, 0.791, 0.856],
  "prauc_scores": [0.756, 0.718, 0.802],
  "threshold_sweeps": [
    {
      "cell_line": "C2BBE1",
      "sweep": [
        {"threshold": -0.00001, "offset": -2.0, "roc_auc": 0.80, "pr_auc": 0.74, "f1_score": 0.65, "balanced_accuracy": 0.69},
        {"threshold": 0.00001, "offset": 0.0, "roc_auc": 0.82, "pr_auc": 0.76, "f1_score": 0.68, "balanced_accuracy": 0.71},
        {"threshold": 0.00003, "offset": 2.0, "roc_auc": 0.81, "pr_auc": 0.75, "f1_score": 0.66, "balanced_accuracy": 0.70}
      ]
    }
  ]
}
```

### Visualization Output

**Files**:
- `ROC_curve_{tissue}.html` - Interactive ROC plot
- `ROC_curve_{tissue}.svg` - Static ROC plot
- `PR_curve_{tissue}.html` - Interactive PR plot
- `PR_curve_{tissue}.svg` - Static PR plot

---

## Key Concepts

### AUC Interpretation

| AUC Value | Interpretation |
|-----------|-----------------|
| 0.90–1.00 | Excellent discrimination |
| 0.80–0.90 | Good discrimination |
| 0.70–0.80 | Fair discrimination |
| 0.60–0.70 | Poor discrimination |
| 0.50–0.60 | Very poor discrimination |
| 0.50      | No discrimination (random) |

### Choosing ROC vs PR

- **ROC-AUC**: Better for balanced datasets, shows overall performance across thresholds
- **PR-AUC**: Better for imbalanced datasets (many more negatives than positives), focuses on positive class performance

SYNCO includes both metrics for comprehensive evaluation.

### Handling Missing Data

Cell lines without experimental or predicted data are:
1. Recorded in `skipped_cell_lines` list
2. Included in output DataFrame with `NaN` values
3. Logged if `verbose=True`

---

## References

- [ROC Curve - Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Precision and Recall - Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Plotly Graph Objects](https://plotly.com/python/graph-objects/)

---

**Last Updated**: January 24, 2026  
**Module**: `synco.features.roc_metrics` & `synco.plotting.roc_plots`