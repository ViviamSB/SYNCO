import pandas
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    f1_score,
    balanced_accuracy_score,
)
from ..utils import save_file
import plotly.graph_objects as go
import re
import json


#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN FEATURE: calculate_roc_metrics
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////
def _melt_cell_lines(
        df: pandas.DataFrame,
        cell_line_list: list
) -> pandas.DataFrame:
    # Melt cell lines into one column for the df_exp
    id_cols = ['Perturbation']
    value_cols = cell_line_list
    df = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='cell_line',
        value_name='synergy'
    )

    return df

#/////////////////////////////////////////////////////
def _collect_true_scores(
        df_experiment: pandas.DataFrame, # full df or drug_names_synergies_df
        df_predictions: pandas.DataFrame, # full df or drug_names_synergies_df
        cell_line_list: Optional[List[str]] = None
) -> tuple[dict, dict, list]:
    """
    Collect true scores and predicted scores for ROC calculation.

    This function normalizes cell-line names for matching (strip + upper)
    and iterates the provided `cell_line_list` when given so that missing
    or unmatched cell lines can be reported and later included as NaN rows
    in the final dataframe.
    """
    y_true_dict = {}
    y_score_dict = {}
    skipped_cell_lines = []

    # Ensure columns exist and create normalized match column
    df_exp = df_experiment.copy()
    df_pred = df_predictions.copy()

    if 'cell_line' not in df_exp.columns:
        df_exp['cell_line'] = pandas.NA
    if 'cell_line' not in df_pred.columns:
        df_pred['cell_line'] = pandas.NA

    # Normalization helper: uppercase and remove non-alphanumeric characters
    def _normalize_name(val):
        if val is None or pandas.isna(val):
            return pandas.NA
        s = str(val).strip().upper()
        # Treat common placeholders as missing
        if s in {'', '-', 'NA', 'N/A', 'NONE'}:
            return pandas.NA
        # Remove any non-alphanumeric characters (e.g., '-', '_', spaces)
        s = re.sub(r'[^A-Z0-9]', '', s)
        return s if s != '' else pandas.NA

    df_exp['cell_line_norm'] = df_exp['cell_line'].apply(_normalize_name)
    df_pred['cell_line_norm'] = df_pred['cell_line'].apply(_normalize_name)

    # Build the list of normalized cell lines to iterate
    if cell_line_list is not None:
        norm_cell_lines = [_normalize_name(x) for x in cell_line_list]
    else:
        norm_cell_lines = list(df_exp['cell_line_norm'].unique())

    for norm_cell in norm_cell_lines:
        # Map back to original label if provided in cell_line_list, otherwise use normalized
        original_label = None
        if cell_line_list is not None:
            for cl in cell_line_list:
                if str(cl).strip().upper() == norm_cell:
                    original_label = cl
                    break
        if original_label is None:
            original_label = norm_cell

        # Filter for the current normalized cell line
        df_exp_cl = df_exp[df_exp['cell_line_norm'] == norm_cell]
        df_pred_cl = df_pred[df_pred['cell_line_norm'] == norm_cell]

        y_true_list = []
        y_score_list = []

        # If there are no experimental rows for this cell line, record and continue
        if df_exp_cl.empty:
            skipped_cell_lines.append((original_label, 'no_experimental'))
            continue

        # Loop through each perturbation in the experimental df
        for perturbation in df_exp_cl['Perturbation'].unique():
            exp_rows = df_exp_cl[df_exp_cl['Perturbation'] == perturbation]
            true_values = exp_rows['synergy'].tolist()

            # Prediction value (single value per perturbation)
            pred_match = df_pred_cl[df_pred_cl['Perturbation'] == perturbation]

            if pred_match.empty:
                # No prediction for this perturbation -> note and skip
                continue
            pred_value = pred_match['synergy'].values[0]

            # Repeat the prediction value for each experimental entry
            for true_val in true_values:
                y_true_list.append(true_val)
                y_score_list.append(pred_value)

        # Save results for this cell line if we collected matching pairs
        if y_true_list and y_score_list:
            y_true_dict[original_label] = np.array(y_true_list)
            y_score_dict[original_label] = np.array(y_score_list)
        else:
            # If we had experimental rows but no matching predictions for any perturbation
            skipped_cell_lines.append((original_label, 'no_predictions'))

    return y_true_dict, y_score_dict, skipped_cell_lines

#/////////////////////////////////////////////////////
def _calculate_roc_metrics(
        y_true,
        y_score,
        cell_line: str,
        threshold: float = 0.01,
        compute_traces: bool = True,
        n_bootstrap: Optional[int] = None,
        ci_level: float = 0.95,
        rng: Optional[np.random.Generator] = None,
):
    """
    Calculate ROC, PR, F1-score, MCC, balanced accuracy, and optional bootstrap CIs.
    """
    # Apply threshold to y_true (binary synergies: 1 for synergy, 0 for no synergy)
    # Exp > threshold -> synergy (1)
    y_true_binary = (y_true > threshold).astype(int)
    
    # For predictions: Pred < threshold -> synergy (1) 
    # So we need to flip the predictions for binary classification
    y_score_binary = (y_score < threshold).astype(int)
    
    # For ROC/PR curves, we need continuous scores where higher = more likely to be synergy
    # Since lower prediction values indicate synergy, we flip the sign
    y_score_continuous = -y_score
    
    # Check if we have both classes (0 and 1)
    if len(np.unique(y_true_binary)) <= 1:
        print(f"Warning: {cell_line} has only one class (all {np.unique(y_true_binary)[0]})")
        return None, None, None
        
    try:
        # Core metrics
        roc_auc = roc_auc_score(y_true_binary, y_score_continuous)
        pr_auc = average_precision_score(y_true_binary, y_score_continuous)
        f1 = f1_score(y_true_binary, y_score_binary)

        try:
            bal_acc = balanced_accuracy_score(y_true_binary, y_score_binary)
        except ValueError:
            bal_acc = np.nan

        fpr = tpr = roc_thresholds = precision = recall = None
        if compute_traces:
            # ROC curve for plotting
            fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_score_continuous)
            # PR curve for plotting  
            precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_score_continuous)

        # Optional bootstrap CIs for AUCs
        def _bootstrap_ci(metric_fn):
            if n_bootstrap is None or n_bootstrap <= 0:
                return None, None
            auc_values = []
            n_samples = len(y_true_binary)
            generator = rng if rng is not None else np.random.default_rng()
            for _ in range(n_bootstrap):
                indices = generator.integers(0, n_samples, n_samples)
                y_b = y_true_binary[indices]
                s_b = y_score_continuous[indices]
                # Skip if only one class in bootstrap
                if len(np.unique(y_b)) < 2:
                    continue
                try:
                    auc_values.append(metric_fn(y_b, s_b))
                except ValueError:
                    continue
            if not auc_values:
                return None, None
            lower_q = 50 * (1 - ci_level)
            upper_q = 100 - lower_q
            return float(np.percentile(auc_values, lower_q)), float(np.percentile(auc_values, upper_q))

        roc_auc_ci_low, roc_auc_ci_high = _bootstrap_ci(roc_auc_score)
        pr_auc_ci_low, pr_auc_ci_high = _bootstrap_ci(average_precision_score)

        # Store results
        roc_results = {
            'cell_line': cell_line,
            'threshold': threshold,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1,
            'balanced_accuracy': bal_acc,
            'roc_auc_ci_low': roc_auc_ci_low,
            'roc_auc_ci_high': roc_auc_ci_high,
            'pr_auc_ci_low': pr_auc_ci_low,
            'pr_auc_ci_high': pr_auc_ci_high,
            'n_positive': int(np.sum(y_true_binary)),
            'n_negative': int(len(y_true_binary) - np.sum(y_true_binary)),
            'pred_min': float(y_score.min()),
        }
        if compute_traces and fpr is not None and tpr is not None:
            roc_results['fpr'] = fpr.tolist()
            roc_results['tpr'] = tpr.tolist()
            roc_results['roc_thresholds'] = roc_thresholds.tolist()
            roc_trace = (
                roc_auc,
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f'{cell_line} (AUC={roc_auc:.3f})',
                    mode='lines'
            ))
            pr_trace = (
                pr_auc,
                go.Scatter(
                    x=recall,
                    y=precision,
                    name=f'{cell_line} (PR AUC={pr_auc:.3f})',
                    mode='lines'
            ))
        else:
            roc_trace = None
            pr_trace = None
        
        return roc_results, roc_trace, pr_trace
        
    except ValueError:
        # Silently handle ROC calculation errors (e.g., NaN values, insufficient data)
        return None, None, None

    return None, None, None

#/////////////////////////////////////////////////////
def _make_metrics_df(
        all_cell_lines: list,
        successful_results: dict
):
    """
    Extract metrics and create dataframe with NaN for failed calculations.
    """
    data = []
    for cell_line in all_cell_lines:
        if cell_line in successful_results:
            result = successful_results[cell_line]
            data.append({
                'cell_line': cell_line,
                'threshold': result.get('threshold', np.nan),
                'roc_auc': result['roc_auc'],
                'pr_auc': result['pr_auc'],
                'f1_score': result['f1_score'],
                'balanced_accuracy': result.get('balanced_accuracy', np.nan),
                'roc_auc_ci_low': result.get('roc_auc_ci_low', np.nan),
                'roc_auc_ci_high': result.get('roc_auc_ci_high', np.nan),
                'pr_auc_ci_low': result.get('pr_auc_ci_low', np.nan),
                'pr_auc_ci_high': result.get('pr_auc_ci_high', np.nan),
                'n_positive': result.get('n_positive', np.nan),
                'n_negative': result.get('n_negative', np.nan),
                'pred_min': result.get('pred_min', np.nan),
            })
        else:
            # Add NaN values for failed calculations
            data.append({
                'cell_line': cell_line,
                'threshold': np.nan,
                'roc_auc': np.nan,
                'pr_auc': np.nan,
                'f1_score': np.nan,
                'balanced_accuracy': np.nan,
                'roc_auc_ci_low': np.nan,
                'roc_auc_ci_high': np.nan,
                'pr_auc_ci_low': np.nan,
                'pr_auc_ci_high': np.nan,
                'n_positive': np.nan,
                'n_negative': np.nan,
                'pred_min': np.nan,
            })
    
    df = pandas.DataFrame(data)
    return df

#/////////////////////////////////////////////////////
def _collect_roc_metrics(
    y_true_dict,
    y_score_dict,
    all_cell_lines: Optional[List[str]] = None,
    threshold: float = 0.01,
    threshold_offsets: Optional[List[float]] = None,
    n_bootstrap: Optional[int] = None,
    ci_level: float = 0.95,
    verbose: bool = False
) -> Tuple[List[go.Scatter], List[go.Scatter], List[float], List[float], pandas.DataFrame, Dict]:
    """
    Collect ROC metrics for all cell lines.
    """
    traces_roc = []
    traces_pr = []
    rocauc_score_list = []
    prauc_score_list = []
    successful_results = {}  # Only successful calculations
    sweep_results = {}       # Per-cell-line threshold sweep summaries
    # Use provided list of all cell lines (from config) if available, otherwise
    # fall back to the cell lines that were actually present in the experimental data
    if all_cell_lines is None:
        all_cell_lines = list(y_true_dict.keys())
    else:
        # ensure it's a list copy
        all_cell_lines = list(all_cell_lines)

    # Default small sweep around the provided threshold (multiplicative offsets)
    if threshold_offsets is None:
        threshold_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]

    def _run_threshold_sweep(cell_line_name, y_true_vals, y_score_vals):
        sweep = []
        base = threshold
        delta = abs(base) if base != 0 else 1e-6
        for offset in threshold_offsets:
            thr_candidate = base + offset * delta
            metrics = _calculate_roc_metrics(
                y_true_vals,
                y_score_vals,
                cell_line=cell_line_name,
                threshold=thr_candidate,
                compute_traces=False,
                n_bootstrap=None,
                ci_level=ci_level,
            )
            if metrics and metrics[0] is not None:
                sweep.append({
                    'cell_line': cell_line_name,
                    'threshold': thr_candidate,
                    'offset': offset,
                    'roc_auc': metrics[0]['roc_auc'],
                    'pr_auc': metrics[0]['pr_auc'],
                    'f1_score': metrics[0]['f1_score'],
                    'balanced_accuracy': metrics[0].get('balanced_accuracy'),
                    'n_positive': metrics[0].get('n_positive'),
                    'n_negative': metrics[0].get('n_negative'),
                })
        return sweep

    for cell_line, y_true in y_true_dict.items():
        y_score = y_score_dict.get(cell_line, None)
        
        if y_score is None:
            if verbose:
                print(f"No prediction data available for {cell_line}")
            continue
        
        # Calculate ROC/PR metrics
        metrics = _calculate_roc_metrics(
            y_true,
            y_score,
            cell_line,
            threshold=threshold,
            compute_traces=True,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
        )
        
        # Check if all three values are returned and not None
        if metrics and all(item is not None for item in metrics):
            roc_results, roc_trace, pr_trace = metrics
            traces_roc.append(roc_trace)
            traces_pr.append(pr_trace)
            rocauc_score_list.append(roc_results['roc_auc'])
            prauc_score_list.append(roc_results['pr_auc'])
            # Store successful results for DataFrame creation
            successful_results[cell_line] = roc_results
            # Threshold sweep summary (optional)
            sweep = _run_threshold_sweep(cell_line, y_true, y_score)
            if sweep:
                sweep_results[cell_line] = sweep
        else:
            if verbose:
                print(f"ROC Metrics could not be calculated for {cell_line}")
    
    # Create metrics DataFrame including failed calculations with NaN
    metrics_df = _make_metrics_df(all_cell_lines, successful_results)

    # Attach sweeps inside successful_results for downstream JSON export
    for cl, sweeps in sweep_results.items():
        if cl in successful_results:
            successful_results[cl]['threshold_sweep'] = sweeps

    return traces_roc, traces_pr, rocauc_score_list, prauc_score_list, metrics_df, successful_results

#/////////////////////////////////////////////////////
def _save_curves_json(
        roc_metrics_results: Tuple,
        output_path: Path,
        verbose: bool = False
) -> None:
    """
    Save ROC/PR curve data as JSON for later plotting.
    
    Args:
        roc_metrics_results: Tuple containing (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df, successful_results)
        output_path: Directory path to save the JSON file
        verbose: Whether to print save confirmation
    """
    traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df, successful_results = roc_metrics_results
    
    # Build JSON structure with all curve data
    curves_data = {
        'roc_curves': [],
        'pr_curves': [],
        'rocauc_scores': rocauc_scores,
        'prauc_scores': prauc_scores,
        'threshold_sweeps': []
    }
    
    # Extract ROC curve data from successful_results (which has fpr, tpr, etc.)
    for cell_line, result in successful_results.items():
        roc_curve_data = {
            'cell_line': cell_line,
            'auc': float(result['roc_auc']),
            'threshold': result.get('threshold'),
            'n_positive': result.get('n_positive'),
            'n_negative': result.get('n_negative'),
            'balanced_accuracy': result.get('balanced_accuracy'),
            'roc_auc_ci_low': result.get('roc_auc_ci_low'),
            'roc_auc_ci_high': result.get('roc_auc_ci_high'),
            'pr_auc_ci_low': result.get('pr_auc_ci_low'),
            'pr_auc_ci_high': result.get('pr_auc_ci_high'),
            'fpr': result.get('fpr', []),  # already converted to list
            'tpr': result.get('tpr', []),  # already converted to list
        }
        curves_data['roc_curves'].append(roc_curve_data)
        # Add sweep results if present
        if 'threshold_sweep' in result:
            curves_data['threshold_sweeps'].append({
                'cell_line': cell_line,
                'sweep': result['threshold_sweep']
            })
    
    # Extract PR curve data from traces (since successful_results doesn't have precision/recall)
    for auc, trace in traces_pr:
        pr_curve_data = {
            'cell_line': trace.name.split(' (')[0],  # Extract cell line name from trace name
            'auc': float(auc),
            'recall': trace.x.tolist() if hasattr(trace.x, 'tolist') else list(trace.x),
            'precision': trace.y.tolist() if hasattr(trace.y, 'tolist') else list(trace.y),
        }
        curves_data['pr_curves'].append(pr_curve_data)
    
    # Save to JSON
    json_path = output_path / 'roc_pr_curves.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(curves_data, f, indent=2)
    
    if verbose:
        print(f"ROC/PR curves saved to {json_path}")

#/////////////////////////////////////////////////////
# Removed JSON loader; curve loading now lives in plotting.load_results

#/////////////////////////////////////////////////////
def calculate_roc_metrics(
        df_experiment: pandas.DataFrame,
        df_predictions: pandas.DataFrame,
        threshold: float,
        cell_line_list: list,
    threshold_offsets: Optional[List[float]] = None,
    n_bootstrap: Optional[int] = None,
    ci_level: float = 0.95,
    verbose: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[Tuple, List[str]]:
    """
    Main function to calculate ROC metrics.
    """
    # Clean cell_line_list to match the cleaned column names
    cell_line_list_cleaned = [name.upper().replace('-', '') for name in cell_line_list]
    
    # Clean only the cell line column names in DataFrames (not metadata columns like 'Perturbation')
    # Build a selective mapping that only renames columns matching cell_line_list
    def clean_cell_columns(df, cell_lines_to_clean):
        column_mapping = {}
        for col in df.columns:
            # Only rename columns that match the original cell line names
            for original_name in cell_line_list:
                if col == original_name:
                    cleaned = original_name.upper().replace('-', '')
                    column_mapping[col] = cleaned
                    break
        return df.rename(columns=column_mapping)
    
    df_predictions = clean_cell_columns(df_predictions, cell_line_list)
    df_experiment = clean_cell_columns(df_experiment, cell_line_list)
    
    # Only melt the predictions DataFrame (experimental is already in correct format)
    df_pred = _melt_cell_lines(df_predictions, cell_line_list_cleaned)

    # Collect true scores (use cleaned cell_line_list for matching)
    y_true_dict, y_score_dict, skipped_cell_lines = _collect_true_scores(
        df_experiment,
        df_pred,
        cell_line_list=cell_line_list_cleaned
    )

    # Collect roc metrics; pass the cleaned cell_line_list so the final DataFrame
    # contains all requested cell lines (with NaNs where calculation failed)
    roc_metrics_results = _collect_roc_metrics(
        y_true_dict,
        y_score_dict,
        all_cell_lines=cell_line_list_cleaned,
        threshold=threshold,
        threshold_offsets=threshold_offsets,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        verbose=verbose
    )
    if verbose:
        print(f"Skipped cell lines: {skipped_cell_lines}")
    
    # Save files if output path is specified
    if output_path is not None:
        # Save metrics DataFrame as CSV
        save_file(roc_metrics_results[4], output_path / 'roc_metrics_df.csv', file_type='csv')
        # Save ROC/PR curves as JSON for later plotting
        _save_curves_json(roc_metrics_results, output_path, verbose=verbose)
    
    # Return only the first 5 elements (without successful_results dict) for backward compatibility
    return roc_metrics_results[:5], skipped_cell_lines
