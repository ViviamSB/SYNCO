import pandas
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, f1_score
from ..utils import save_file
import plotly.graph_objects as go


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
) -> tuple[dict, dict, list]:
    """
    Collect true scores and predicted scores for ROC calculation.
    """
    y_true_dict = {}
    y_score_dict = {}
    skipped_cell_lines = []

    for cell_line in df_experiment['cell_line'].unique():
        # Filter for the current cell line
        df_exp = df_experiment[df_experiment['cell_line'] == cell_line]
        df_pred = df_predictions[df_predictions['cell_line'] == cell_line]

        y_true_list = []
        y_score_list = []

        # Loop through each perturbation in the experimental df
        for perturbation in df_exp['Perturbation'].unique():
            # Get the synergy value for the current perturbation
            exp_rows = df_exp[df_exp['Perturbation'] == perturbation]
            true_values = exp_rows['synergy'].tolist()

            # Prediction value (single value per perturbation)
            pred_match = df_pred[df_pred['Perturbation'] == perturbation]

            if pred_match.empty:
                skipped_cell_lines.append(cell_line)
                continue
            pred_value = pred_match['synergy'].values[0]

            # Repeat the prediction value for each experimental entry
            for true_val in true_values:
                y_true_list.append(true_val)
                y_score_list.append(pred_value)
        
        # Save results for this cell line
        if y_true_list and y_score_list:
            y_true_dict[cell_line] = np.array(y_true_list)
            y_score_dict[cell_line] = np.array(y_score_list)
        
    return y_true_dict, y_score_dict, skipped_cell_lines

#/////////////////////////////////////////////////////
def _calculate_roc_metrics(
        y_true,
        y_score,
        cell_line: str,
        threshold: float = 0.01
):
    """
    Calculate ROC, PR, F1-score 
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
        # ROC AUC using flipped continuous prediction scores
        roc_auc = roc_auc_score(y_true_binary, y_score_continuous)
        
        # PR AUC using flipped continuous prediction scores
        pr_auc = average_precision_score(y_true_binary, y_score_continuous)
        
        # F1 score using the threshold-based binary predictions
        f1 = f1_score(y_true_binary, y_score_binary)
        
        # ROC curve for plotting
        fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_score_continuous)
        
        # PR curve for plotting  
        precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_score_continuous)

        # Store results
        roc_results = {
            'cell_line': cell_line,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1,
            'n_positive': int(np.sum(y_true_binary)),
            'n_negative': int(len(y_true_binary) - np.sum(y_true_binary)),
            'pred_min': float(y_score.min()),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_thresholds': roc_thresholds.tolist()
        }
        
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
        
        return roc_results, roc_trace, pr_trace
        
    except ValueError as e:
        print(f"Error calculating metrics for {cell_line}: {e}")
        print(f"  y_true_binary unique: {np.unique(y_true_binary)}")
        print(f"  y_score range: [{y_score.min():.3f}, {y_score.max():.3f}]")
        return None, None, None

    return None, None, None

#/////////////////////////////////////////////////////
def _make_metrics_df(
        all_cell_lines: list,
        successful_results: dict
):
    """
    Extract metrics and create dataframe with NaN for failed calculations
    """
    data = []
    for cell_line in all_cell_lines:
        if cell_line in successful_results:
            result = successful_results[cell_line]
            data.append({
                'cell_line': cell_line,
                'roc_auc': result['roc_auc'],
                'pr_auc': result['pr_auc'],
                'f1_score': result['f1_score'],
                'n_positive': result.get('n_positive', np.nan),
                'n_negative': result.get('n_negative', np.nan),
                'pred_min': result.get('pred_min', np.nan),
            })
        else:
            # Add NaN values for failed calculations
            data.append({
                'cell_line': cell_line,
                'roc_auc': np.nan,
                'pr_auc': np.nan,
                'f1_score': np.nan,
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
        threshold: float = 0.01,
        verbose: bool = False
) -> Tuple[List[go.Scatter], List[go.Scatter], List[float], List[float], pandas.DataFrame]:
    """
    Collect ROC metrics for all cell lines.
    """
    traces_roc = []
    traces_pr = []
    rocauc_score_list = []
    prauc_score_list = []
    successful_results = {}  # Only successful calculations
    all_cell_lines = list(y_true_dict.keys())  # All cell lines that were attempted

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
            threshold
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
        else:
            if verbose:
                print(f"ROC Metrics could not be calculated for {cell_line}")
    
    # Create metrics DataFrame including failed calculations with NaN
    metrics_df = _make_metrics_df(all_cell_lines, successful_results)

    return traces_roc, traces_pr, rocauc_score_list, prauc_score_list, metrics_df

#/////////////////////////////////////////////////////
def calculate_roc_metrics(
        df_experiment: pandas.DataFrame,
        df_predictions: pandas.DataFrame,
        threshold: float,
        cell_line_list: list,
        verbose: bool = False,
        output_path: Optional[Union[str, Path]] = None
) -> Tuple[Tuple, List[str]]:
    """
    Main function to calculate ROC metrics.
    """
    # Only melt the predictions DataFrame (experimental is already in correct format)
    df_pred = _melt_cell_lines(df_predictions, cell_line_list)

    # Collect true scores
    y_true_dict, y_score_dict, skipped_cell_lines = _collect_true_scores(df_experiment, df_pred)

    # Collect roc metrics
    roc_metrics_results = _collect_roc_metrics(
        y_true_dict,
        y_score_dict,
        threshold,
        verbose
    )
    if verbose:
        print(f"Skipped cell lines: {skipped_cell_lines}")
    # Save files if output path is specified
    if output_path is not None:
        save_file(roc_metrics_results[4], output_path / 'roc_metrics_df.csv', file_type='csv')

    return roc_metrics_results, skipped_cell_lines
