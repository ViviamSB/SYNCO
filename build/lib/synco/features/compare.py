import pandas
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from ..utils import save_file

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN FEATURE: compare_synergies
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////
def _normalize_inhibitor_combination(combination: str, separator: str = ' + ') -> str:
    """
    Normalize inhibitor combination by sorting the components alphabetically.
    This ensures that 'A + B' and 'B + A' are treated as the same combination.
    
    Args:
        combination (str): Inhibitor combination string (e.g., 'AKT_inhibitors + mTOR_inhibitors')
        separator (str): Separator used in the combination string
    
    Returns:
        str: Normalized combination with components sorted alphabetically
    """
    if separator not in combination:
        return combination
    
    components = [comp.strip() for comp in combination.split(separator)]
    components.sort()  # Sort alphabetically
    return separator.join(components)

#/////////////////////////////////////////////////////
def _normalize_combinations_in_df(df: pandas.DataFrame, combination_column: str = 'inhibitor_combination') -> pandas.DataFrame:
    """
    Normalize all inhibitor combinations in a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing inhibitor combinations
        combination_column (str): Name of the column containing combinations
    
    Returns:
        pandas.DataFrame: DataFrame with normalized combinations
    """
    df = df.copy()
    df[combination_column] = df[combination_column].apply(_normalize_inhibitor_combination)
    return df

#/////////////////////////////////////////////////////
def _make_boolean_df(
        df: pandas.DataFrame,
        cell_line_list: list,
        synergy_column: str = 'synergy',
        threshold: float = 0.5
) -> pandas.DataFrame:
    """
    Create a boolean DataFrame indicating whether synergy values exceed a threshold.

    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        cell_line_list (list): List of cell lines to include.
        synergy_column (str): Column name for synergy values.
        threshold (float): Threshold value for determining synergy.

    Returns:
        pandas.DataFrame: Boolean DataFrame indicating synergy presence.
    """
    boolean_df = df[df['cell_line'].isin(cell_line_list)].copy()
    boolean_df[synergy_column] = boolean_df[synergy_column] > threshold
    return boolean_df

#/////////////////////////////////////////////////////
def _match_synergies(
        df_experiment: pandas.DataFrame,
        df_prediction: pandas.DataFrame,
        combi: list
) -> Tuple[Dict, Dict]:
    """
    Match synergies across different cell lines or inhibitor combinations. Compare the boolean values between the observed and predicted data
    - If both obs and pred values are True, add 1 to the Match count
    - If both obs and pred values are False, add 1 to the Match count
    - If obs and pred values are different, add 1 to the Mismatch count

    Args:
        df_experiment (pandas.DataFrame): DataFrame containing experimental synergy data.
        df_prediction (pandas.DataFrame): DataFrame containing predicted synergy data.
        combi (list): List of combinations or cell lines to include.

    Returns:
        match_counts (dict): Dictionary with counts of matching synergies.
        mismatch_counts (dict): Dictionary with counts of mismatching synergies.
    """
    match_counts = {}
    mismatch_counts = {}
    # Align the DataFrames on both index and columns
    df_exp_aligned = df_experiment.copy()
    df_pred_aligned = df_prediction.reindex(index=df_exp_aligned.index, columns=df_exp_aligned.columns)

    for item in combi:
        # Determine if combination is a column or index label
        if item in df_exp_aligned.columns:
            df_exp = df_exp_aligned[item]
            df_pred = df_pred_aligned[item]
            key_label = item
        elif item in df_exp_aligned.index:
            df_exp = df_exp_aligned.loc[item, :]
            df_pred = df_pred_aligned.loc[item, :]
            key_label = item
        else:
            raise KeyError(f"Combination '{item}' not found in columns or index.")

        match_count = (df_exp == df_pred).sum()
        mismatch_count = (df_exp != df_pred).sum()

        match_counts[key_label] = match_count
        mismatch_counts[key_label] = mismatch_count
        mismatch_count = (df_exp != df_pred).sum()

        match_counts[item] = match_count
        mismatch_counts[item] = mismatch_count

    return match_counts, mismatch_counts

#/////////////////////////////////////////////////////
def _calculate_confusion_matrix(
        df_experiment: pandas.DataFrame,
        df_prediction: pandas.DataFrame,
        combi: list,
) -> pandas.DataFrame:
    """
    Calculate the number of True Positives, True Negatives, False Positives, and False Negatives.

    Args:
        df_experiment (pandas.DataFrame): DataFrame containing experimental data.
        df_prediction (pandas.DataFrame): DataFrame containing predicted data.
        combi (list): List of combinations or cell lines to include.

    Returns:
        Dataframe: DataFrame containing the confusion matrix counts.
    """
    true_positive = {}
    true_negative = {}
    false_positive = {}
    false_negative = {}
    for item in combi:
        # Create a small DataFrame or Series for the current combination
        # Check if item is in columns or index
        if item in df_experiment.columns:
            df_exp = df_experiment.loc[:, item]
            df_pred = df_prediction.loc[:, item]
        elif item in df_experiment.index:
            df_exp = df_experiment.loc[item, :]
            df_pred = df_prediction.loc[item, :]
        else:
            raise KeyError(f"Item '{item}' not found in columns or index of dataframes.")
        # Calculate True Positives, True Negatives, False Positives, and False Negatives
        tp = ((df_exp) & (df_pred)).sum()
        tn = ((~df_exp.astype(bool)) & (~df_pred.astype(bool))).sum()
        fp = ((~df_exp.astype(bool)) & (df_pred)).sum()
        fn = ((df_exp) & (~df_pred.astype(bool))).sum()
        # Store the counts in dictionaries
        true_positive[item] = tp
        true_negative[item] = tn
        false_positive[item] = fp
        false_negative[item] = fn

    # Create a DataFrame to hold the results
    compared_df = pandas.DataFrame([true_positive, true_negative, false_positive, false_negative]).T
    compared_df.columns = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    compared_df['Total'] = compared_df.sum(axis=1)    

    return compared_df

#/////////////////////////////////////////////////////
def _calculate_accuracy_recall_precision(
        df_compared: pandas.DataFrame,        
) -> pandas.DataFrame:
    """
    Calculate from the compared DataFrame:
    - Accuracy = (correct classifications) / (total classifications)
    - Accuracy = (True Positives + True Negatives) / (Total)
    - Recall = (correct positive classifications) / (total actual positives)
    - Recall = True Positives / (True Positives + False Negatives)
    - Precision = (correct positive classifications) / (total predicted positives)
    - Precision = True Positives / (True Positives + False Positives)

    Args:
        df_compared (pandas.DataFrame): DataFrame containing comparison results.

    Returns:
        dataframe: DataFrame containing accuracy, recall, and precision metrics.
    """
    # Calculate metrics per row (per combination) instead of globally
    def calculate_row_metrics(row):
        tp = row['True Positive']
        tn = row['True Negative']
        fp = row['False Positive']
        fn = row['False Negative']
        
        total = tp + tn + fp + fn
        accuracy = ((tp + tn) / total * 100) if total > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        
        return pandas.Series({
            'Accuracy': round(accuracy, 2),
            'Recall': round(recall, 2), 
            'Precision': round(precision, 2)
        })
    
    # Apply the calculation to each row
    metrics_df = df_compared.apply(calculate_row_metrics, axis=1)
    
    # Add the metrics to the original DataFrame
    df_compared = pandas.concat([df_compared, metrics_df], axis=1)
    df_compared = df_compared.fillna(0)  # Fill NaN values with 0

    return df_compared

#/////////////////////////////////////////////////////
def _find_common_and_skipped_items(
        df_exp: pandas.DataFrame,
        df_pred: pandas.DataFrame,
        type: str
) -> Tuple[set, set, set, set]:
    """
    Find common items between datasets and identify items that will be skipped.
    
    Args:
        df_exp (pandas.DataFrame): Experimental dataset after pivot.
        df_pred (pandas.DataFrame): Predicted dataset after pivot.
        type (str): Type of comparison ('cell_line' or 'inhibitor_combination').
        
    Returns:
        Tuple[set, set, set, set]: exp_items, pred_items, common_items, skipped_exp, skipped_pred
    """
    # Extract unique combinations or cell lines for each dataset
    if type == 'cell_line':
        exp_items = set(df_exp.columns.tolist())
        pred_items = set(df_pred.columns.tolist())
    elif type == 'inhibitor_combination':
        exp_items = set(df_exp.index.tolist())
        pred_items = set(df_pred.index.tolist())
    else:
        raise ValueError("Type must be either 'cell_line' or 'inhibitor_combination'.")
    
    # Find common items and items that will be skipped
    common_items = exp_items & pred_items
    skipped_exp = exp_items - pred_items
    skipped_pred = pred_items - exp_items
    
    return exp_items, pred_items, common_items, skipped_exp, skipped_pred

#/////////////////////////////////////////////////////
def _print_comparison_summary(
        type: str,
        exp_items: set = None,
        pred_items: set = None, 
        common_items: set = None,
        skipped_exp: set = None,
        skipped_pred: set = None,
        results_df: pandas.DataFrame = None
) -> None:
    """
    Print a summary of the comparison between experimental and predicted datasets.
    
    Args:
        exp_items (set): Items in experimental dataset.
        pred_items (set): Items in predicted dataset.
        common_items (set): Items common to both datasets.
        skipped_exp (set): Items only in experimental dataset.
        skipped_pred (set): Items only in predicted dataset.
        type (str): Type of comparison ('cell_line' or 'inhibitor_combination').
        results_df (pandas.DataFrame, optional): Results dataframe with metrics for global summary.
    """
    if exp_items is not None and pred_items is not None:
        print("=== COMPARISON SUMMARY ===\n")
        print(f"Items in experimental data: {len(exp_items)}")
        print(f"Items in predicted data: {len(pred_items)}")
        print(f"Common items to compare: {len(common_items)}")
        print(f"Skipped from experimental: {len(skipped_exp)}")
        print(f"Skipped from predicted: {len(skipped_pred)}")
        
        if skipped_exp:
            print(f"Experimental-only {type}s: {sorted(list(skipped_exp))}")
        if skipped_pred:
            print(f"Predicted-only {type}s: {sorted(list(skipped_pred))}")

    # Print global results if results_df is provided
    if results_df is not None and not results_df.empty:
        print("\n=== GLOBAL RESULTS ===\n")
        
        # Calculate global totals
        global_tp = results_df['True Positive'].sum()
        global_tn = results_df['True Negative'].sum()
        global_fp = results_df['False Positive'].sum()
        global_fn = results_df['False Negative'].sum()
        global_total = results_df['Total'].sum()
        global_match = results_df['Match'].sum()
        global_mismatch = results_df['Mismatch'].sum()
        
        # Calculate global percentages
        global_match_pct = (global_match / global_total * 100) if global_total > 0 else 0
        global_mismatch_pct = (global_mismatch / global_total * 100) if global_total > 0 else 0
        
        # Calculate global metrics
        global_accuracy = ((global_tp + global_tn) / global_total * 100) if global_total > 0 else 0
        global_recall = (global_tp / (global_tp + global_fn) * 100) if (global_tp + global_fn) > 0 else 0
        global_precision = (global_tp / (global_tp + global_fp) * 100) if (global_tp + global_fp) > 0 else 0
        
        print(f"Total comparisons: {global_total}")
        print(f"Global matches: {global_match} ({global_match_pct:.2f}%)")
        print(f"Global mismatches: {global_mismatch} ({global_mismatch_pct:.2f}%)")
        print(f"Global True Positives: {global_tp}")
        print(f"Global True Negatives: {global_tn}")
        print(f"Global False Positives: {global_fp}")
        print(f"Global False Negatives: {global_fn}")
        print(f"Global Accuracy: {global_accuracy:.2f}%")
        print(f"Global Recall: {global_recall:.2f}%")
        print(f"Global Precision: {global_precision:.2f}%")


#/////////////////////////////////////////////////////
def compare_synergies(
        df_experiment: pandas.DataFrame,
        df_prediction: pandas.DataFrame,
        cell_line_list: list,
        synergy_column: str = 'synergy',
        threshold: float = 0.5,
        analysis_mode: str = 'cell_line', # 'cell_line' or 'inhibitor_combination'
        output_path: Optional[Union[str, Path]] = None
) -> Tuple[pandas.DataFrame, Dict]:
    """
    Compare experimental and predicted synergies values across different cell lines or inhibitor combinations.
    Collect the number of matches and mismatches, and calculate the confusion matrix.
    Only compares combinations that exist in both datasets.

    Args:
        df_experiment (pandas.DataFrame): DataFrame containing experimental synergy data.
        df_prediction (pandas.DataFrame): DataFrame containing predicted synergy data.
        cell_line_list (list): List of cell lines to include in the comparison.
        synergy_column (str): Column name for synergy values.
        threshold (float): Threshold value for determining synergy.
        type (str): Type of comparison ('cell_line' or 'inhibitor_combination').
    Returns:
        Tuple[pandas.DataFrame, Dict]: DataFrame containing the comparison results and a dictionary with skipped items info.
    """
    # Normalize inhibitor combinations to handle bidirectional combinations (e.g., A+B vs B+A)
    df_experiment = _normalize_combinations_in_df(df_experiment, 'inhibitor_combination')
    df_prediction = _normalize_combinations_in_df(df_prediction, 'inhibitor_combination')
    
    # Create boolean DataFrames for both experimental and predicted synergies
    df_experiment_bool = _make_boolean_df(df_experiment, cell_line_list, synergy_column, threshold)
    df_prediction_bool = _make_boolean_df(df_prediction, cell_line_list, synergy_column, threshold)

    # Pivot dataframes to have cell lines as columns and combinations as index
    df_exp = df_experiment_bool.pivot(index='inhibitor_combination', columns='cell_line', values=synergy_column)
    df_pred = df_prediction_bool.pivot(index='inhibitor_combination', columns='cell_line', values=synergy_column)

    # Find common items and items that will be skipped
    exp_items, pred_items, common_items, skipped_exp, skipped_pred = _find_common_and_skipped_items(df_exp, df_pred, analysis_mode)
    
    # Create skipped items report
    skipped_info = {
        'common_items': list(common_items),
        'total_common': len(common_items),
        'skipped_from_experimental': list(skipped_exp),
        'skipped_from_predicted': list(skipped_pred),
        'total_skipped_exp': len(skipped_exp),
        'total_skipped_pred': len(skipped_pred)
    }
    
    # Print initial summary using helper function
    _print_comparison_summary(analysis_mode, exp_items, pred_items, common_items, skipped_exp, skipped_pred,)
    
    if len(common_items) == 0:
        print("WARNING: No common items found between datasets!")
        return pandas.DataFrame(), skipped_info
    
    # Use only common items for comparison
    combi = list(common_items)
    
    # Match synergies
    match_counts, mismatch_counts = _match_synergies(df_exp, df_pred, combi)
    # Calculate confusion matrix
    confusion_matrix = _calculate_confusion_matrix(df_exp, df_pred, combi)
    # Add match and mismatch counts to the confusion matrix
    results_df = confusion_matrix.copy()
    match_results = {
        'Match': pandas.Series(match_counts),
        'Mismatch': pandas.Series(mismatch_counts)
    }
    results_df = results_df.join(pandas.DataFrame(match_results))
    # Calculate percentages
    results_df['Match %'] = results_df.apply(lambda row: (row['Match'] / row['Total']) * 100 if row['Total'] > 0 else 0, axis=1)
    results_df['Mismatch %'] = results_df.apply(lambda row: (row['Mismatch'] / row['Total']) * 100 if row['Total'] > 0 else 0, axis=1)
    
    # Calculate accuracy, recall, and precision
    results_df = _calculate_accuracy_recall_precision(results_df)
    
    # Print final summary with global results
    _print_comparison_summary(analysis_mode, results_df=results_df)

    if output_path:
        save_file(results_df, output_path / f"{analysis_mode}_comparison_results.csv", file_type='csv')

    return results_df, skipped_info
