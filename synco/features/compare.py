import pandas
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from ..utils import save_file, clean_cell_names

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
    s = boolean_df[synergy_column]
    # Compute comparison and immediately cast to pandas nullable boolean dtype
    # so we can safely assign <NA> without dtype-incompatibility warnings.
    bool_series = s.gt(threshold).astype('boolean')
    mask_na = s.isna()
    if mask_na.any():
        bool_series.loc[mask_na] = pandas.NA
    boolean_df[synergy_column] = bool_series
    return boolean_df


def _compute_item_diagnostics(df_exp: pandas.DataFrame, df_pred: pandas.DataFrame, analysis_mode: str):
    """
    Compute per-item diagnostics: missing pairs in pred/exp, present counts and overlap counts.
    Returns five dictionaries: missing_in_pred_by_item, missing_in_exp_by_item,
    present_counts_exp, present_counts_pred, overlap_counts
    Also returns the list `all_items` used for diagnostics.
    """
    missing_in_pred_by_item = {}
    missing_in_exp_by_item = {}
    present_counts_exp = {}
    present_counts_pred = {}
    overlap_counts = {}

    if analysis_mode == 'cell_line':
        all_items = sorted(set(df_exp.columns.tolist()) | set(df_pred.columns.tolist()))
    else:
        all_items = sorted(set(df_exp.index.tolist()) | set(df_pred.index.tolist()))

    for item in all_items:
        try:
            if analysis_mode == 'cell_line':
                exp_series = df_exp[item] if item in df_exp.columns else pandas.Series(index=df_exp.index, dtype='float')
                pred_series = df_pred[item] if item in df_pred.columns else pandas.Series(index=df_exp.index, dtype='float')
                pred_series = pred_series.reindex(exp_series.index)
            else:
                exp_series = df_exp.loc[item, :] if item in df_exp.index else pandas.Series(index=df_exp.columns, dtype='float')
                pred_series = df_pred.loc[item, :] if item in df_pred.index else pandas.Series(index=df_exp.columns, dtype='float')
                pred_series = pred_series.reindex(exp_series.index)

            missing_in_pred = int((~exp_series.isna() & pred_series.isna()).sum())
            missing_in_exp = int((~pred_series.isna() & exp_series.isna()).sum())

            n_exp = int((~exp_series.isna()).sum())
            n_pred = int((~pred_series.isna()).sum())
            n_overlap = int(((~exp_series.isna()) & (~pred_series.isna())).sum())

            missing_in_pred_by_item[item] = missing_in_pred
            missing_in_exp_by_item[item] = missing_in_exp
            present_counts_exp[item] = n_exp
            present_counts_pred[item] = n_pred
            overlap_counts[item] = n_overlap
        except Exception:
            missing_in_pred_by_item[item] = 0
            missing_in_exp_by_item[item] = 0
            present_counts_exp[item] = 0
            present_counts_pred[item] = 0
            overlap_counts[item] = 0

    return missing_in_pred_by_item, missing_in_exp_by_item, present_counts_exp, present_counts_pred, overlap_counts, all_items


def _build_skipped_info(df_exp: pandas.DataFrame, df_pred: pandas.DataFrame, analysis_mode: str,
                        missing_in_pred_by_item: dict, missing_in_exp_by_item: dict,
                        present_counts_exp: dict, present_counts_pred: dict, overlap_counts: dict):
    """
    Build the skipped_info dictionary from diagnostics and common/skipped items.
    Returns skipped_info and the tuple (exp_items, pred_items, common_items, skipped_exp, skipped_pred)
    """
    exp_items, pred_items, common_items, skipped_exp, skipped_pred = _find_common_and_skipped_items(df_exp, df_pred, analysis_mode)

    skipped_info = {
        'common_items': list(common_items),
        'total_common': len(common_items),
        'skipped_from_experimental': list(skipped_exp),
        'skipped_from_predicted': list(skipped_pred),
        'total_skipped_exp': len(skipped_exp),
        'total_skipped_pred': len(skipped_pred)
    }

    skipped_info['missing_pairs_in_pred_by_item'] = missing_in_pred_by_item
    skipped_info['missing_pairs_in_exp_by_item'] = missing_in_exp_by_item
    skipped_info['total_missing_pairs_in_pred'] = sum(missing_in_pred_by_item.values())
    skipped_info['total_missing_pairs_in_exp'] = sum(missing_in_exp_by_item.values())

    skipped_info['present_counts_exp'] = present_counts_exp
    skipped_info['present_counts_pred'] = present_counts_pred
    skipped_info['overlap_counts'] = overlap_counts

    return skipped_info, (exp_items, pred_items, common_items, skipped_exp, skipped_pred)


def _build_fn_fp_examples(df_exp: pandas.DataFrame, df_pred: pandas.DataFrame, combi: list):
    """Build small example lists for false negatives / false positives per item."""
    fn_examples = {}
    fp_examples = {}
    for item in combi:
        try:
            if item in df_exp.columns:
                exp_series = df_exp[item]
                pred_series = df_pred[item]
            elif item in df_exp.index:
                exp_series = df_exp.loc[item, :]
                pred_series = df_pred.loc[item, :]
            else:
                fn_examples[item] = []
                fp_examples[item] = []
                continue

            pred_series = pred_series.reindex(exp_series.index)
            valid_mask = (~exp_series.isna()) & (~pred_series.isna())

            fn_mask = (exp_series.eq(True) & pred_series.eq(False) & valid_mask)
            fp_mask = (exp_series.eq(False) & pred_series.eq(True) & valid_mask)

            fn_list = list(exp_series.index[fn_mask][:10].astype(str)) if hasattr(exp_series.index, 'astype') else list(fn_mask[fn_mask].index)[:10]
            fp_list = list(exp_series.index[fp_mask][:10].astype(str)) if hasattr(exp_series.index, 'astype') else list(fp_mask[fp_mask].index)[:10]

            fn_examples[item] = fn_list
            fp_examples[item] = fp_list
        except Exception:
            fn_examples[item] = []
            fp_examples[item] = []

    return fn_examples, fp_examples


def _debug_print_items(df_experiment, df_prediction, df_exp, df_pred, debug_items, present_counts_exp, present_counts_pred, overlap_counts):
    """Optional debug printing for selected items."""
    if not debug_items:
        return
    try:
        print("\nDEBUG: compare_synergies diagnostics for items:", debug_items)
        for item in debug_items:
            print(f"\n--- DEBUG ITEM: {item} ---")
            try:
                if 'cell_line' in df_experiment.columns:
                    rows_exp_raw = df_experiment[df_experiment['cell_line'] == item]
                    print("Raw df_experiment rows (cell_line filter):")
                    print(rows_exp_raw.to_string(index=False) if not rows_exp_raw.empty else f"  (no rows in df_experiment with cell_line == {item})")
                else:
                    print("  df_experiment has no 'cell_line' column")
            except Exception as e:
                print("  Error inspecting raw experimental rows:", e)

            try:
                if 'cell_line' in df_prediction.columns:
                    rows_pred_raw = df_prediction[df_prediction['cell_line'] == item]
                    print("Raw df_prediction rows (cell_line filter):")
                    print(rows_pred_raw.to_string(index=False) if not rows_pred_raw.empty else f"  (no rows in df_prediction with cell_line == {item})")
                else:
                    print("  df_prediction has no 'cell_line' column")
            except Exception as e:
                print("  Error inspecting raw prediction rows:", e)

            try:
                print("df_exp shape:", getattr(df_exp, 'shape', None))
                print("df_pred shape:", getattr(df_pred, 'shape', None))
                print(f"Item in df_pred.columns: {item in getattr(df_pred, 'columns', [])}")
                print(f"Item in df_exp.columns: {item in getattr(df_exp, 'columns', [])}")

                if item in getattr(df_pred, 'columns', []):
                    s = df_pred[item]
                    print("df_pred column head (first 20):")
                    print(s.head(20).to_string())
                    print("Number of NaNs in df_pred column:", int(s.isna().sum()))
                else:
                    print("df_pred has no column for item")

                if item in getattr(df_exp, 'columns', []):
                    s2 = df_exp[item]
                    print("df_exp column head (first 20):")
                    print(s2.head(20).to_string())
                    print("Number of NaNs in df_exp column:", int(s2.isna().sum()))
                else:
                    print("df_exp has no column for item")

                print("present_counts_exp:", present_counts_exp.get(item))
                print("present_counts_pred:", present_counts_pred.get(item))
                print("overlap_counts:", overlap_counts.get(item))
            except Exception as e:
                print("  Error in pivoted diagnostics:", e)

            try:
                if item in getattr(df_pred, 'index', []):
                    row = df_pred.loc[item]
                    print("df_pred row for combination (head):")
                    print(row.head(20).to_string())
                if item in getattr(df_exp, 'index', []):
                    row2 = df_exp.loc[item]
                    print("df_exp row for combination (head):")
                    print(row2.head(20).to_string())
            except Exception:
                pass
    except Exception:
        pass

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

        # Only compare where both are not NaN
        valid_mask = (~df_exp.isna()) & (~df_pred.isna())
        match_count = ((df_exp == df_pred) & valid_mask).sum()
        mismatch_count = ((df_exp != df_pred) & valid_mask).sum()

        match_counts[key_label] = match_count
        mismatch_counts[key_label] = mismatch_count

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
        # Only compare where both are not NaN
        valid_mask = (~df_exp.isna()) & (~df_pred.isna())
        # Calculate True Positives, True Negatives, False Positives, and False Negatives only for valid pairs
        # Use .eq(True)/.eq(False) to safely handle pandas nullable booleans (pd.NA)
        tp = ((df_exp.eq(True)) & (df_pred.eq(True)) & valid_mask).sum()
        tn = ((df_exp.eq(False)) & (df_pred.eq(False)) & valid_mask).sum()
        fp = ((df_exp.eq(False)) & (df_pred.eq(True)) & valid_mask).sum()
        fn = ((df_exp.eq(True)) & (df_pred.eq(False)) & valid_mask).sum()
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
) -> Optional[dict]:
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
        # Calculate global totals and metrics, return summary dict
        global_tp = int(results_df['True Positive'].sum())
        global_tn = int(results_df['True Negative'].sum())
        global_fp = int(results_df['False Positive'].sum())
        global_fn = int(results_df['False Negative'].sum())
        global_total = int(results_df['Total'].sum())
        global_match = int(results_df['Match'].sum()) if 'Match' in results_df.columns else None
        global_mismatch = int(results_df['Mismatch'].sum()) if 'Mismatch' in results_df.columns else None

        global_match_pct = (global_match / global_total * 100) if (global_total and global_match is not None) else 0
        global_mismatch_pct = (global_mismatch / global_total * 100) if (global_total and global_mismatch is not None) else 0

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

        # Build and return summary dict
        return {
            'Total comparisons': int(global_total),
            'Global matches': int(global_match) if global_match is not None else None,
            'Global matches %': round(global_match_pct, 2),
            'Global mismatches': int(global_mismatch) if global_mismatch is not None else None,
            'Global mismatches %': round(global_mismatch_pct, 2),
            'Global True Positives': int(global_tp),
            'Global True Negatives': int(global_tn),
            'Global False Positives': int(global_fp),
            'Global False Negatives': int(global_fn),
            'Global Accuracy %': round(global_accuracy, 2),
            'Global Recall %': round(global_recall, 2),
            'Global Precision %': round(global_precision, 2)
        }

#/////////////////////////////////////////////////////
def _handle_duplicates_before_pivot(
        df: pandas.DataFrame, 
        key_columns: list, 
        value_column: str, 
    strategy: str = "mean",
    threshold: Optional[float] = None
) -> pandas.DataFrame:
    """
    Handle duplicate entries before pivot operation.
    
    Args:
        df: DataFrame with potential duplicates
        key_columns: List of columns that define unique combinations  
        value_column: Column containing values to aggregate
        strategy: "mean" or "ignore" (keep first)
    
    Returns:
        DataFrame with duplicates resolved
    """
    # Check for duplicates
    duplicates = df.duplicated(subset=key_columns).sum()
    
    if duplicates == 0:
        return df
    
    print(f"Found {duplicates} duplicate entries for pivot operation")
    
    if strategy == "mean":
        print(f"Strategy: MEAN - Averaging {value_column} values for duplicate combinations")
        # Group by key columns and calculate mean for value column
        result = df.groupby(key_columns, as_index=False)[value_column].mean()
        
        # Add back any other columns (take first occurrence)
        other_cols = [col for col in df.columns if col not in key_columns + [value_column]]
        if other_cols:
            other_data = df.groupby(key_columns, as_index=False)[other_cols].first()
            result = result.merge(other_data, on=key_columns)
        # If the aggregated value column is numeric (averaged booleans), convert back to boolean
        # using the provided threshold (if any). This avoids later bitwise operations on floats.
        if pandas.api.types.is_numeric_dtype(result[value_column].dtype):
            if threshold is not None:
                result[value_column] = result[value_column] > threshold
            else:
                # Default: treat average > 0.5 as True
                result[value_column] = result[value_column] > 0.5
            
    elif strategy == "ignore":
        print("Strategy: IGNORE - Keeping first occurrence of duplicate combinations")
        # Keep only the first occurrence of each duplicate
        result = df.drop_duplicates(subset=key_columns, keep='first')
        
    else:
        raise ValueError(f"Unknown duplicate strategy: {strategy}. Use 'mean' or 'ignore'")
    
    print(f"Original rows: {len(df)}, After handling duplicates: {len(result)}")
    return result

#/////////////////////////////////////////////////////
def compare_synergies(
        df_experiment: pandas.DataFrame,
        df_prediction: pandas.DataFrame,
        cell_line_list: list,
        synergy_column: str = 'synergy',
        threshold: float = 0.5,
        analysis_mode: str = 'cell_line', # 'cell_line' or 'inhibitor_combination'
        duplicate_strategy: str = 'mean', # 'mean' or 'ignore'
        output_path: Optional[Union[str, Path]] = None,
        debug_items: Optional[list] = None,
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
        analysis_mode (str): Type of comparison ('cell_line' or 'inhibitor_combination').
        duplicate_strategy (str): How to handle duplicates ('mean' or 'ignore').
        output_path (Optional[Union[str, Path]]): Path to save output files.
        
    Returns:
        Tuple[pandas.DataFrame, Dict]: DataFrame containing the comparison results and a dictionary with skipped items info.
    """
    # Normalize inhibitor combinations to handle bidirectional combinations (e.g., A+B vs B+A)
    df_experiment = _normalize_combinations_in_df(df_experiment, 'inhibitor_combination')
    df_prediction = _normalize_combinations_in_df(df_prediction, 'inhibitor_combination')

    # Clean cell line names
    df_experiment = clean_cell_names(df_experiment)
    df_prediction = clean_cell_names(df_prediction)

    # Create boolean DataFrames for both experimental and predicted synergies
    df_experiment_bool = _make_boolean_df(df_experiment, cell_line_list, synergy_column, threshold)
    df_prediction_bool = _make_boolean_df(df_prediction, cell_line_list, synergy_column, threshold)

    # Handle duplicates before pivot operation
    pivot_key_cols = ['inhibitor_combination', 'cell_line']
    df_experiment_bool = _handle_duplicates_before_pivot(
        df_experiment_bool, 
        key_columns=pivot_key_cols, 
        value_column=synergy_column,
        strategy=duplicate_strategy,
        threshold=threshold
    )
    df_prediction_bool = _handle_duplicates_before_pivot(
        df_prediction_bool,
        key_columns=pivot_key_cols,
        value_column=synergy_column, 
        strategy=duplicate_strategy,
        threshold=threshold
    )

    # Pivot dataframes to have cell lines as columns and combinations as index
    df_exp = df_experiment_bool.pivot(index='inhibitor_combination', columns='cell_line', values=synergy_column)
    df_pred = df_prediction_bool.pivot(index='inhibitor_combination', columns='cell_line', values=synergy_column)

    # Compute per-item diagnostics (missing pairs, present counts, overlap counts)
    missing_in_pred_by_item, missing_in_exp_by_item, present_counts_exp, present_counts_pred, overlap_counts, all_items = _compute_item_diagnostics(
        df_exp, df_pred, analysis_mode
    )

    # Build skipped info using helper that includes common/skipped items and diagnostics
    skipped_info, (_exp_items, _pred_items, common_items, skipped_exp, skipped_pred) = _build_skipped_info(
        df_exp, df_pred, analysis_mode,
        missing_in_pred_by_item, missing_in_exp_by_item,
        present_counts_exp, present_counts_pred, overlap_counts
    )

    # Preserve exp_items / pred_items variables for summary printing
    exp_items = set(_exp_items)
    pred_items = set(_pred_items)
    
    # Print initial summary using helper function
    _print_comparison_summary(analysis_mode, exp_items, pred_items, common_items, skipped_exp, skipped_pred,)

    # Print concise missing-pairs summary so notebooks show what's ignored
    # This reports per-item counts where experimental entries exist but predictions
    # are missing (and vice versa). It helps catch cases like a cell line that
    # has fewer predicted combinations.
    try:
        # Items with missing predictions
        missing_pred_items = {k: v for k, v in missing_in_pred_by_item.items() if v > 0}
        missing_exp_items = {k: v for k, v in missing_in_exp_by_item.items() if v > 0}

        if missing_pred_items:
            print("\nItems with missing prediction pairs (experimental present, prediction missing):")
            for k, v in sorted(missing_pred_items.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")

        if missing_exp_items:
            print("\nItems with missing experimental pairs (prediction present, experimental missing):")
            for k, v in sorted(missing_exp_items.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")

        if not missing_pred_items and not missing_exp_items:
            print("\nNo missing pairs detected between experimental and predicted data (per-item).")
    except Exception:
        # Don't fail the analysis because printing failed
        pass
    
    if len(common_items) == 0:
        print("WARNING: No common items found between datasets!")
        return pandas.DataFrame(), skipped_info
    
    # Use only common items for comparison
    combi = list(common_items)
    
    # Match synergies
    match_counts, mismatch_counts = _match_synergies(df_exp, df_pred, combi)
    # Calculate confusion matrix
    confusion_matrix = _calculate_confusion_matrix(df_exp, df_pred, combi)
    # Build small diagnostic examples for False Negatives / False Positives
    # so users can inspect which combinations caused the counts.
    fn_examples, fp_examples = _build_fn_fp_examples(df_exp, df_pred, combi)
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

    # Print final summary with global results and get summary dict (no duplicated computation)
    try:
        summary_dict = _print_comparison_summary(analysis_mode, results_df=results_df)
    except Exception as e:
        summary_dict = {'error_building_summary': str(e)}

    # Attach summary to skipped_info and save if requested
    skipped_info['global_summary'] = summary_dict

    # Optional debug printing for selected items (cell lines or inhibitor combinations)
    _debug_print_items(df_experiment, df_prediction, df_exp, df_pred, debug_items, present_counts_exp, present_counts_pred, overlap_counts)

    if output_path and summary_dict is not None:
        outp = Path(output_path)
        outp.mkdir(parents=True, exist_ok=True)
        save_file(summary_dict, outp / f"{analysis_mode}_comparison_summary.json", file_type='json')
        summary_lines = [f"{k}: {v}" for k, v in summary_dict.items()]
        summary_text = "\n".join(summary_lines)
        save_file(summary_text, outp / f"{analysis_mode}_comparison_summary.txt", file_type='txt')

    if output_path:
        save_file(results_df, Path(output_path) / f"{analysis_mode}_comparison_results.csv", file_type='csv')

    return results_df, skipped_info, summary_dict