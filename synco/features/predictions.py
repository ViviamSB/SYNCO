import pandas
from pathlib import Path
from typing import Optional, Union

from ..utils import split_column, apply_mapping, deduplicate_list_column, save_file
PathLike = Union[str, Path]

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN FEATURE: synergy_predictions
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////
def _merge_synergies(
        synergy_results_dict: dict,
        ) -> pandas.DataFrame:
    """
    Merge pipeline predictions for all cell lines into a single DataFrame.
    Handles different data lengths by filling missing values with NaN.
    Handles both scenarios for observations:
    1. Observations with synergy values
    2. Observations with only perturbation names
    """
    synergy_predictions_df = pandas.DataFrame()
    synergy_observations_df = pandas.DataFrame()
    
    for cell_line, (observed, predicted) in synergy_results_dict.items():
        # Handle predictions
        if predicted is not None:
            if synergy_predictions_df.empty:
                synergy_predictions_df = predicted.copy()
            else:
                synergy_predictions_df = pandas.merge(
                    synergy_predictions_df,
                    predicted,
                    on='Perturbation',
                    how='outer',
                    suffixes=('', f'_{cell_line}')
                )
        else:
            print(f"Warning: No predictions for cell line {cell_line}. Skipping.")
        
        # Handle observations
        if observed is not None:
            # Check if observed data has synergy values or just perturbation names
            value_columns = [col for col in observed.columns if col != 'Perturbation']
            
            if len(value_columns) > 0:
                # Scenario 1: Has synergy values - merge normally
                    # Select synergy column
                observed = observed[['Perturbation', 'synergy']].copy()
                observed.rename(columns={'synergy': f'Experimental_{cell_line}'}, inplace=True) 
                if synergy_observations_df.empty:
                    synergy_observations_df = observed.copy()
                else:
                    synergy_observations_df = pandas.merge(
                        synergy_observations_df,
                        observed,
                        on='Perturbation',
                        how='outer'
                    )
            else:
                # Scenario 2: Only perturbation names - add a flag column
                observed_with_flag = observed.copy()
                observed_with_flag[f'Experimental_{cell_line}'] = 1
                
                if synergy_observations_df.empty:
                    synergy_observations_df = observed_with_flag.copy()
                else:
                    synergy_observations_df = pandas.merge(
                        synergy_observations_df,
                        observed_with_flag,
                        on='Perturbation',
                        how='outer'
                    )
        # else:
            # Silently skip cell lines with no experimental observations
    
    return synergy_predictions_df, synergy_observations_df

#/////////////////////////////////////////////////////
def _split_perturbations(
        df: pandas.DataFrame,   # synergy_predictions_df
        source_col: str = 'Perturbation',
    ) -> pandas.DataFrame:
    """
    Split the 'Perturbation' into drug pairs.
    """
    new_cols = ['PD_A', 'PD_B']
    df = split_column(df, source_col, new_cols, separator='-')
    return df

#/////////////////////////////////////////////////////
def _map_drug_pairs(
        df: pandas.DataFrame,
        mapping_dict: dict,
        mapping_type: str,
        remove_duplicates: bool = False
    ) -> pandas.DataFrame:
    """
    Map drug pairs to their corresponding values (names, targets, etc.) based on the mapping type.
    
    Args:
        df (DataFrame): DataFrame containing drug pairs.
        mapping_dict (dict): Dictionary containing the mapping data. Can be drug names or targets.
        mapping_type (str): Type of mapping to apply (values from mapping_dict).
        remove_duplicates (bool): Whether to remove duplicate values in the new columns.
    
    Returns:
        DataFrame: DataFrame with new mapped columns.
    """
    new_cols = [f'{mapping_type}_A', f'{mapping_type}_B']
    key_cols = ['PD_A', 'PD_B']

    # Apply mapping for both drugs in the pair
    for i, key_col in enumerate(key_cols):
        df = apply_mapping(
            df,
            key_col=key_col,
            new_col=new_cols[i],
            dictionary=mapping_dict
        )
    
    # Remove duplicate values if requested
    if remove_duplicates:
        df = deduplicate_list_column(df, new_cols)
    
    return df

#/////////////////////////////////////////////////////
def _add_experimental_observations(
        df: pandas.DataFrame,   # DataFrame to add experimental observations to. Should be synergy_predictions_df
        synergy_observations_df: pandas.DataFrame,  # From _merge_synergies results
    ) -> pandas.DataFrame:
    """
    Add experimental observations to the DataFrame.
    
    Args:
        df (DataFrame): DataFrame to add experimental observations to.
        synergy_observations_df (DataFrame): DataFrame containing experimental observations.
    """
    # Merge the experimental observations
    df = pandas.merge(
        df,
        synergy_observations_df,
        on='Perturbation',
        how='left'
    )
    
    return df

#/////////////////////////////////////////////////////
def _create_combinations(
        df: pandas.DataFrame,
        combination_type: str = 'drugnames',    # 'drugnames' or 'targets'
        drugID_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> pandas.DataFrame:
    """
    Create combinations of drug pairs based on the specified type.
    Args:
        df (DataFrame): DataFrame containing drug pairs.
        combination_type (str): Type of combination to create ('drugnames' or 'targets').
        drugID_type (str): Type of drug ID to use for drug names.
        target_type (str): Type of target to use for targets.
    """
    if combination_type == 'drugnames':
        # Handle case where drug name columns might contain lists
        drug_a_col = drugID_type + '_A'
        drug_b_col = drugID_type + '_B'
        
        # Convert lists to strings if needed
        df[drug_a_col] = df[drug_a_col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        df[drug_b_col] = df[drug_b_col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        df['drug_combination'] = '(' + df[drug_a_col] + ')' + ' + ' + '(' + df[drug_b_col] + ')'
        df = deduplicate_list_column(df, 'drug_combination', as_string=True, separator=' + ')

    elif combination_type == 'targets':
        # Handle case where target columns might contain lists or strings
        target_a_col = target_type + '_A'
        target_b_col = target_type + '_B'
        
        # Convert to strings if needed
        df[target_a_col] = df[target_a_col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        df[target_b_col] = df[target_b_col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        df['target_combination'] = '(' + df[target_a_col] + ')' + ' + ' + '(' + df[target_b_col] + ')'
        df = deduplicate_list_column(df, 'target_combination', as_string=True, separator=' + ')
    else:
        raise ValueError(f"Invalid combination_type '{combination_type}'. Expected 'drugnames' or 'targets'.")     

    return df

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# Main methods: get_synergy_predictions
#///////////////////////////////////////////////////////////////////////////////////////////////////////
def get_synergy_predictions(
        synergy_results_dict: dict,
        combination_type: str = 'drugnames',  # 'drugnames' or 'targets'
        mapping_names_dict: Optional[dict] = None,
        mapping_target_dict: Optional[dict] = None,
        drugID_type: str = 'drug_name',
        target_type: str = 'targets',
        remove_duplicates: bool = False,
        add_experimental_observations: bool = False,
        output_path: Optional[PathLike] = None
    ) -> pandas.DataFrame:
    """
    Process synergy predictions from the results dictionary.
    Args:
        synergy_results_dict (dict): Dictionary containing synergy results for each cell line (predicted and observed synergies).
        combination_type (str): Type of combination to create ('drugnames' or 'targets').
        mapping_dict (dict): Dictionary containing the mapping data. Can be drug names or targets.
        mapping_type (str): Type of mapping to apply (values from mapping_dict).
        remove_duplicates (bool): Whether to remove duplicate entries.
        add_experimental_observations (bool): Whether to add experimental observations. Default is False.
    Returns:
        DataFrame: Processed synergy predictions DataFrame.
    """
    # Step 1: Merge synergy results from all cell lines
    synergy_predictions_df, synergy_observations_df = _merge_synergies(synergy_results_dict)
    # Step 2: Split the 'Perturbation' column into drug pairs
    synergy_predictions_df = _split_perturbations(synergy_predictions_df)
    # Step 3: Map drug pairs to their corresponding values based on the mapping type
    if mapping_names_dict is not None:
        synergy_predictions_df = _map_drug_pairs(synergy_predictions_df, mapping_names_dict, drugID_type, remove_duplicates)
    if mapping_target_dict is not None:
        synergy_predictions_df = _map_drug_pairs(synergy_predictions_df, mapping_target_dict, target_type, remove_duplicates)
    # Step 4: Create combinations of drug pairs based on the specified type
    synergy_predictions_df = _create_combinations(synergy_predictions_df, combination_type, drugID_type, target_type)
    # Step 5: Add experimental observations (Optional)
    if add_experimental_observations:
        synergy_predictions_df = _add_experimental_observations(synergy_predictions_df, synergy_observations_df)
    
    if output_path:
        save_file(synergy_predictions_df, output_path / 'synergy_predictions.csv')

    return synergy_predictions_df


