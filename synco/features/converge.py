import pandas
from pathlib import Path
from typing import Optional, Union

from ..utils import apply_mapping, save_file, clean_cell_names

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN FEATURE: converge_synergies
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////
PathLike = Union[str, Path]

def _map_inhibitor_groups(
        df: pandas.DataFrame,
        anchorID: str,
        libraryID: str,
        inhibitor_groups: dict
) -> pandas.DataFrame:
    """
    Map inhibitor groups to the synergy DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        anchorID (str): Column name for anchor IDs.
        libraryID (str): Column name for library IDs.
        inhibitor_groups (dict): Dictionary mapping perturbation pairs to inhibitor groups.

    Returns:
        pandas.DataFrame: DataFrame with mapped inhibitor groups.
    """
    df = apply_mapping(df, anchorID, 'inhibitor_group_A', inhibitor_groups)
    df = apply_mapping(df, libraryID, 'inhibitor_group_B', inhibitor_groups)
    df['inhibitor_combination'] = df['inhibitor_group_A'] + ' + ' + df['inhibitor_group_B']
    return df

#/////////////////////////////////////////////////////
def _map_targets(
        df: pandas.DataFrame,
        anchorID: str,
        libraryID: str,
        targets_dict: dict
) -> pandas.DataFrame:
    """
    Map targets to the synergy DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        anchorID (str): Column name for anchor IDs.
        libraryID (str): Column name for library IDs.
        targets_dict (dict): Dictionary mapping perturbation pairs to targets.

    Returns:
        pandas.DataFrame: DataFrame with mapped targets.
    """
    df = apply_mapping(df, anchorID, 'targets_A', targets_dict)
    df = apply_mapping(df, libraryID, 'targets_B', targets_dict)
    df['target_combination'] = df['targets_A'] + ' + ' + df['targets_B']
    return df

#/////////////////////////////////////////////////////
def _group_by_inhibitor_combination(
        df: pandas.DataFrame,
        synergy_column: str,
        cell_line: str,
) -> pandas.DataFrame:
    """
    Group the DataFrame by inhibitor combination and cell line and calculate the mean synergy.

    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        synergy_column (str): Column name for synergy values.
        cell_line (str): Column name for cell line identifiers.

    Returns:
        pandas.DataFrame: Grouped DataFrame with mean synergy values.
    """
    mean_synergies = df.groupby(['inhibitor_combination', cell_line])[synergy_column].mean().reset_index()
    mean_synergies = mean_synergies.rename(columns={synergy_column: 'mean_synergy'})
    return mean_synergies

#/////////////////////////////////////////////////////
def _process_predictions(
        df: pandas.DataFrame,
        synergy_column: str,
        cell_line: str,
        cell_line_list: list
) -> pandas.DataFrame:
    """
    Process predictions by melting the DataFrame to long format and cleaning cell line names.

    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        synergy_column (str): Column name for synergy values.
        cell_line (str): Column name for cell line identifiers.
        cell_line_list (list): List of cell line names to filter the DataFrame.

    Returns:
        tuple: A tuple containing two DataFrames:
            - inhibitor_df: DataFrame with melted format and cleaned cell line names for inhibitors.
            - drugnames_df: DataFrame with melted format and cleaned cell line names for drug combinations.
    """

    inhibitor_df = df.set_index(['inhibitor_combination'])
    drugnames_df = df.set_index(['drug_combination'])
    
    # Use original cell_line_list since DataFrame columns haven't been cleaned yet
    inhibitor_df = inhibitor_df.loc[:, inhibitor_df.columns.isin(cell_line_list)]
    if inhibitor_df.shape[1] == 0:
        raise ValueError("None of the cell_line_list columns matched columns in inhibitor_df.")
    inhibitor_df = inhibitor_df.reset_index().melt(
        id_vars=['inhibitor_combination'],
        var_name=cell_line,
        value_name=synergy_column
    )

    drugnames_df = drugnames_df.loc[:, drugnames_df.columns.isin(cell_line_list)]
    if drugnames_df.shape[1] == 0:
        raise ValueError("None of the cell_line_list columns matched columns in drugnames_df.")
    drugnames_df = drugnames_df.reset_index().melt(
        id_vars=['drug_combination'],
        var_name=cell_line,
        value_name=synergy_column
    )

    processed = [inhibitor_df, drugnames_df]
    for df in processed:
        df[synergy_column] = df[synergy_column] * -1
        # Clean cell line names AFTER melting
        df[cell_line] = df[cell_line].astype(str).str.upper().str.replace("-", "")

    return inhibitor_df, drugnames_df

#/////////////////////////////////////////////////////
def _process_experimental(
        df: pandas.DataFrame,
        synergy_column: str = 'synergy',
        cell_line: str = 'cell_line',
        cell_line_list: Optional[list] = None
) -> pandas.DataFrame:
    """
    Process experimental synergies by:
    - Grouping by inhibitor combination and cell line
    - Calculating mean synergy values
    - Melting the DataFrame to long format
    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        synergy_column (str): Column name for synergy values.
        cell_line (str): Column name for cell line identifiers.
        cell_line_list (list, optional): List of cell line names to filter the DataFrame.

    Returns:
        pandas.DataFrame: Processed DataFrame with mean synergy values.
    """
    df = clean_cell_names(df, column=cell_line)
    tissue_map = None
    if 'tissue' in df.columns:
        tissue_map = df[[cell_line, 'tissue']].dropna(subset=[cell_line, 'tissue'])
        tissue_map = tissue_map.drop_duplicates(subset=[cell_line]).set_index(cell_line)['tissue']
    
    if cell_line_list is not None:
        # Clean cell_line_list to match the cleaned DataFrame names
        cell_line_list_cleaned = [name.upper().replace('-', '') for name in cell_line_list]
        df = df[df[cell_line].isin(cell_line_list_cleaned)]

    # Group by inhibitor combination and cell line, and calculate mean synergy
    inhibitor_group_synergies_df = _group_by_inhibitor_combination(df, synergy_column, cell_line)
    inhibitor_group_synergies_df = inhibitor_group_synergies_df.pivot(index='inhibitor_combination', columns=cell_line, values='mean_synergy')
    inhibitor_group_synergies_df = inhibitor_group_synergies_df.reset_index().melt(
        id_vars=['inhibitor_combination'],
        var_name=cell_line,
        value_name=synergy_column
    )

    # Group by drug combination and cell line to handle duplicates
    drug_names_mean_synergies = df.groupby(['drug_combination', cell_line])[synergy_column].mean().reset_index()
    drug_names_synergies_df = drug_names_mean_synergies.pivot(index='drug_combination', columns=cell_line, values=synergy_column)
    drug_names_synergies_df = drug_names_synergies_df.reset_index().melt(
        id_vars=['drug_combination'],
        var_name=cell_line,
        value_name=synergy_column
    )

    # Attach tissue annotation per cell line if available.
    if tissue_map is not None and not tissue_map.empty:
        inhibitor_group_synergies_df['tissue'] = inhibitor_group_synergies_df[cell_line].map(tissue_map)
        drug_names_synergies_df['tissue'] = drug_names_synergies_df[cell_line].map(tissue_map)
    return inhibitor_group_synergies_df, drug_names_synergies_df

#/////////////////////////////////////////////////////
def converge_synergies(
        df: pandas.DataFrame,
        anchorID: str,
        libraryID: str,
        inhibitor_groups: dict,
        targets_dict: dict,
        synergy_column: str = 'synergy',
        cell_line: str = 'cell_line',
        cell_line_list: Optional[list] = None,
        anchor_name: Optional[str] = None,
        library_name: Optional[str] = None,
        predicted: bool = False,
        output_path: Optional[PathLike] = None
) -> pandas.DataFrame:
    """
    Converge experimental synergies by mapping inhibitor groups and targets, and calculating mean synergy.

    Args:
        df (pandas.DataFrame): DataFrame containing synergy data.
        anchorID (str): Column name for anchor IDs.
        libraryID (str): Column name for library IDs.
        inhibitor_groups (dict): Dictionary mapping perturbation pairs to inhibitor groups.
        targets_dict (dict): Dictionary mapping perturbation pairs to targets.
        synergy_column (str): Column name for synergy values.
        cell_line (str): Column name for cell line identifiers.

    Returns:
        tuple: A tuple containing three DataFrames:
            - full_df: DataFrame with mapped inhibitor groups and targets.
            - drug_names_synergies_df: DataFrame with drug combination synergies.
            - inhibitor_group_synergies_df: DataFrame with inhibitor group synergies.
    """
    # Check columns, if not present, create it
    if 'Perturbation' not in df.columns:
        df['Perturbation'] = df.apply(lambda row: f"{row[anchorID]}-{row[libraryID]}", axis=1)

    if 'drug_combination' not in df.columns and not (anchor_name and library_name):
        raise ValueError("The DataFrame must contain 'drug_combination' column or anchor_name and library_name must be provided.")
    if 'drug_combination' not in df.columns and anchor_name and library_name:
            df['drug_combination'] = df.apply(lambda row: f"{row[anchor_name]} + {row[library_name]}", axis=1)

    # Map inhibitor groups and targets
    df = _map_inhibitor_groups(df, anchorID, libraryID, inhibitor_groups)
    df = _map_targets(df, anchorID, libraryID, targets_dict)
    full_df = df.copy()

    # Remove rows with NaN values
    df = df.dropna(subset=['Perturbation', 'inhibitor_combination', 'target_combination'])

    # Process predicted synergies
    if predicted: 
        inhibitor_group_synergies_df, drug_names_synergies_df = _process_predictions(df, synergy_column, cell_line, cell_line_list)
    
    if not predicted:
        inhibitor_group_synergies_df, drug_names_synergies_df = _process_experimental(df, synergy_column, cell_line, cell_line_list)

    if output_path:
        # Use different filenames based on whether it's predicted or experimental data
        data_type = "predictions" if predicted else "experimental"
        save_file(full_df, output_path / f'{data_type}_full_df.csv')
        save_file(drug_names_synergies_df, output_path / f'{data_type}_drug_names_synergies_df.csv')
        save_file(inhibitor_group_synergies_df, output_path / f'{data_type}_inhibitor_group_synergies_df.csv')

    return full_df, drug_names_synergies_df, inhibitor_group_synergies_df