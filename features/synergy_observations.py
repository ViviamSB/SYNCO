import pandas
from pathlib import Path
from typing import Optional, Union

from utils import filter_synergies, create_cell_line_dir, get_output_file

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN FEATURE: get_experimental_cell_synergies
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////
PathLike = Union[str, Path]

def _extract_synergies(
        output_file: Path,
        synergies_exp_df: pandas.DataFrame,
        column_anchorID: str,
        column_libraryID: str,
        column_synergy: Optional[str] = None,
) -> None:
    """
    Write the synergy data with PD profile IDs to a file.
    
    Args:
        output_file (Path): The path to the output file.
        synergies_exp_df (pandas.DataFrame): DataFrame containing synergy data.
        column_anchorID (str): Column name for anchor IDs.
        column_libraryID (str): Column name for library IDs.
        column_synergy (Optional[str]): Column name for synergy status, if available.
    """
    unique_combinations = set()
    try:
        with open(output_file, 'w') as f:
            # write header if synergy column is provided
            if column_synergy:
                f.write(f"Perturbation\tanchor_ID\tlibrary_ID\t{column_synergy}\n")
            else:
                f.write("Perturbation\n")

            for _, row in synergies_exp_df.iterrows():
                anchor_id = row[column_anchorID]
                library_id = row[column_libraryID]            
                # Skip rows with NaN values in ID columns
                if pandas.isna(anchor_id) or pandas.isna(library_id):
                    continue                    
                synergy_value = row[column_synergy] if column_synergy else None
                sorted_ids = tuple(sorted([anchor_id, library_id]))
                perturbation_pair = f"{sorted_ids[0]}~{sorted_ids[1]}"
                if sorted_ids not in unique_combinations:
                    unique_combinations.add(sorted_ids)
                    if synergy_value is not None:
                        f.write(f"{perturbation_pair}\t{sorted_ids[0]}\t{sorted_ids[1]}\t{synergy_value}\n")
                    else:
                        f.write(f"{perturbation_pair}\n")
    except Exception as e:
        print(f"Error occurred while processing the DataFrame: {e}")

#/////////////////////////////////////////////////////
def get_experimental_cell_synergies(
        cell_line: str,
        synergies_exp_df: pandas.DataFrame,
        output_directory: PathLike,
        column_cell_line_name: str = 'cell_line',
        column_synergy: str = 'synergy',
        anchorID: str = 'anchor_ID',
        libraryID: str = 'library_ID',
        synergy_values: Optional[bool] = True
) -> bool:
    """
    Create observed synergies file for a specific cell line.
    
    Args:
        cell_line (str): The name of the cell line.
        synergies_exp_df (pandas.DataFrame): DataFrame containing synergy data.
        output_directory (PathLike): Directory to save the output file.
        column_cell_line_name (str): Column name for cell line names.
        column_synergy (str): Column name for synergy status.
        column_anchorID (str): Column name for anchor IDs.
        column_libraryID (str): Column name for library IDs.
        synergy_values (Optional[bool]): Whether to include synergy values in the output.

    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    # Make sure values in column_cell_line_name are uppercase and no spaces
    synergies_exp_df[column_cell_line_name] = synergies_exp_df[column_cell_line_name].str.upper().str.replace('-', '') 
    synergies_exp_df = filter_synergies(cell_line, synergies_exp_df, column_cell_line_name, column_synergy)
    if synergies_exp_df.empty:
        print(f"No synergies found for cell line {cell_line}.")
        return False

    cell_line_dir = create_cell_line_dir(cell_line, output_directory)
    
    output_file = get_output_file(cell_line_dir, f"{cell_line}_observed_synergies")
    synergy_arg = column_synergy if synergy_values else None
    _extract_synergies(
        output_file,
        synergies_exp_df,
        anchorID,
        libraryID,
        synergy_arg
    )

    print(f"Observed synergies file created for {cell_line} in output directory\n")
    return True