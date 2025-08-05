import pandas
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Try relative import first, fall back to absolute import for notebook compatibility
try:
    from ..utils import ensure_directory, copy_files, load_dataframe
except ImportError:
    # Add parent directory to path for notebook/standalone usage
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import ensure_directory, copy_files, load_dataframe


class InsilicoDataLoader:
    """
    InsilicoDataLoader is a class for loading and processing in silico synergy data from the DrugLogics (or BooLEVARD) pipeline.

        This class covers the transfer of results files from the results folder to the analysis folder.
        Copying files from the pipeline output to the analysis folder.

    Args:

    Methods:
        get_analysis_folders: Create a main analysis folder and cell line sub-folders with the results files.
        load_predictions: Load the observed and ensemble-wise synergies files into dataframes for all cell lines.
        
    """

    def __init__(self,
                base_path: str = None, # Path to the base folder.
                cell_info_path: str = None, # Path to the cell line information folder.
                run_results_path: str = None, # Path to the pipeline results folder.
                pipeline: str = 'DrugLogics', # Name of the pipeline. DrugLogics or BooLEVARD.
                cell_lines: list = None, # List of cell lines.
                run_date: str = None, # Specific date run folder.
                analysis_folder: str = None, # Name of the analysis folder to create.
                verbose: bool = False, # Verbose mode.
                ):
        
        self.verbose = verbose
        self.base_path = base_path
        self.cell_line_info_path = cell_info_path
        self.run_results_path = run_results_path
        self.pipeline = pipeline
        self.cell_lines = cell_lines if cell_lines else self._discover_cell_lines()
        self.run_date = run_date
        

        # Analysis folder paths.
        if base_path:
            self.analysis_path = os.path.join(self.base_path, 'run_analysis')
            analysis_folder_name = analysis_folder if analysis_folder else f"results_{datetime.now().strftime('%Y%m%d')}"
            self.analysis_folder_path = os.path.join(self.analysis_path, analysis_folder_name)

        # Tracking statistics.
        self.total_folders_created = 0
        self.folders_with_no_files_copied = 0
        self.empty_folders = []

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _discover_cell_lines(self):
        """
        Discover cell line names based on sub-folders.
        Returns:
            list: List of cell line names.
        """
        cell_line_folders = []
        if self.cell_line_info_path and os.path.exists(self.cell_line_info_path):
            cell_line_folders = [
                folder for folder in os.listdir(self.cell_line_info_path)
                if os.path.isdir(os.path.join(self.cell_line_info_path, folder))
            ]        
            if self.verbose:
                print(f"\nDiscovered cell lines from {self.cell_line_info_path}")
                print(f"Cell lines: {cell_line_folders}")

        else:
            if self.verbose:
                print("\nCell line list and info path not found.")
                print("Please provide a list of cell lines or a cell line info path to discover the cell line names.")
        return cell_line_folders 

#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _create_main_analysis_folder(self) -> Path:
        """
        Create the main analysis folder and sub-folders. Resets if it already exists.
        Returns:
            Path: The path object of the main analysis folder.
        """
        # Try to reset, but if it fails due to permissions, just ensure the directory exists
        try:
            return ensure_directory(self.analysis_folder_path, reset=True)
        except PermissionError:
            if self.verbose:
                print(f"Warning: Could not reset existing folder due to permissions. Using existing folder: {self.analysis_folder_path}")
            return ensure_directory(self.analysis_folder_path, reset=False)
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _get_latest_run_folder(self, cell_line: str) -> None:
        """
        Get the latest run folder for a specific cell line based on the run date.
        Args:
            cell_line (str): The name of the cell line.
        Returns:
            Path: The path object of the latest run folder.
        """
        if self.verbose:
            print(f"\nFinding latest run folder for cell line: {cell_line}")
            print(f"Run date: {self.run_date}")
        
        cell_line_path = os.path.join(self.run_results_path, cell_line)
        if os.path.exists(cell_line_path):
            if self.run_date:
                # Filter by run date if specified
                run_folders = [
                    folder for folder in os.listdir(cell_line_path)
                    if os.path.isdir(os.path.join(cell_line_path, folder)) and (self.run_date in folder)
                ]
            else:
                # Get all run folders if no specific date is provided
                run_folders = [
                    folder for folder in os.listdir(cell_line_path)
                    if os.path.isdir(os.path.join(cell_line_path, folder))
                ]
            if run_folders:
                latest_run_folder = max(run_folders, key=lambda x: os.path.getmtime(os.path.join(cell_line_path, x)))
                if self.verbose:
                    if self.run_date:
                        print(f"Latest run folder for cell line {cell_line} with date {self.run_date}: {latest_run_folder}")
                    else:
                        print(f"Latest run folder for cell line {cell_line}: {latest_run_folder}")
                return Path(os.path.join(cell_line_path, latest_run_folder))
            else:
                if self.verbose:
                    if self.run_date:
                        print(f"No run folders found for cell line {cell_line} with date {self.run_date}.")
                    else:
                        print(f"No run folders found for cell line {cell_line}.")
        return None
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _copy_pipeline_files(self, source_dir: str, destination_dir: str, cell_line: str) -> int:
        """
        Copy files from the pipeline output to the analysis folder.

        Args:
            file_paths (list): List of file paths to copy.
            analysis_folder (str): The analysis folder path.
            patterns (list): List of file patterns to filter files.
            overwrite (bool): If True, overwrite existing files in the destination.

        Returns:
            int: Number of files copied.
        """
        if self.pipeline not in ['DrugLogics', 'BooLEVARD']:
            raise ValueError("Unsupported pipeline. Supported pipelines are: 'DrugLogics' and 'BooLEVARD'.")
        if self.pipeline == 'DrugLogics':
            file_patterns = ['*ensemblewise_synergies*', '*observed_synergy*']
        elif self.pipeline == 'BooLEVARD':
            file_patterns = [f'SynergyExcess_{cell_line}.tsv']

        # Use the utils copy_files function to copy files
        files_copied = copy_files(
            sources=[source_dir],
            destination_dir=destination_dir,
            file_patterns=file_patterns,
            overwrite=True,
            verbose=self.verbose
        )
        if files_copied == 0:
            if self.verbose:
                print(f"No files copied for cell line {cell_line} from {source_dir}.")
        return files_copied
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _find_results_files(self, cell_line: str) -> None:
        """
        Find results files for a specific cell line.
        Args:
            cell_line (str): The name of the cell line.
        """
        # Create a sub-folder for the cell line in the analysis folder
        cell_line_folder = os.path.join(self.analysis_folder_path, cell_line)
        
        # Try to reset, but if it fails due to permissions, just ensure the directory exists
        try:
            ensure_directory(cell_line_folder, reset=True)
        except PermissionError:
            if self.verbose:
                print(f"Warning: Could not reset existing folder due to permissions. Using existing folder: {cell_line_folder}")
            ensure_directory(cell_line_folder, reset=False)
        
        if self.pipeline == 'DrugLogics':
            latest_run_folder = self._get_latest_run_folder(cell_line)
            if latest_run_folder is None:
                if self.verbose:
                    print("Skipping file copy.")
                files_copied = 0
            else:
                # Copy files from the latest run folder to the cell line folder
                files_copied = self._copy_pipeline_files(latest_run_folder, cell_line_folder, cell_line)
            
        elif self.pipeline == 'BooLEVARD':
            # For BooLEVARD, look for the specific cell line subfolder in run_results_path
            cell_line_subfolder_path = os.path.join(self.run_results_path, cell_line)
            if os.path.exists(cell_line_subfolder_path):
                # Copy files from the BooLEVARD results folder to the cell line folder
                files_copied = self._copy_pipeline_files(cell_line_subfolder_path, cell_line_folder, cell_line)
            else:
                if self.verbose:
                    print(f"No subfolder found for cell line {cell_line} in {self.run_results_path}")
                files_copied = 0

        # Count the number of files copied   
        if files_copied > 0:
            self.total_folders_created += 1
        else:
            self.folders_with_no_files_copied += 1
            self.empty_folders.append(cell_line_folder)
            if self.verbose:
                print(f"No files copied for cell line {cell_line}. Folder created but empty: {cell_line_folder}")

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# Main method: get_analysis_folders
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def get_analysis_folders(self) -> None:
        """
        Create the main analysis folder and sub-folders for each cell line with the results files.
        """
        self._create_main_analysis_folder()
        if self.verbose:
            print(f"\nMain analysis folder created at: {self.analysis_folder_path}")

        for cell_line in self.cell_lines:
            self._find_results_files(cell_line)

        if self.verbose:
            print(f"\nTotal folders created: {self.total_folders_created}")
            print(f"Folders with no files copied: {self.folders_with_no_files_copied}")
            if self.empty_folders:
                print(f"Empty folders: {self.empty_folders}")

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////////////////////////////////

    def load_observed_synergies(self, cell_line_path: str) -> pandas.DataFrame:
        """
        Load the observed synergies file for a specific cell line.
        Returns:
            pandas.DataFrame: A DataFrame containing the observed synergies data, or None if not found.
        """
        file_pattern = '*observed_synergies*'
        try:
            observed_synergies_df = load_dataframe(cell_line_path, file_pattern, header=None, names=['Perturbation'])
            observed_synergies_df = observed_synergies_df.iloc[1:]
            observed_synergies_df['Perturbation'] = observed_synergies_df['Perturbation'].str.replace('~', '-')
            return observed_synergies_df
        except FileNotFoundError:
            if self.verbose:
                print(f"No observed synergies file found in {cell_line_path}")
            return None
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def load_ensemble_synergies(self, cell_line_path: str, cell_line_name: str) -> pandas.DataFrame:
        """
        Load the ensemble synergies file for a specific cell line.
        Returns:
            pandas.DataFrame: A DataFrame containing the ensemble synergies data, or None if not found.
        """
        if self.pipeline == 'DrugLogics':
            file_pattern = '*ensemblewise_synergies*'
        elif self.pipeline == 'BooLEVARD':
            file_pattern = f'SynergyExcess_{cell_line_name}.tsv'

        try:
            ensemble_predictions_df = load_dataframe(cell_line_path, file_pattern, sep='\t')

            if self.pipeline == 'DrugLogics':
                ensemble_predictions_df['Perturbation'] = ensemble_predictions_df['Perturbation'].str.replace('\\[', '', regex=True).str.replace('\\]', '', regex=True)
                ensemble_predictions_df = ensemble_predictions_df.rename(columns={'Response excess over subset': cell_line_name})
            elif self.pipeline == 'BooLEVARD':
                ensemble_predictions_df['Perturbation'] = ensemble_predictions_df['Perturbation'].apply(lambda x: '-'.join(['_'.join(x.split('_')[0:2]), '_'.join(x.split('_')[2:4])]))
                ensemble_predictions_df = ensemble_predictions_df.drop(columns=['Emax(k)', 'Bliss(k)'], errors='ignore')
                ensemble_predictions_df = ensemble_predictions_df.rename(columns={'Excess': cell_line_name})
            return ensemble_predictions_df
        except FileNotFoundError:
            if self.verbose:
                print(f"No ensemble synergies file found in {cell_line_path}")
            return None
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////
# Main method: load_predictions
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def load_predictions(self, analysis_folder: str = None, experimental_observations: bool = False) -> dict:
        """
        Load the observed and ensemble-wise synergies files into dataframes for all cell lines.
        Returns:
            dict: A dictionary with cell line names as keys and DataFrames as values.
        """
        synergy_results_dict = {}

        # Use the provided analysis folder or the default one for the class
        analysis_folder = analysis_folder if analysis_folder else self.analysis_folder_path
        if not os.path.exists(analysis_folder):
            # Take the latest created analysis folder
            analysis_folder = max(os.listdir(self.analysis_path), key=lambda x: os.path.getmtime(os.path.join(self.analysis_path, x)))
            analysis_folder = os.path.join(self.analysis_path, analysis_folder)

        for cell_line in self.cell_lines:
            cell_line_path = os.path.join(analysis_folder, cell_line)
            if not os.path.exists(cell_line_path):
                if self.verbose:
                    print(f"Sub-folder for cell line {cell_line} does not exist in {analysis_folder}. Skipping.")
                    synergy_results_dict[cell_line] = (None, None)
                continue

            # Load observed synergies if requested
            if experimental_observations:
                observed_synergies_df = self.load_observed_synergies(cell_line_path)
            else:
                observed_synergies_df = None
            # Load ensemble synergies
            ensemble_synergies_df = self.load_ensemble_synergies(cell_line_path, cell_line)

            # Store the results in the dictionary.
            synergy_results_dict[cell_line] = (observed_synergies_df, ensemble_synergies_df)
            
        # Store the dictionary in the class.
        self.synergy_results_dict = synergy_results_dict
        return synergy_results_dict


