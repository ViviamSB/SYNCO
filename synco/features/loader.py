import pandas
import os
import shutil
import sys
import fnmatch
from pathlib import Path
from datetime import datetime

# Try relative import first, fall back to absolute import for notebook compatibility
try:
    from ..utils import ensure_directory, copy_files, load_dataframe
except ImportError:
    # Add parent directory to path for notebook/standalone usage
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import ensure_directory, copy_files, load_dataframe

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# UTILITY FUNCTION: resolve_cell_lines
#///////////////////////////////////////////////////////////////////////////////////////////////////////

def resolve_cell_lines(
    cell_line_input,
    pipeline_runs_path: str = None,
    input_path: str = None,
    verbose: bool = False
) -> list:
    """
    Resolve cell line names from multiple input types.
    
    This function supports three input modes:
    1. Manual list: If cell_line_input is a non-empty list, return it as-is
    2. Automatic inference: If cell_line_input is empty list or None, infer from pipeline_runs_path
    3. CSV file: If cell_line_input is a string (filename/path), load from CSV file
    
    Args:
        cell_line_input: Can be:
            - list: Manual cell line names (returned as-is if non-empty)
            - str: Path/filename to CSV file with cell line names
            - None or []: Auto-discover from pipeline_runs_path
        pipeline_runs_path: Path to pipeline results directory for auto-discovery
        input_path: Path to input directory for CSV file lookup
        verbose: If True, print detailed information
    
    Returns:
        list: List of cell line names
        
    Raises:
        ValueError: If cell lines cannot be resolved from any source
    """
    # Case 1: Manual list provided (non-empty)
    if isinstance(cell_line_input, list) and len(cell_line_input) > 0:
        if verbose:
            print(f"Using manually specified cell lines: {cell_line_input}")
        return cell_line_input
    
    # Case 2: String provided - treat as CSV filename/path
    if isinstance(cell_line_input, str):
        # Determine the full path to the CSV file
        csv_path = Path(cell_line_input)
        
        # If not absolute, look in input_path
        if not csv_path.is_absolute() and input_path:
            csv_path = Path(input_path) / cell_line_input
        
        if not csv_path.exists():
            raise ValueError(
                f"Cell line CSV file not found: {csv_path}. "
                f"Please provide a valid path or place the file in the input directory."
            )
        
        try:
            # Load CSV and extract cell line names
            df = pandas.read_csv(csv_path)
            
            # Look for common column names for cell line identifiers
            possible_columns = ['cell_line_name', 'cell_line', 'cellline', 'name', 'Cell_Line', 'CellLine']
            cell_line_column = None
            
            for col in possible_columns:
                if col in df.columns:
                    cell_line_column = col
                    break
            
            if cell_line_column is None:
                # Use the first column if no standard column name found
                cell_line_column = df.columns[0]
                if verbose:
                    print(f"No standard cell line column found. Using first column: {cell_line_column}")
            
            cell_lines = df[cell_line_column].dropna().astype(str).tolist()
            
            if len(cell_lines) == 0:
                raise ValueError(f"No cell lines found in CSV file: {csv_path}")
            
            if verbose:
                print(f"Loaded {len(cell_lines)} cell lines from CSV: {csv_path}")
                print(f"Cell lines: {cell_lines}")
            
            return cell_lines
            
        except Exception as e:
            raise ValueError(f"Error reading cell line CSV file {csv_path}: {e}") from e
    
    # Case 3: Empty list or None - auto-discover from pipeline_runs_path
    if pipeline_runs_path and os.path.exists(pipeline_runs_path):
        cell_lines = []
        
        # Get all directories in pipeline_runs_path
        for entry in os.listdir(pipeline_runs_path):
            entry_path = os.path.join(pipeline_runs_path, entry)
            if os.path.isdir(entry_path):
                # Skip common non-cell-line directories
                if entry.lower() not in ['drabme_out', 'results', 'analysis', 'logs', 'config']:
                    cell_lines.append(entry)
        
        # Also check for drabme_out subfolder (common in DrugLogics pipelines)
        drabme_path = os.path.join(pipeline_runs_path, 'drabme_out')
        if os.path.exists(drabme_path):
            for entry in os.listdir(drabme_path):
                entry_path = os.path.join(drabme_path, entry)
                if os.path.isdir(entry_path) and entry not in cell_lines:
                    cell_lines.append(entry)
        
        if len(cell_lines) > 0:
            if verbose:
                print(f"Auto-discovered {len(cell_lines)} cell lines from: {pipeline_runs_path}")
                print(f"Cell lines: {cell_lines}")
            return cell_lines
    
    # If we get here, we couldn't resolve cell lines
    raise ValueError(
        "Could not resolve cell lines. Please provide one of:\n"
        "  1. A manual list in config['general']['cell_lines']\n"
        "  2. A CSV filename/path in config['general']['cell_lines']\n"
        "  3. Valid pipeline_runs path for auto-discovery\n"
        f"Current input: {cell_line_input}, pipeline_runs_path: {pipeline_runs_path}"
    )

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# CLASS: InsilicoDataLoader
# MAIN METHODS: make_analysis_folders() fetch_synergy_data()

#///////////////////////////////////////////////////////////////////////////////////////////////////////
class DataLoader:
    """
    InsilicoDataLoader is a class for loading and processing in silico synergy data from the DrugLogics (or BooLEVARD) pipeline.

        This class covers the transfer of results files from the results folder to the analysis folder.
        Copying files from the pipeline output to the analysis folder.

    Args:

    Methods:
        make_analysis_folders: Create a main analysis folder and cell line sub-folders with the results files.
        load_predictions: Load the observed and ensemble-wise synergies files into dataframes for all cell lines.
        
    """

    def __init__(self,
                base_path: str = None, # Path to the base folder.
                cell_info_path: str = None, # Path to the cell line information folder. Contains sub-folders for each cell line.
                run_results_path: str = None, # Path to the pipeline results folder. Contains sub-folders for each cell line.
                prediction_method: str = 'DrugLogics', # Name of the pipeline. DrugLogics or BooLEVARD.
                experimental_observations: bool = False, # If True, load experimental observations.
                cell_line_list = None, # List of cell lines, CSV filename, or None for auto-discovery.
                run_date: str = None, # Specific date run folder.
                analysis_folder: str = None, # Name of the analysis folder to create.
                verbose: bool = False, # Verbose mode.
                ):
        
        self.verbose = verbose
        self.base_path = base_path
        self.cell_line_info_path = cell_info_path
        self.run_results_path = run_results_path
        self.pipeline = prediction_method
        self.experimental_observations = experimental_observations
        # Use new resolve_cell_lines function to handle all input types
        self.cell_lines = resolve_cell_lines(
            cell_line_input=cell_line_list,
            pipeline_runs_path=run_results_path,
            input_path=cell_info_path,
            verbose=verbose
        )
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

    def _resolve_cell_line_source(self, cell_line: str) -> str:
        """
        Resolve the most likely source directory for a given cell line by trying several
        fallback locations and performing case-insensitive / normalized matching.

        Returns:
            str or None: path to the source directory if found, else None
        """
        # Primary expected location
        candidate = os.path.join(self.run_results_path, cell_line)
        if os.path.exists(candidate):
            return candidate

        # Common nested folder used by some pipelines
        nested = os.path.join(self.run_results_path, 'drabme_out', cell_line)
        if os.path.exists(nested):
            return nested

        # Search top-level children and drabme_out children with tolerant matching
        def normalize(name: str) -> str:
            return name.replace('-', '').replace('_', '').lower()

        # gather candidates
        candidates = []
        if os.path.exists(self.run_results_path):
            for entry in os.listdir(self.run_results_path):
                full = os.path.join(self.run_results_path, entry)
                if os.path.isdir(full):
                    candidates.append(full)
                    # also consider nested drabme_out if present
                    if entry == 'drabme_out':
                        for sub in os.listdir(full):
                            subfull = os.path.join(full, sub)
                            if os.path.isdir(subfull):
                                candidates.append(subfull)

        # Try to find a match by normalized names
        target_norm = normalize(cell_line)
        for c in candidates:
            if normalize(os.path.basename(c)) == target_norm:
                return c

        # Case-insensitive exact match fallback
        for c in candidates:
            if os.path.basename(c).lower() == cell_line.lower():
                return c

        return None
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _copy_pipeline_files(self, source_dir: str, destination_dir: str, cell_line: str, run_date: str = None) -> int:
        """
        Copy files from the pipeline output to the analysis folder.

        Args:
            source_dir (str): Source directory to copy files from.
            destination_dir (str): Destination directory to copy files to.
            cell_line (str): The name of the cell line.
            run_date (str): Optional run date to filter subdirectories.

        Returns:
            int: Number of files copied.
        """
        if self.pipeline not in ['DrugLogics', 'BooLEVARD']:
            raise ValueError("Unsupported pipeline. Supported pipelines are: 'DrugLogics' and 'BooLEVARD'.")
        
        # Check if source directory exists
        if not os.path.exists(source_dir):
            if self.verbose:
                print(f"Source directory does not exist: {source_dir}")
            return 0
            
        if self.pipeline == 'DrugLogics':
            file_patterns = ['*ensemblewise_synergies*']
        elif self.pipeline == 'BooLEVARD':
            file_patterns = [f'SynergyExcess_{cell_line}.tsv']
        if self.experimental_observations:
            file_patterns.append('*observed_synergies*')

        # Use the enhanced copy_files function with run_date filtering
        files_copied = copy_files(
            sources=[source_dir],
            destination_dir=destination_dir,
            file_patterns=file_patterns,
            overwrite=True,
            verbose=self.verbose,
            run_date_filter=run_date if self.pipeline == 'DrugLogics' else None
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
            # For DrugLogics, copy from the cell line directory which contains both 
            # observed_synergies and run subfolders
            # Resolve the most likely source directory for this cell line
            cell_line_source_path = self._resolve_cell_line_source(cell_line)
            if cell_line_source_path and os.path.exists(cell_line_source_path):
                files_copied = self._copy_pipeline_files(
                    cell_line_source_path,
                    cell_line_folder,
                    cell_line,
                    self.run_date
                )
            else:
                if self.verbose:
                    print(f"No cell line folder found for {cell_line} in {self.run_results_path}")
                files_copied = 0
                
        elif self.pipeline == 'BooLEVARD':
            # For BooLEVARD, look for the specific cell line subfolder in run_results_path
            cell_line_subfolder_path = os.path.join(self.run_results_path, cell_line)
            if os.path.exists(cell_line_subfolder_path):
                # Copy files from the BooLEVARD results folder to the cell line folder
                # BooLEVARD doesn't use run_date filtering
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
# Main method: make_analysis_folders
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def make_analysis_folders(self) -> None:
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

    def _load_observed_synergies(self, cell_line_path: str) -> pandas.DataFrame:
        """
        Load the observed synergies file for a specific cell line.
        Returns:
            pandas.DataFrame: A DataFrame containing the observed synergies data, or None if not found.
        """
        file_pattern = '*observed_synergies*'
        try:
            observed_synergies_df = load_dataframe(cell_line_path, file_pattern, sep='\t')
            observed_synergies_df['Perturbation'] = observed_synergies_df['Perturbation'].str.replace('~', '-')
            return observed_synergies_df
        except FileNotFoundError:
            if self.verbose:
                print(f"No observed synergies file found in {cell_line_path}")
            return None
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def _load_ensemble_synergies(self, cell_line_path: str, cell_line_name: str) -> pandas.DataFrame:
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
# Main method: fetch_synergy_data
#///////////////////////////////////////////////////////////////////////////////////////////////////////
    def fetch_synergy_data(self, analysis_folder: str = None, experimental_observations: bool = False) -> dict:
        """
        Load the observed and ensemble-wise synergies files into dataframes for all cell lines.
        Returns:
            dict: A dictionary with cell line names as keys and DataFrames as values.
        """
        synergy_data_dict = {}

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
                    synergy_data_dict[cell_line] = (None, None)
                continue

            # Load observed synergies if requested
            if experimental_observations:
                observed_synergies_df = self._load_observed_synergies(cell_line_path)
            else:
                observed_synergies_df = None
            # Load ensemble synergies
            ensemble_synergies_df = self._load_ensemble_synergies(cell_line_path, cell_line)

            # Store the results in the dictionary.
            synergy_data_dict[cell_line] = (observed_synergies_df, ensemble_synergies_df)
            
        # Store the dictionary in the class.
        self.synergy_results_dict = synergy_data_dict
        return synergy_data_dict
