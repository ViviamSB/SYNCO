# synco/config.py

"""
Configuration defaults for the SYNCO pipeline.

This module defines the BASE_DEFAULTS dictionary which contains default values
for all pipeline steps. These defaults are merged with user-provided configuration
in the main pipeline execution.

Configuration Structure:
    The user configuration is expected to have the following top-level sections:
    
    - paths: Directory paths for input, output, and pipeline runs
        - base: Base directory for the project
        - pipeline_runs: Directory containing prediction method output
        - input: Directory containing input files (drug profiles, synergies, etc.)
        - output: Directory for pipeline output files (can be None)
    
    - general: General pipeline settings
        - cell_lines: Cell line specification (flexible input options):
            * Manual list: ["AsPC-1", "BxPC-3", "CAPAN-1"] - explicit cell line names
            * CSV file: "cell_line_list.csv" - path to CSV with cell line names
            * Auto-discovery: [] or None - automatically detect from pipeline_runs folders
        - run_date: Optional specific run date to use
        - verbose: Enable verbose logging (boolean)
    
    - compare: Settings for synergy comparison (STEP 5)
        - prediction_method: Name of prediction method used (e.g., "DrugLogics", "BooLEVARD")
        - threshold: Synergy threshold value for classification (float)
        - synergy_column: Name of synergy score column (default: "synergy")
        - analysis_mode: Comparison mode (default: "inhibitor_combination")
        - duplicate_strategy: How to handle duplicates (default: "mean")
        - debug_items: Optional list of items to debug (default: None)
    
    - advance: (Optional) Advanced overrides for step-specific defaults
        - Any step from BASE_DEFAULTS can be overridden here
        - Values are deep-merged with BASE_DEFAULTS

Pipeline Steps:
    1. SETUP & FETCH (data_loading): Load experimental data and prediction results
    2. DRUG PROFILES: Process drug information and target mappings
    3. SYNERGY PREDICTIONS: Generate synergy predictions from raw data
    4. SYNERGY CONVERGENCE: Aggregate and converge synergy data
    5. SYNERGY COMPARISON: Compare experimental vs predicted synergies
    6. ROC & PR CALCULATIONS: Calculate performance metrics
"""

# VARIABLES FOR PIPELINE STEPS

# Default variables used in specific pipeline steps
BASE_DEFAULTS = {
    # STEP 1: SETUP & FETCH
    'data_loading': {
        # Path to cell line information (if None, uses paths.input from main config)
        "cell_info_path": None,
        
        # Whether to include experimental observations in the data loading step
        "experimental_observations": False,
        
        # Specific analysis folder to use (if None, auto-determined)
        "analysis_folder": None,
        
        # Pattern to locate the experimental synergies file in the input folder.
        # Default matches files named like 'synergies_observed*.csv'.
        # Supports glob patterns (e.g., "synergies_*.csv", "experiment_data.csv")
        "synergy_pattern": "synergies_observed*.csv",
        
        # Optional: exact filename or path to the synergies file.
        # If provided, this takes precedence over `synergy_pattern`.
        # Can be absolute path or relative to input folder.
        "synergy_filename": None,
        
        # Base name for inhibitor profiles file (without extension)
        # Looks for files like 'inhibitor_profiles.csv' in input folder
        "inhibitor_profiles": 'inhibitor_profiles',
        
        # Base name for drug information file (without extension)
        # Looks for files like 'drug_profiles.csv' in input folder
        "drug_info": 'drug_profiles',
    },

    # STEP 3: Synergy Predictions
    'synergy_predictions': {
        # Type of drug combination identifier to use
        # Options: "drugnames", "drug_ids", "targets"
        "combination_type": "drugnames",
        
        # Column name for drug ID in the data
        "drugID_type": "drug_name",
        
        # Type of target information to use
        # Options: "node_targets", "all_targets"
        "target_type": "node_targets",
        
        # Whether to remove duplicate drug combinations
        "remove_duplicates": False,
        
        # Whether to add experimental observations to predictions
        "add_experimental_observations": False,
    },

    # STEP 4: Converge Synergies
    'synergy_convergence': {
        # Column name for anchor drug identifier in source data
        "anchorID": "PD_A",
        
        # Column name for library drug identifier in source data
        "libraryID": "PD_B",
        
        # Column name for anchor drug name in processed data
        "anchor_name": "drug_name_A",
        
        # Column name for library drug name in processed data
        "library_name": "drug_name_B",
        
        # Column name containing synergy scores
        "synergy_column": "synergy",
        
        # Column name containing cell line identifiers
        "cell_line_column": "cell_line",
        
        # Whether the data being converged is predicted data (vs experimental)
        # This is automatically set appropriately during pipeline execution
        "predicted_data": False
    },
}