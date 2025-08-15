# synco/pipeline_defaults.py

# VARIABLES FOR PIPELINE STEPS

# Defaults variables used in specific pipeline steps

BASE_DEFAULTS = {
    # STEP 1: SETUP & FETCH
    'data_loading': {
        "cell_info_path": None,
        "experimental_observations": False,
        "analysis_folder": None,
        "inhibitor_profiles": 'inhibitor_profiles',
        "drug_info": 'drug_profiles',
    },

    # STEP 3: Synergy Predictions
    'synergy_predictions': {
        "combination_type": "drugnames",
        "drugID_type": "drug_name",
        "target_type": "node_targets",
        "remove_duplicates": False,
        "add_experimental_observations": False,
    },

    # STEP 4: Converge Synergies
    'synergy_convergence': {
        "anchorID": "PD_A",
        "libraryID": "PD_B",
        "anchor_name": "drug_name_A",
        "library_name": "drug_name_B",
        "synergy_column": "synergy",
        "cell_line_column": "cell_line",
        "predicted_data": False
    },

    # # STEP 5: Compare Synergies
    # 'compare_synergies': {
    #     "threshold": 0.0,
    #     "analysis_mode": "inhibitor_combination",
    #     "synergy_column": "synergy",
    # }
}