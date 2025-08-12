import yaml
from pathlib import Path

# PROJECT FUNCTIONS
import utils
import data.loader
from data.loader import DataLoader
from data.drug_profiles import get_drugprofiles
from features.synergy_predictions import get_synergy_predictions
from features.converge_synergies import converge_synergies
from features.compare_synergies import compare_synergies

def main():
    # Load configuration / Read parameters
    ## TODO: implement 

    # MAIN SYNCO PIPELINE

    # STEP 1: Analysis folder & data loading
        # Set specific class parameters
    pipeline_results = DataLoader(
        base_path,
        cell_info_path,
        cell_line_list,
        pipeline,
        experimental_observations,
        run_results_path,
        analysis_folder,
        run_date,
        verbose,
    )
    pipeline_results.make_analysis_folders()
    synergy_data_dict = pipeline_results.fetch_synergy_data(experimental_observations)

    # STEP 2: Drug profiles
    drug_profiles = get_drugprofiles(input_path)

    # STEP 3: Synergy predictions
    synergy_results = get_synergy_predictions()
