from ..utils import make_dictionary, save_file, load_dataframe
from typing import Optional

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN METHOD: get_drugprofiles()

#///////////////////////////////////////////////////////////////////////////////////////////////////////
def get_drugprofiles(
        input_path: str,
        output_path: Optional[str] = None,
        inhibitor: str = 'inhibitor_profiles',
        drug_info: str = 'drug_profiles'
):
    
    """
    Load drug profiles and make all mapping dictionaries.
    Save dictionaries in <output_path>

    Args:
        input_path (str): Path to the input file containing drug profiles.
        output_path (Optional[str]): Path to the output file to save mapping dictionaries.

    Returns:
        Six mapping dictionaries
    """
    inhibitor_df = load_dataframe(folder=input_path, pattern=inhibitor + "*.csv")
    drug_info_df = load_dataframe(folder=input_path, pattern=drug_info + "*.csv")

    # Inhibitor dictionaries
    PD_inhibitors_dict = make_dictionary(inhibitor_df, 'PD_profile', 'inhibitor_group')
    Drugnames_PD_dict = make_dictionary(inhibitor_df, 'drug_name', 'PD_profile', long='keys')
    PD_drugnames_dict = make_dictionary(inhibitor_df, 'PD_profile', 'drug_name')
    inhibitorgroups_dict = make_dictionary(inhibitor_df, 'inhibitor_group', 'drug_name',)
    Drugnames_inhibitor_dict = make_dictionary(inhibitor_df, 'drug_name', 'inhibitor_group', long='keys')

    # Drug info dictionaries
    PD_targets_dict = make_dictionary(drug_info_df, 'PD_profile', 'node_targets')

    # Save as json if output_path
    if output_path:
        save_file(PD_inhibitors_dict, output_path / 'PD_inhibitors_dict.json', file_type='json')
        save_file(Drugnames_PD_dict, output_path / 'Drugnames_PD_dict.json', file_type='json')
        save_file(PD_drugnames_dict, output_path / 'PD_drugnames_dict.json', file_type='json')
        save_file(inhibitorgroups_dict, output_path / 'inhibitorgroups_dict.json', file_type='json')
        save_file(Drugnames_inhibitor_dict, output_path / 'Drugnames_inhibitor_dict.json', file_type='json')
        save_file(PD_targets_dict, output_path / 'PD_targets_dict.json', file_type='json')

    # Collect all dictionaries
    drug_profiles_dictionaries = {
        "PD_inhibitors_dict": PD_inhibitors_dict,
        "Drugnames_PD_dict": Drugnames_PD_dict,
        "PD_drugnames_dict": PD_drugnames_dict,
        "inhibitorgroups_dict": inhibitorgroups_dict,
        "Drugnames_inhibitor_dict": Drugnames_inhibitor_dict,
        "PD_targets_dict": PD_targets_dict
    }

    return drug_profiles_dictionaries