"""
Profile categories plotting: load -> process -> plot

This module process the full experimental drug panel to the pipeline drug profiles (PD profiles)
PD_profile categories are then plotted as parcat plots.
The plotting script consist of three steps:
- _load_profilecat_inputs(results_dir)
- _process_profilecat_inputs(drugpanel_input)  _process_profile_data(drugpanel_df)
- make_profilecat_plots(results_dir, plots_dir, show=False)

"""

import os
import json
import logging

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ..utils import save_fig
from .load_results import (_load_main_results,)

#//////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------
# LOAD
#---------------------------------------------------------------------

def _load_profilecat_inputs(results_dir):
    """Load only the input data for profile category plotting.

    Tries the per-tissue results_dir first; falls back to a sibling
    ``synco_shared/`` directory for both the experimental CSV and any JSON
    dictionaries (PD_mechanism_dict, PD_inhibitors_dict, …) that are not
    present in the per-tissue output.
    """
    results = _load_main_results(results_dir)
    experimental = results.get('files', {}).get('experimental')
    dicts = dict(results.get('dicts', {}) or {})

    shared_dir = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "synco_shared")

    # Fallback: experimental CSV
    if experimental is None:
        for fname in [
            "experimental_full_df.csv",
            "experimental_drug_names_synergies_df.csv",
            "experimental_matrix_df.csv",
            "experimental_window_df.csv",
        ]:
            cand = os.path.join(shared_dir, fname)
            if os.path.exists(cand):
                try:
                    experimental = pd.read_csv(cand)
                    logging.getLogger(__name__).info("Loaded experimental data from synco_shared: %s", cand)
                    break
                except Exception:
                    pass

    # Fallback: JSON dictionaries
    dict_filenames = {
        "PD_inhibitors_dict":      "PD_inhibitors_dict.json",
        "PD_mechanism_dict":       "PD_mechanism_dict.json",
        "mechanism_PD_dict":       "mechanism_PD_dict.json",
        "Drugnames_PD_dict":       "Drugnames_PD_dict.json",
        "PD_drugnames_dict":       "PD_drugnames_dict.json",
        "inhibitorgroups_dict":    "inhibitorgroups_dict.json",
    }
    for key, fname in dict_filenames.items():
        if not dicts.get(key):
            cand = os.path.join(shared_dir, fname)
            if os.path.exists(cand):
                try:
                    with open(cand, 'r', encoding='utf-8') as fh:
                        dicts[key] = json.load(fh)
                    logging.getLogger(__name__).info("Loaded %s from synco_shared", key)
                except Exception:
                    pass

    # Scope experimental data to the tissue-specific cell lines using predictions columns.
    if experimental is not None and 'cell_line' in experimental.columns:
        predictions_df = results.get('files', {}).get('predictions')
        if predictions_df is not None:
            _pred_meta = {
                'Perturbation', 'PD_A', 'PD_B', 'drug_name_A', 'drug_name_B',
                'node_targets_A', 'node_targets_B', 'drug_combination',
                'inhibitor_group_A', 'inhibitor_group_B', 'inhibitor_combination',
                'targets_A', 'targets_B', 'target_combination',
            }
            tissue_cell_lines = [c for c in predictions_df.columns if c not in _pred_meta]
            if tissue_cell_lines:
                experimental = experimental[experimental['cell_line'].isin(tissue_cell_lines)]

    drugpanel_input = {
        'files': {
            'experimental': experimental,
        },
        'dicts': dicts,
        }
    return drugpanel_input

#----------------------------------------------------------------------
# PREPARE
#----------------------------------------------------------------------

def _prepare_inputs(drugpanel_input):
    """Process the loaded input data for profile dimension plotting.
    """
    experimental_df = drugpanel_input['files']['experimental']
    if experimental_df is None:
        raise ValueError(
            "Experimental data not found in results_dir. "
            "Profile category plots require an experimental data file."
        )

    # List all drugs in the experimental data
    experimental_drugs = experimental_df[['drug_name_A','drug_name_B','PD_A','PD_B', ]].copy()
    drugA = experimental_drugs[['drug_name_A','PD_A']].rename(columns={'drug_name_A':'compound','PD_A':'PD'})
    drugB = experimental_drugs[['drug_name_B','PD_B']].rename(columns={'drug_name_B':'compound','PD_B':'PD'})
    # Combine and drop duplicates
    all_drugs = pd.concat([drugA, drugB], ignore_index=True).drop_duplicates().reset_index(drop=True)
    # Build categories data frame
    profilecat_df = all_drugs[['compound','PD']].copy()

    # Dictionary of categories (may contain either plain strings or dicts)
    pm = drugpanel_input['dicts'].get('PD_mechanism_dict', {}) or {}
    pi = drugpanel_input['dicts'].get('PD_inhibitors_dict', {}) or {}

    # Normalise mappings: support values that are either dicts or plain strings
    def _map_inhibitor(pd_key):
        if pd_key is None or pd_key is np.nan:
            return np.nan
        val = pi.get(pd_key)
        if isinstance(val, dict):
            return val.get('InhibitorGroup') if 'InhibitorGroup' in val else np.nan
        if isinstance(val, str):
            return val
        return np.nan

    def _map_mechanism(pd_key):
        if pd_key is None or pd_key is np.nan:
            return np.nan
        val = pm.get(pd_key)
        if isinstance(val, dict):
            return val.get('Mechanism') if 'Mechanism' in val else np.nan
        if isinstance(val, str):
            return val
        return np.nan

    # Map categories onto the DataFrame
    profilecat_df['InhibitorGroup'] = profilecat_df['PD'].map(_map_inhibitor)
    profilecat_df['Mechanism'] = profilecat_df['PD'].map(_map_mechanism)

    # Log any unmapped PD entries for easier debugging
    unmapped = profilecat_df[profilecat_df['PD'].notna() & (profilecat_df['InhibitorGroup'].isna() | profilecat_df['Mechanism'].isna())]
    if not unmapped.empty:
        logging.debug('Unmapped PD entries (compound, PD):\n%s', unmapped[['compound','PD']].to_string(index=False))
    
    # Build combination category data
    combicat_df = experimental_df[['PD_A','PD_B', 'Perturbation', 'drug_combination', 'inhibitor_combination']].copy()
            # Map mechanism to PD ids and add mechanism combination
    combicat_df['Mechanism_A'] = combicat_df['PD_A'].map(_map_mechanism)
    combicat_df['Mechanism_B'] = combicat_df['PD_B'].map(_map_mechanism)
    combicat_df['PD_combination'] = combicat_df['PD_A'] + ' + ' + combicat_df['PD_B']
    if bool(pm):
        combicat_df['mech_combination'] = (
            combicat_df['Mechanism_A'].fillna('Unknown') + ' + ' + combicat_df['Mechanism_B'].fillna('Unknown')
        )
    else:
        # No mechanism dict available: use inhibitor combination as the style column
        combicat_df['mech_combination'] = combicat_df['inhibitor_combination']

    return profilecat_df, combicat_df

def _prepare_dimensions(profilecat_df, combicat_df):
    """Prepare the profile category dimensions for plotting.

    When mechanism data is available (Mechanism column populated), includes a
    Mechanism dimension in the drug-profile chart and a Mechanism Combination
    dimension in the combination chart.  When mechanism data is absent, falls
    back to Inhibitor-Group-only dimensions.
    """
    has_mechanism = not combicat_df['Mechanism_A'].isna().all()

    # Define the mechanism order array
    mechanism_order = sorted(profilecat_df['Mechanism'].dropna().unique())

    # Map each mechanism to its compounds, inhibitor groups, and PDs in the same order
    def ordered_unique(series, order_by):
        # Returns unique values of `series`, ordered by the order of `order_by`
        df = pd.DataFrame({'val': series, 'order': order_by})
        # Drop rows where either is nan
        df = df.dropna(subset=['val', 'order'])
        # Sort by order, then drop duplicates keeping the first occurrence
        return df.sort_values('order')['val'].drop_duplicates().tolist()

    compound_order = ordered_unique(profilecat_df['compound'], profilecat_df['Mechanism'])
    inhibitorgroup_order = ordered_unique(profilecat_df['InhibitorGroup'], profilecat_df['Mechanism'])
    PD_order = ordered_unique(profilecat_df['PD'], profilecat_df['Mechanism'])

    compound_dim = go.parcats.Dimension(
        values=profilecat_df['compound'],
        label='Compound',
        categoryorder='array',
        categoryarray=compound_order if has_mechanism else None
    )
    inhibitorgroup_dim = go.parcats.Dimension(
        values=profilecat_df['InhibitorGroup'],
        label='Inhibitor Group',
        categoryorder='array',
        categoryarray=inhibitorgroup_order if has_mechanism else None
    )
    PD_dim = go.parcats.Dimension(
        values=profilecat_df['PD'],
        label='In silico Profile',
        categoryorder='array',
        categoryarray=PD_order if has_mechanism else None
    )

    if has_mechanism:
        mechanism_dim = go.parcats.Dimension(
            values=profilecat_df['Mechanism'],
            label='Mechanism',
            categoryorder='array',
            categoryarray=mechanism_order,
        )
        prof_dimensions = [compound_dim, inhibitorgroup_dim, PD_dim, mechanism_dim]
    else:
        prof_dimensions = [compound_dim, inhibitorgroup_dim, PD_dim]

    # Order categories by mechanism combination array, and apply the same order to each dimension
    if has_mechanism:
        # Get unique mechanism combinations in the order they appear
        mechanismcombi_order = sorted(combicat_df['mech_combination'].dropna().unique(), key=lambda x: combicat_df['mech_combination'].dropna().tolist().index(x))

        # Map each mechanism combination to its drug and inhibitor combinations, ordered by the mechanism combination order
        # Use function ordered_unique to get the unique values of drug_combination and inhibitor_combination ordered by mech_combination
        drugcombi_order = ordered_unique(combicat_df['drug_combination'], combicat_df['mech_combination'])
        PD_combi_order = ordered_unique(combicat_df['PD_combination'], combicat_df['mech_combination'])
        inhibitorcombi_order = ordered_unique(combicat_df['inhibitor_combination'], combicat_df['mech_combination'])


        drugcombi_dim = go.parcats.Dimension(
            values=combicat_df['drug_combination'],
            label='Drug',
            categoryorder='array',
            categoryarray=drugcombi_order
        )
        PD_combi_dim = go.parcats.Dimension(
            values=combicat_df['PD_combination'],
            label='Profile',
            categoryorder='array',
            categoryarray=PD_combi_order
        )
        inhibitorcombi_dim = go.parcats.Dimension(
            values=combicat_df['inhibitor_combination'],
            label='Inhibitor',
            categoryorder='array',
            categoryarray=inhibitorcombi_order
        )
        mechanismcombi_dim = go.parcats.Dimension(
            values=combicat_df['mech_combination'],
            label='Mechanism',
            categoryorder='array',
            categoryarray=mechanismcombi_order
        )
        combi_dimensions = [drugcombi_dim, inhibitorcombi_dim, mechanismcombi_dim]
    else:
        drugcombi_dim = go.parcats.Dimension(
            values=combicat_df['drug_combination'],
            label='Drug',
        )
        inhibitorcombi_dim = go.parcats.Dimension(
            values=combicat_df['inhibitor_combination'],
            label='Inhibitor',
        )
        combi_dimensions = [drugcombi_dim, inhibitorcombi_dim]

    return prof_dimensions, combi_dimensions

def _style_dimensions(profilecat_df, combicat_df):
    """Style the profile category dimensions for plotting.

    Colors by Mechanism when mechanism data is available; falls back to
    Inhibitor Group coloring when no mechanism dict was loaded.
    """
    # Use new mecha_palette to style plot by mechanism
    mecha_palette = {
        "Apoptosis": "#FF97FF",
        "Cell cycle": "#636EFA", 
        "Not mapped": "#3C6152", 
        "DNA repair": "#F09138",
        "MAPK": "#BD7EF7", 
        "PI3K/AKT/MTOR": "#16B7D3",
        "RTK": "#FC7299",
        "WNT": "#71C715",
        "Other": "#FF6F61",
        "Chemotherapy": "#B1B1B1",
    }

    has_mechanism = not profilecat_df['Mechanism'].isna().all()

    # Profile chart: color by Mechanism if available, else by Inhibitor Group
    if has_mechanism:
        style_col = 'Mechanism'
        color_map = {val: mecha_palette.get(val, '#cccccc') for val in profilecat_df[style_col].dropna().unique()}
        line_prof_colors = profilecat_df[style_col].map(color_map).fillna('#cccccc')
    else:
        style_col = 'InhibitorGroup'
        unique_vals = profilecat_df[style_col].dropna().unique()
        default_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Vivid
        color_map = {val: default_colors[i % len(default_colors)] for i, val in enumerate(unique_vals)}
        line_prof_colors = profilecat_df[style_col].map(color_map).fillna('#cccccc')

    # Combination chart: color by unique mechanism combinations, not by inhibitors
    unique_combi = combicat_df['mech_combination'].dropna().unique()
    default_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Vivid
    combi_color_map = {val: default_colors[i % len(default_colors)] for i, val in enumerate(unique_combi)}
    line_combi_colors = combicat_df['mech_combination'].map(combi_color_map).fillna('#cccccc')

    return line_prof_colors, line_combi_colors

#----------------------------------------------------------------------
# PLOT
#----------------------------------------------------------------------
def plot_profile_categories(dimensions, line_colors, 
                            title_text='Drug Profiles Categories',
                            show=False):
    """Plot the profile category parcat plot.
    """
    fig = go.Figure(data=
        go.Parcats(
            dimensions=dimensions,
            line={'color': line_colors,},
            hoveron='color',
            hoverinfo='all',
            labelfont={'size': 14}
        )
    )

    fig.update_layout(
        title=title_text,
        height=1000,
        width=900,
        margin=dict(l=150, r=180, t=80, b=50),
        font=dict(size=14,),
    )

    if show:
        try:
            fig.show()
        except Exception as exc:
            logging.warning(
                "Could not display interactive figure (fig.show() failed): %s.\n"
                "If you're running in a Jupyter environment install/upgrade nbformat: `pip install \"nbformat>=4.2.0\"`\n"
                "Or call this function with `show=False` to skip interactive display.",
                exc,
            )
    return fig


#----------------------------------------------------------------------
# WRAPPER
#----------------------------------------------------------------------

def make_profilecat_plots(results_dir, plots_dir, show=False, debug=False, return_fig: bool = False):
    """Make profile category parcat plots.
    """
    # Load inputs
    drugpanel_input = _load_profilecat_inputs(results_dir)

    # Prepare data
    profilecat_df, combicat_df = _prepare_inputs(drugpanel_input)
    prof_dimensions, combi_dimensions = _prepare_dimensions(profilecat_df, combicat_df)
    line_prof_colors, line_combi_colors = _style_dimensions(profilecat_df, combicat_df)
    if debug:
        print("DEBUG: profilecat_df:\n", profilecat_df)
        print("DEBUG: combicat_df:\n", combicat_df)
    # Plot
    prof_fig = plot_profile_categories(prof_dimensions, line_prof_colors, show=show)
    combi_fig = plot_profile_categories(combi_dimensions, line_combi_colors, title_text='Combination Categories', show=show)

    if return_fig:
        return [(prof_fig, 'plotly'), (combi_fig, 'plotly')]

    # Save figure
    os.makedirs(plots_dir, exist_ok=True)
    save_fig(prof_fig, plots_dir, 'drug_profiles_categories', formats=['png', 'html'], fig_type='plotly')
    save_fig(combi_fig, plots_dir, 'combination_categories', formats=['png', 'html'], fig_type='plotly')
    logging.info(f'Profile categories parcat plot saved to: {plots_dir}')
    logging.info(f'Combination categories parcat plot saved to: {plots_dir}')
    return