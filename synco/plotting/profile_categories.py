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
    """
    results = _load_main_results(results_dir)
    drugpanel_input = {
        'files': {
            'experimental': results.get('files', {}).get('experimental'),
        },
        'dicts': results.get('dicts', {}),
        }
    return drugpanel_input

#----------------------------------------------------------------------
# PREPARE
#----------------------------------------------------------------------

def _prepare_inputs(drugpanel_input):
    """Process the loaded input data for profile dimension plotting.
    """
    experimental_df = drugpanel_input['files']['experimental']

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
    combicat_df['mech_combination'] = combicat_df['Mechanism_A'] + ' + ' + combicat_df['Mechanism_B']

    return profilecat_df, combicat_df

def _prepare_dimensions(profilecat_df, combicat_df):
    """Prepare the profile category dimensions for plotting.
    """
    compound_dim = go.parcats.Dimension(
        values=profilecat_df['compound'],
        label='Compound',
    )

    inhibitorgroup_dim = go.parcats.Dimension(
        values=profilecat_df['InhibitorGroup'],
        label='Inhibitor Group',
    )
    mechanism_dim = go.parcats.Dimension(
        values=profilecat_df['Mechanism'],
        label='Mechanism',
    )
    PD_dim = go.parcats.Dimension(
        values=profilecat_df['PD'],
        label='PD',
    )
    prof_dimensions = [compound_dim, inhibitorgroup_dim, PD_dim, mechanism_dim]

    drugcombi_dim = go.parcats.Dimension(
        values=combicat_df['drug_combination'],
        label='Drug Combination',
    )
    mechanismcombi_dim = go.parcats.Dimension(
        values=combicat_df['mech_combination'],
        label='Mechanism Combination',
    )
    PD_dim = go.parcats.Dimension(
        values=combicat_df['PD_combination'],
        label='PD Combination',
    )
    inhibitorcombi_dim = go.parcats.Dimension(
        values=combicat_df['inhibitor_combination'],
        label='Inhibitor Combination',
    )

    combi_dimensions = [drugcombi_dim, inhibitorcombi_dim, PD_dim, mechanismcombi_dim]

    return prof_dimensions, combi_dimensions

def _style_dimensions(profilecat_df, combicat_df):
    """Style the profile category dimensions for plotting.
    """
    # Color by Mechanism
    mechanism_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Vivid
    unique_mechanisms = profilecat_df['Mechanism'].unique()
    color_map = {mech: mechanism_colors[i % len(mechanism_colors)] for i, mech in enumerate(unique_mechanisms)}
    line_prof_colors = profilecat_df['Mechanism'].map(color_map)

    # Color by Mechanism Combination
    unique_combi_mechanisms = combicat_df['mech_combination'].unique()
    combi_color_map = {mech: mechanism_colors[i % len(mechanism_colors)] for i, mech in enumerate(unique_combi_mechanisms)}
    line_combi_colors = combicat_df['mech_combination'].map(combi_color_map)

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

def make_profilecat_plots(results_dir, plots_dir, show=False, debug=False):
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

    # Save figure
    os.makedirs(plots_dir, exist_ok=True)
    save_fig(prof_fig, plots_dir, 'drug_profiles_categories', formats=['png', 'html'], fig_type='plotly')
    save_fig(combi_fig, plots_dir, 'combination_categories', formats=['png', 'html'], fig_type='plotly')
    logging.info(f'Profile categories parcat plot saved to: {plots_dir}')
    logging.info(f'Combination categories parcat plot saved to: {plots_dir}')
    return