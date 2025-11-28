"""
Experimental distribution plotting: load -> process -> plot

This module process the full experimental synergy distribution data by cell line and combinations.
Experimental distributions are then plotted as (1) scatter plots, (2) bar plots.
The plotting script consist of three steps:
- _load_experimental_inputs(results_dir)
- _process_experimental_inputs(results)  _process_experimental_data()
- make_experimental_distribution_plots(results_dir, plots_dir, show=False)

"""

import os
import logging

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import save_fig
from .load_results import (_load_main_results,)

#//////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------
# LOAD
#---------------------------------------------------------------------

def _load_experimental_inputs(results_dir):
    """Load only the input data for experimental distribution plotting.
    """
    results = _load_main_results(results_dir)
    experimental_input = {
        'files': {
            'experimental': results.get('files', {}).get('experimental'),
        },
        'dicts': results.get('dicts', {}),
        }
    return experimental_input

#----------------------------------------------------------------------
# PREPARE
#----------------------------------------------------------------------

def _prepare_experimental_counts(experimental_input, threshold=0):
    """Process the loaded input data for experimental distribution plotting.
    """
    experimental_df = experimental_input['files']['experimental']

    pm = experimental_input['dicts'].get('PD_mechanism_dict', {}) or {}

    # Normalize mapping of mechanisms
    def _map_mechanism(pd_key):
        if pd_key is None or pd_key is np.nan:
            return np.nan
        val = pm.get(pd_key)
        if isinstance(val, dict):
            return val.get('Mechanism') if 'Mechanism' in val else np.nan
        if isinstance(val, str):
            return val
        return np.nan
    
    # Map mechanisms onto the DataFrame
    experimental_df['Mechanism_A'] = experimental_df['PD_A'].map(_map_mechanism)
    experimental_df['Mechanism_B'] = experimental_df['PD_B'].map(_map_mechanism)
    experimental_df['mech_combination'] = experimental_df['Mechanism_A'] + " + " + experimental_df['Mechanism_B']
    mech_combi_list = experimental_df['mech_combination'].unique().tolist()
    # Count synergies occurrences
    mechanism_matrix = experimental_df.copy()
    mechanism_matrix['synergy_binary'] = mechanism_matrix['synergy'].apply(lambda x: 1 if x >= threshold else 0)
    
    mechanism_matrix['n_synergies_per_drugcombi'] = mechanism_matrix.groupby(
        ['drug_name_A', 'drug_name_B'])['synergy_binary'].transform('sum').round(0).astype(int)
    
    mechanism_matrix['n_synergies_per_mechanism'] = mechanism_matrix.groupby(
        ['Mechanism_A', 'Mechanism_B'])['synergy_binary'].transform('sum').round(0).astype(int)
    
    mechanism_matrix['n_synergies_per_inhibitor'] = mechanism_matrix.groupby(
        ['inhibitor_group_A', 'inhibitor_group_B'])['synergy_binary'].transform('sum').round(0).astype(int)
    
    mechanism_matrix['total_combinations_per_mechanism'] = mechanism_matrix.groupby(
        ['Mechanism_A', 'Mechanism_B'])['synergy_binary'].transform('count').round(0).astype(int)
    
    mechanism_matrix = mechanism_matrix[['inhibitor_combination', 'n_synergies_per_inhibitor', 'mech_combination']].drop_duplicates().reset_index(drop=True)
    mechanism_matrix['total_combinations_per_inhibitor'] = mechanism_matrix.groupby(
        ['inhibitor_combination'])['synergy_binary'].transform('count').round(0).astype(int)
    mechanism_matrix['percentage'] = mechanism_matrix['n_synergies_per_inhibitor'] / mechanism_matrix['total_combinations_per_inhibitor'] * 100

    inhibitor_synergy_summary = mechanism_matrix.groupby('inhibitor_combination').agg({
        'n_synergies_per_inhibitor': 'first',
        'total_combinations_per_inhibitor': 'first',
        'percentage': 'first',
        'mech_combination': 'first',
    }).reset_index()
    inhibitor_synergy_summary = inhibitor_synergy_summary.sort_values(by='percentage', ascending=True).reset_index(drop=True)

    combi_order = inhibitor_synergy_summary['inhibitor_combination'].tolist()
    synergy_counts = inhibitor_synergy_summary.set_index('inhibitor_combination').reindex(combi_order)


    return inhibitor_synergy_summary, synergy_counts, mech_combi_list

def _style_mechanism_colors(mech_combi_list):
    """Define colors for each mechanism combination.
    """
    base_colors = px.colors.qualitative.Plotly
    mechanism_combi_color = {}
    for i, mech_combi in enumerate(mech_combi_list):
        mechanism_combi_color[mech_combi] = base_colors[i % len(base_colors)]
    return mechanism_combi_color

#----------------------------------------------------------------------
# PLOT
#----------------------------------------------------------------------

def _plot_stackedbars_synergy_counts(synergy_counts, inhibitor_synergy_summary, mechanism_combi_color, show=False):
    """Plot two stacked bar plots of synergy counts by inhibitor combination and by cell line.
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.02,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=('Tested combinations:Synergistic vs Non-synergistic', 'Synergy across cell lines'),
        shared_yaxes=True
    )

    # STAKED BAR inhibitor combination (syn vs- non-syn)
    fig.add_trace(
        # Stacked bar plot
        go.Bar(
            x=inhibitor_synergy_summary['n_synergies_per_inhibitor'],
            y=inhibitor_synergy_summary['inhibitor_combination'],
            orientation='h',
            name='Synergistic',
            marker=dict(color=inhibitor_synergy_summary['moa_combination'].map(mechanism_combi_color)),
            text=inhibitor_synergy_summary['n_synergies_per_inhibitor'],
            hoverinfo='x+y+text',
        ), row=1, col=1
    )
    fig.add_trace(
        # Non-synergistic bar plot
        go.Bar(
            x=inhibitor_synergy_summary['total_combinations_per_inhibitor'] - inhibitor_synergy_summary['n_synergies_per_inhibitor'],
            y=inhibitor_synergy_summary['inhibitor_combination'],
            orientation='h',
            name='Non-synergistic',
            marker=dict(color='lightgray'),
            text=inhibitor_synergy_summary['total_combinations_per_inhibitor'] - inhibitor_synergy_summary['n_synergies_per_inhibitor'],
            hoverinfo='x+y+text',
        ), row=1, col=1
    )

    fig.update_xaxes(title_text='Number of combinations across the drug screen', row=1, col=1)
    fig.update_yaxes(title_text='Inhibitor combination', row=1, col=1)
    fig.update_xaxes(title_text='Number of cell lines', row=1, col=2)

    # Add percentage text outside the barplot
    for i in range(len(inhibitor_synergy_summary)):
        fig.add_annotation(
            x=106,
            y=inhibitor_synergy_summary['inhibitor_combination'].iloc[i],
            text=f"{inhibitor_synergy_summary['n_synergies_per_inhibitor'].iloc[i] / inhibitor_synergy_summary['total_combinations_per_inhibitor'].iloc[i] * 100:.1f}%",
            showarrow=False,
            font=dict(size=12),
            xanchor='left',
            yanchor='middle'
        )

    # Add total number of tests per inhibitor combination at the end of the bar
    for i in range(len(inhibitor_synergy_summary)):
        fig.add_annotation(
            x=inhibitor_synergy_summary['total_combinations_per_inhibitor'].iloc[i],
            y=inhibitor_synergy_summary['inhibitor_combination'].iloc[i],
            text=f"  ({inhibitor_synergy_summary['total_combinations_per_inhibitor'].iloc[i]})",
            showarrow=False,
            font=dict(size=12),
            xanchor='left',
            yanchor='middle'
        )

    # HORIZONTAL BAR inhibitor combination across cell lines
    fig.add_trace(
        go.Bar(
            y=inhibitor_synergy_summary['inhibitor_combination'],
            x=inhibitor_synergy_summary['No.synergy across cell lines'],
            orientation='h',
            name='Mean synergy',
            marker=dict(color=inhibitor_synergy_summary['moa_combination'].map(mechanism_combi_color)),
            text=inhibitor_synergy_summary['No.synergy across cell lines'],
            hoverinfo='x+y+text',
        ), row=1, col=2
    )

    fig.update_layout(barmode='stack', 
        title='Tested Combinations vs. Synergistic Combinations and Synergy Counts',
        height=600,
        width=1200,
    )
    
    if show:
        fig.show()
