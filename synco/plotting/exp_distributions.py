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

    # Remove rows with missing PD information and ensure we have a copy
    experimental_df = experimental_df.dropna(subset=['PD_A', 'PD_B']).copy()

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

    # Compute total combinations per inhibitor BEFORE subsetting/dropping columns
    mechanism_matrix['total_combinations_per_inhibitor'] = mechanism_matrix.groupby(
        ['inhibitor_combination'])['synergy_binary'].transform('count').round(0).astype(int)

    # Compute number of cell lines with at least one synergistic observation per inhibitor combination.
    # Prefer counting unique cell identifiers if such a column exists, otherwise count rows with synergy.
    cell_col = next((c for c in mechanism_matrix.columns if 'cell' in c.lower()), None)
    if cell_col is not None:
        n_synergies_across = (
            mechanism_matrix.loc[mechanism_matrix['synergy_binary'] == 1]
            .groupby('inhibitor_combination')[cell_col]
            .nunique()
        )
    else:
        n_synergies_across = (
            mechanism_matrix.loc[mechanism_matrix['synergy_binary'] == 1]
            .groupby('inhibitor_combination')
            .size()
        )

    mechanism_matrix = mechanism_matrix[['inhibitor_combination', 'n_synergies_per_inhibitor', 'mech_combination', 'total_combinations_per_inhibitor']].drop_duplicates().reset_index(drop=True)
    mechanism_matrix['percentage'] = mechanism_matrix['n_synergies_per_inhibitor'] / mechanism_matrix['total_combinations_per_inhibitor'] * 100

    inhibitor_synergy_summary = mechanism_matrix.groupby('inhibitor_combination').agg({
        'n_synergies_per_inhibitor': 'first',
        'total_combinations_per_inhibitor': 'first',
        'percentage': 'first',
        'mech_combination': 'first',
    }).reset_index()
    inhibitor_synergy_summary = inhibitor_synergy_summary.sort_values(by='percentage', ascending=True).reset_index(drop=True)

    # Map previously computed per-inhibitor combo counts (number of cell lines with synergy)
    inhibitor_synergy_summary['n_synergies_across_cell_lines'] = (
        inhibitor_synergy_summary['inhibitor_combination']
        .map(n_synergies_across)
        .fillna(0)
        .astype(int)
    )

    combi_order = inhibitor_synergy_summary['inhibitor_combination'].tolist()
    synergy_counts = inhibitor_synergy_summary.set_index('inhibitor_combination').reindex(combi_order)


    return experimental_df, inhibitor_synergy_summary, synergy_counts, mech_combi_list

def _prepare_experimental_distribution(experimental_df, threshold=0):
    """ 
    Process the loaded input data for experimental distribution plotting.
    """
    exp_mean = experimental_df['synergy'].mean()
    exp_median = experimental_df['synergy'].median()
    exp_std = experimental_df['synergy'].std()

    # Responses above threshold are considered synergistic
    above_threshold_counts = (experimental_df['synergy'] >= threshold).sum()
    total_counts = experimental_df.shape[0]
    percentage_above_threshold = (above_threshold_counts / total_counts) * 100

    # Histogram data
    hist_data = experimental_df['synergy']

    # Scatter data (synergy, cell line, inhibitor combination, mechanism combination)
    scatter_data = experimental_df[['synergy', 'cell_line', 'inhibitor_combination', 'mech_combination', 'drug_name_A', 'drug_name_B']].copy()

    # Satcked bar data (n_synergies_per_cell_line, cell_line, percentage_synergistic)
    stacked_bar_data = experimental_df.copy()
    stacked_bar_data['synergy_binary'] = stacked_bar_data['synergy'].apply(lambda x: 1 if x >= threshold else 0)
    stacked_bar_data['n_synergies_per_cell_line'] = stacked_bar_data.groupby(
        ['cell_line'])['synergy_binary'].transform('sum').round(0).astype(int)
    stacked_bar_data['percentage_synergistic'] = stacked_bar_data['n_synergies_per_cell_line'] / stacked_bar_data.groupby(
        ['cell_line'])['synergy_binary'].transform('count') * 100

    # Reduce to one row per cell_line for plotting: otherwise plotly will draw one bar per
    # original row (many stacked bars). Keep only the aggregated columns.
    stacked_bar_data = stacked_bar_data[['cell_line', 'n_synergies_per_cell_line', 'percentage_synergistic']] \
        .drop_duplicates().reset_index(drop=True)
    
    
    return hist_data, scatter_data, stacked_bar_data, exp_mean, exp_median, exp_std, above_threshold_counts, total_counts, percentage_above_threshold

def _style_mechanism_colors(mech_combi_list):
    """Define colors for each mechanism combination.
    """
    base_colors = px.colors.qualitative.Pastel
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
            marker=dict(color=inhibitor_synergy_summary['mech_combination'].map(mechanism_combi_color)),
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
            x=inhibitor_synergy_summary['n_synergies_across_cell_lines'],
            orientation='h',
            name='Mean synergy',
            marker=dict(color=inhibitor_synergy_summary['mech_combination'].map(mechanism_combi_color)),
            text=inhibitor_synergy_summary['n_synergies_across_cell_lines'],
            hoverinfo='x+y+text',
            showlegend=False,
        ), row=1, col=2
    )

    # Legend of annotation for mechanism combinations, parenthesis and percentage
    for mech_combi, color in mechanism_combi_color.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=mech_combi,
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='lightgray'),
            name='(n): Total number of tested combinations',
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(0,0,0,0)'),
            name='%: Percentage of synergistic combinations',
            showlegend=True,
        )
    )

    fig.update_layout(legend_title_text='Mechanism combinations and annotations',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    fig.update_layout(barmode='stack', 
        title='Tested Combinations vs. Synergistic Combinations and Synergy Counts',
        height=700,
        width=1400,
    )
    
    if show:
        fig.show()
    return fig

def _plot_histogram_experimental_distribution(
        hist_data, scatter_data, stacked_bar_data,
        exp_mean, exp_median, exp_std,
        threshold, above_threshold_counts, total_counts, percentage_above_threshold,
        mechanism_combi_color,
        show=False):
    """Multi plot figure for experimental synergy distribution: histogram, scatter, stacked bar.
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        row_heights=[0.30, 0.70],
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
        shared_yaxes=True  # share y-axis across columns in the same row (so bottom row scatter & bar share y)
    )

    # --- Row 1 Col 1: Histogram + Violin (violin overlaid, semi-transparent) ---
    fig.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=60,
            marker=dict(color="#85A3E2"),
            showlegend=False,
            opacity=0.9
        ),
        row=1, col=1
    )

    # Mean & threshold lines
    fig.add_vline(x=exp_mean, line_color='blue', row=1, col=1,)
    fig.add_vline(x=threshold, line_color='black', row=1, col=1,)
    fig.add_annotation(
        text=f"Responses above threshold: {above_threshold_counts} / {total_counts} ({percentage_above_threshold:.1f}%)",
        xref="x domain", yref="y domain",
        x=0.9, y=1.1,
        showarrow=False,
        row=1, col=1,
        font=dict(size=12)
    )

    fig.add_trace(go.Scatter(x=[exp_mean], y=[0], mode="markers",
                             marker=dict(color="blue", size=10),
                             name="Mean synergy score", showlegend=True),
                    row=1, col=1)
    fig.add_trace(go.Scatter(x=[threshold], y=[0], mode="markers",
                             marker=dict(color="red", size=10),
                             name="Synergy threshold", showlegend=True), 
                    row=1, col=1)

    fig.update_yaxes(title_text='Count', row=1, col=1)

    # --- Row 1 Col 2: Empty (for alignment) ---
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # --- Row 2 Col 1: Scatter plot ---
    fig.add_trace(
        go.Scatter(
            x=scatter_data['synergy'],
            y=scatter_data['cell_line'],
            mode='markers',
            marker=dict(
                color=scatter_data['mech_combination'].map(mechanism_combi_color),
                size=8,
                opacity=0.7),
            text=scatter_data['drug_name_A'] + " + " + scatter_data['drug_name_B'],
            hoverinfo='text+x',
            name='Drug Combination',
            showlegend=False,
        ),
        row=2, col=1
        ),

        # threshold line
    fig.add_vline(x=threshold, line_color='black',
                  annotation_text='Synergy threshold',
                  annotation_position='top left',
                  row=2, col=1,)
    
    #add mechanism combination legend
    for mech_combi, color in mechanism_combi_color.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=mech_combi,
                showlegend=True,
            )
        )
    fig.update_xaxes(title_text='Synergy Score', row=2, col=1)
    fig.update_yaxes(title_text='Cell Line', row=2, col=1)

    # --- Row 2 Col 2: Stacked Bar plot ---
    fig.add_trace(
        go.Bar(
            y=stacked_bar_data['cell_line'],
            x=stacked_bar_data['n_synergies_per_cell_line'],
            orientation='h',
            width=0.7,
            marker=dict(color="#6B9AFF"),
            # use numeric percentage so Plotly texttemplate can format it
            text=stacked_bar_data['percentage_synergistic'],
            hoverinfo='x+y+text',
            name='No. synergies per cell line',
            textposition='outside',
            showlegend=False,
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text='Number of Synergistic Combinations', row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_traces(texttemplate='%{text:.0f}%', textfont_size=12, textposition='inside', row=2, col=2)

    # Layout
    fig.update_layout(
        title='Experimental Synergy Distribution Analysis',
        height=800,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1.02,
        )
    )
    if show:
        fig.show()
    return fig

    
#----------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------

def make_experimental_distribution_plots(results_dir, plots_dir, show=False, debug=False, threshold: float=0):
    """Make experimental distribution plots: load -> process -> plot
    """
    # LOAD
    experimental_input = _load_experimental_inputs(results_dir)

    # PREPARE
    experimental_df, inhibitor_synergy_summary, synergy_counts, mech_combi_list = _prepare_experimental_counts(experimental_input, threshold=threshold)
    mechanism_combi_color = _style_mechanism_colors(mech_combi_list)
    hist_data, scatter_data, stacked_bar_data, exp_mean, exp_median, exp_std, above_threshold_counts, total_counts, percentage_above_threshold = _prepare_experimental_distribution(experimental_df, threshold=threshold)
    if debug:
        print('Inhibitor synergy summary:', inhibitor_synergy_summary)
        print('Synergy counts:', synergy_counts)
        print('Histogram data:', hist_data)
        print('Scatter data:', scatter_data)
        print('Stacked bar data:', stacked_bar_data)
    
    # PLOT
    fig_synergy_counts = _plot_stackedbars_synergy_counts(
        synergy_counts,
        inhibitor_synergy_summary,
        mechanism_combi_color,
        show=show,
    )

    fig_synergy_distributuion = _plot_histogram_experimental_distribution(
        hist_data, scatter_data, stacked_bar_data,
        exp_mean, exp_median, exp_std,
        threshold, above_threshold_counts, total_counts, percentage_above_threshold,
        mechanism_combi_color,
        show=show,
    )

    save_fig(
        fig_synergy_counts,
        output_dir= os.path.join(plots_dir),
        basename='experimental_synergy_counts_stackedbar',
        formats=['png', 'html'],
        fig_type='plotly',
    )

    save_fig(
        fig_synergy_distributuion,
        output_dir= os.path.join(plots_dir),
        basename='experimental_synergy_distribution',
        formats=['png', 'html'],
        fig_type='plotly',
    )