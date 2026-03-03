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
import json
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
    experimental = results.get('files', {}).get('experimental')
    dicts = dict(results.get('dicts', {}) or {})

    shared_dir = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "synco_shared")

    # Fallback: try synco_shared when experimental is not in the per-tissue results_dir
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

    # Fallback: load JSON dicts from synco_shared when missing from per-tissue results
    dict_filenames = {
        "PD_mechanism_dict":    "PD_mechanism_dict.json",
        "PD_inhibitors_dict":   "PD_inhibitors_dict.json",
        "mechanism_PD_dict":    "mechanism_PD_dict.json",
        "Drugnames_PD_dict":    "Drugnames_PD_dict.json",
        "PD_drugnames_dict":    "PD_drugnames_dict.json",
        "inhibitorgroups_dict": "inhibitorgroups_dict.json",
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

    # Scope experimental data to the tissue-specific cell lines.
    # predictions_full_df is in wide format with cell lines as column headers and is
    # already filtered to the current tissue's cell lines by the pipeline.
    # When experimental comes from synco_shared it contains ALL cell lines, so we
    # restrict it to the same set used in predictions.
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
                logging.getLogger(__name__).info(
                    "Filtered experimental data to %d tissue cell lines", len(tissue_cell_lines)
                )

    experimental_input = {
        'files': {
            'experimental': experimental,
        },
        'dicts': dicts,
    }
    return experimental_input

#----------------------------------------------------------------------
# PREPARE
#----------------------------------------------------------------------

def _prepare_experimental_counts(experimental_input, threshold=0):
    """Process the loaded input data for experimental distribution plotting.
    """
    experimental_df = experimental_input['files']['experimental']
    if experimental_df is None:
        raise ValueError(
            "Experimental data not found in results_dir. "
            "Experimental distribution plots require an experimental data file."
        )

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
    if bool(pm):
        experimental_df['mech_combination'] = (
            experimental_df['Mechanism_A'].fillna('Unknown') + " + " + experimental_df['Mechanism_B'].fillna('Unknown')
        )
    else:
        # No mechanism dict — fall back to inhibitor combination (always present)
        experimental_df['mech_combination'] = (
            experimental_df['inhibitor_combination']
            if 'inhibitor_combination' in experimental_df.columns
            else 'Unknown + Unknown'
        )
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
    
    # Sort data by percentage synergistic (highest -> lowest) so bars show
    # cell lines from most to least synergistic. The scatter uses the same
    # categorical ordering so the y-axis aligns across plots.
    stacked_bar_data = stacked_bar_data.sort_values(by='percentage_synergistic', ascending=True).reset_index(drop=True)

    # Have same order of cell lines in scatter and stacked bar
    cell_line_order = stacked_bar_data['cell_line'].tolist()
    scatter_data['cell_line'] = pd.Categorical(scatter_data['cell_line'], categories=cell_line_order, ordered=True)
    
    return hist_data, scatter_data, stacked_bar_data, exp_mean, exp_median, exp_std, above_threshold_counts, total_counts, percentage_above_threshold

def _style_mechanism_colors(mech_combi_list):
    """Define colors for each mechanism combination.
    """
    base_colors = px.colors.qualitative.Pastel
    mechanism_combi_color = {}
    for i, mech_combi in enumerate(mech_combi_list):
        mechanism_combi_color[mech_combi] = base_colors[i % len(base_colors)]
    return mechanism_combi_color

def _style_pairwise_colors(scatter_data, selected: tuple):
    """Define colors for combinations that have the selected mechanism in the pair.
    """
    # `selected` is expected to be a sequence with up to two mechanism names
    # to highlight: (mech1, mech2). We assign distinct colors for each.
    if selected is None:
        selected = ()

    mech1 = None
    mech2 = None
    mech3 = None
    if isinstance(selected, (list, tuple)) and len(selected) > 0:
        mech1 = selected[0]
        if len(selected) > 1:
            mech2 = selected[1]
        if len(selected) > 2:
            mech3 = selected[2]

    # Colors: mech1, mech2, both, and neutral for others
    color_mech1 = '#636EFA'
    color_mech2 = '#FC7299'
    color_mech3 = "#F887F8"
    neutral_color = "#303030"

    # Collect unique mechanism combinations from the scatter data
    mech_combis = []
    if hasattr(scatter_data, 'loc') and 'mech_combination' in scatter_data.columns:
        mech_combis = pd.Series(scatter_data['mech_combination'].dropna().unique()).astype(str).tolist()

    mechanism_pair_color = {}
    for comb in mech_combis:
        # Split using the exact separator used elsewhere (' + ')
        parts = [p.strip() for p in comb.split(' + ')]

        match1 = False
        match2 = False
        match3 = False
        if mech1:
            match1 = any((mech1 == p) or (mech1 in p) for p in parts if isinstance(p, str))
        if mech2:
            match2 = any((mech2 == p) or (mech2 in p) for p in parts if isinstance(p, str))
        if mech3:
            match3 = any((mech3 == p) or (mech3 in p) for p in parts if isinstance(p, str))
        if match3:
            mechanism_pair_color[comb] = color_mech3
        elif match1:
            mechanism_pair_color[comb] = color_mech1
        elif match2:
            mechanism_pair_color[comb] = color_mech2
        else:
            mechanism_pair_color[comb] = neutral_color

    return mechanism_pair_color
    

#----------------------------------------------------------------------
# PLOT
#----------------------------------------------------------------------

def _plot_stackedbars_synergy_counts(synergy_counts, inhibitor_synergy_summary, selected_mechanism, mechanism_combi_color, show=False, height=None, width=None):
    """Plot two stacked bar plots of synergy counts by inhibitor combination and by cell line.
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.02,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=('Synergy frequencies by inhibitor combination', 'Cell-line synergy coverage'),
        shared_yaxes=True
    )

    if selected_mechanism:
        pairwise_color_map = _style_pairwise_colors(inhibitor_synergy_summary, selected_mechanism)
    else:
        pairwise_color_map = None

    # STAKED BAR inhibitor combination (syn vs- non-syn)
    fig.add_trace(
        # Stacked bar plot
        go.Bar(
            x=inhibitor_synergy_summary['n_synergies_per_inhibitor'],
            y=inhibitor_synergy_summary['inhibitor_combination'],
            orientation='h',
            name='Synergistic',
            marker=dict(color=inhibitor_synergy_summary['mech_combination'].map(pairwise_color_map if pairwise_color_map else mechanism_combi_color)),
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
            marker=dict(color=inhibitor_synergy_summary['mech_combination'].map(pairwise_color_map if pairwise_color_map else mechanism_combi_color)),
            text=inhibitor_synergy_summary['n_synergies_across_cell_lines'],
            hoverinfo='x+y+text',
            showlegend=False,
        ), row=1, col=2
    )

    # Legend of annotation for mechanism combinations, parenthesis and percentage
    if pairwise_color_map:
        for mech_combi, color in pairwise_color_map.items():
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
    else:
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
        height=height or 600,
        width=width or 1400,
    )
    
    if show:
        fig.show()
    return fig

def _plot_histogram_experimental_distribution(
        hist_data, scatter_data, stacked_bar_data,
        exp_mean, exp_median, exp_std,
        threshold, above_threshold_counts, total_counts, percentage_above_threshold,
        mechanism_combi_color,
        selected_mechanism = None,
        height=None, width=None,
        show=False):
    """Multi plot figure for experimental synergy distribution: histogram, scatter, stacked bar.
    """
    # Sort the date by highest synergy

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

    # --- Row 1 Col 1: Histogram---
    fig.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=60,
            marker=dict(color="#737986"),
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
                            marker=dict(color="black", size=10),
                            name="Synergy threshold", showlegend=True), 
                    row=1, col=1)

    fig.update_yaxes(title_text='Count', row=1, col=1)

    # --- Row 1 Col 2: Empty (for alignment) ---
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # --- Row 2 Col 1: Scatter plot ---
    pairwise_color_map = None
    if selected_mechanism:
        pairwise_color_map = _style_pairwise_colors(scatter_data, selected_mechanism)

    fig.add_trace(
        go.Scatter(
            x=scatter_data['synergy'],
            y=scatter_data['cell_line'],
            mode='markers',
            marker=dict(
                # Use pairwise highlighting map when provided, otherwise
                # fall back to mechanism combination colors.
                color=(scatter_data['mech_combination'].map(pairwise_color_map)
                    if pairwise_color_map is not None else
                    scatter_data['mech_combination'].map(mechanism_combi_color)),
                size=8,
                opacity=0.7),
            text=scatter_data['drug_name_A'] + " + " + scatter_data['drug_name_B'] + " (" + scatter_data['mech_combination'] + ")",
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
    
    #add selected mechanism combination legend
    for mechanism, color in (pairwise_color_map.items() if selected_mechanism else mechanism_combi_color.items()):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=mechanism,
                showlegend=True,
            )
        )

    fig.update_xaxes(title_text='Synergy Score', row=2, col=1)
    fig.update_yaxes(title_text='Cell Line', row=2, col=1)

    # Ensure both subplots (scatter and bar) use the same category order so
    # the y-axis aligns. Build ordering from the stacked bar data which was
    # sorted by percentage (highest -> lowest) in the preparation step.
    try:
        cell_line_order = stacked_bar_data['cell_line'].tolist()
        fig.update_yaxes(categoryorder='array', categoryarray=cell_line_order, row=2, col=1)
        fig.update_yaxes(categoryorder='array', categoryarray=cell_line_order, row=2, col=2)
    except Exception:
        # If anything goes wrong (missing column), fall back to default ordering
        pass

    # --- Row 2 Col 2: Stacked Bar plot ---
    fig.add_trace(
        go.Bar(
            y=stacked_bar_data['cell_line'],
            x=stacked_bar_data['n_synergies_per_cell_line'],
            orientation='h',
            width=0.7,
            marker=dict(color="#737986"),
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
        height=height or 800,
        width=width or 1000,
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

def make_experimental_distribution_plots(results_dir, plots_dir, show=False, debug=False,
                                        threshold: float=0, selected_mechanism: tuple=None,
                                        distribution_size: tuple=None,
                                        stackedbar_size: tuple=None,
                                        return_fig: bool = False
                                        ):
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
    height_sb, width_sb = stackedbar_size if stackedbar_size else (600, 1400)
    fig_synergy_counts = _plot_stackedbars_synergy_counts(
        synergy_counts,
        inhibitor_synergy_summary,
        selected_mechanism,
        mechanism_combi_color,
        show=show,
        height=height_sb,
        width=width_sb,
    )

    height_dist, width_dist = distribution_size if distribution_size else (800, 1000)
    fig_synergy_distributuion = _plot_histogram_experimental_distribution(
        hist_data, scatter_data, stacked_bar_data,
        exp_mean, exp_median, exp_std,
        threshold, above_threshold_counts, total_counts, percentage_above_threshold,
        mechanism_combi_color,
        selected_mechanism,
        height=height_dist,
        width=width_dist,
        show=show,
    )

    if return_fig:
        return [(fig_synergy_counts, 'plotly'), (fig_synergy_distributuion, 'plotly')]

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