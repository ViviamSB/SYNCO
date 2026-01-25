"""
ROC/PR plotting: load -> process -> plot

This module provides functions to load ROC/PR curve data from saved results
and generate ROC and PR curve plots. The plotting script follows three steps:
- _load_roc_inputs(results_dir)
- _prepare_roc_data(roc_traces_data) [optional processing]
- make_roc_plots(results_dir, plots_dir, show=False)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .load_results import _load_main_results

#//////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------
# LOAD
#----------------------------------------------------------------------

def _load_roc_inputs(results_dir: str) -> Dict[str, Any]:
    """
    Load ROC/PR curve data from a results directory.
    
    Args:
        results_dir: Path to the directory containing pipeline results
        
    Returns:
        Dictionary containing:
        - 'traces_roc': List of (auc, go.Scatter) tuples for ROC curves
        - 'traces_pr': List of (auc, go.Scatter) tuples for PR curves  
        - 'rocauc_scores': List of ROC AUC scores
        - 'prauc_scores': List of PR AUC scores
        - 'results_dir': Path to results directory
    """
    results = _load_main_results(results_dir)
    roc_data = results.get('roc_traces')
    
    if roc_data is None:
        logging.warning(f"No ROC/PR curve data found in {results_dir}. "
                       "Ensure the pipeline was run with output_path specified.")
        return {
            'traces_roc': [],
            'traces_pr': [],
            'rocauc_scores': [],
            'prauc_scores': [],
            'roc_meta': [],
            'threshold_sweeps': [],
            'results_dir': str(results_dir)
        }
    
    return {
        'traces_roc': roc_data.get('traces_roc', []),
        'traces_pr': roc_data.get('traces_pr', []),
        'rocauc_scores': roc_data.get('rocauc_scores', []),
        'prauc_scores': roc_data.get('prauc_scores', []),
        'roc_meta': roc_data.get('roc_meta', []),
        'threshold_sweeps': roc_data.get('threshold_sweeps', []),
        'results_dir': str(results_dir)
    }

#----------------------------------------------------------------------
# PREPARE (optional - data is already in the right format)
#----------------------------------------------------------------------

def _prepare_roc_data(roc_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare ROC/PR data for plotting (currently just passes through).
    
    Args:
        roc_inputs: Dictionary from _load_roc_inputs
        
    Returns:
        Same dictionary (data is already in plotting format)
    """
    return roc_inputs

#----------------------------------------------------------------------
# PLOT
#----------------------------------------------------------------------

def plot_curves(
        traces, 
        auc_score_list, 
        tissue,
        metric='ROC', 
        width=800, 
        height=800,
    output=None,
    meta: Optional[Dict[str, Dict[str, Any]]] = None
        ):
    
    fig = go.Figure()

    sorted_traces = sorted(traces, key=lambda x: x[0], reverse=True)
    num_traces = len(sorted_traces)

    # Add sorted traces to the figure
    for _, trace in sorted_traces:
        cl_name = trace.name.split(' (')[0]
        md = meta.get(cl_name, {}) if meta else {}
        custom = [[
            md.get('threshold', None),
            md.get('balanced_accuracy', None),
            md.get('n_positive', None),
            md.get('n_negative', None),
            md.get('roc_auc_ci_low', None),
            md.get('roc_auc_ci_high', None),
            md.get('pr_auc_ci_low', None),
            md.get('pr_auc_ci_high', None),
        ] for _ in trace.x]
        if custom:
            trace.update(
                customdata=custom,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "x=%{x:.3f}, y=%{y:.3f}<br>"
                    "threshold=%{customdata[0]}<br>"
                    "BalAcc=%{customdata[1]}<br>"
                    "Npos=%{customdata[2]} | Nneg=%{customdata[3]}<br>"
                    "ROC CI=[%{customdata[4]}, %{customdata[5]}]<br>"
                    "PR CI=[%{customdata[6]}, %{customdata[7]}]"
                    "<extra></extra>"
                ),
                text=[cl_name]*len(trace.x)
            )
        fig.add_trace(trace)

    # Add average auc_score annotation
    avg_auc_score = sum(auc_score_list) / len(auc_score_list)
    fig.add_annotation(
        x=0.5, y=1.05,
        text=f'Average {metric}-AUC Score: {avg_auc_score:.4f}',
        showarrow=False,
        font=dict(size=14)
    )

    # Add median auc_score annotation
    median_auc_score = np.median(auc_score_list)
    fig.add_annotation(
        x=0.5, y=1.02,
        text=f'Median {metric}-AUC Score: {median_auc_score:.4f}',
        showarrow=False,
        font=dict(size=14)
    )

    if metric == 'ROC':
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            title=f'{metric} Curve for {tissue} cell lines (n={num_traces})',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR)',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=width, height=height
        )
    elif metric == 'PR':
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig.update_layout(
            title=f'{metric} Curve for {tissue} cell lines (n={num_traces})',
            xaxis_title='Recall',
            yaxis_title='Precision',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=width, height=height
        )

    if output:
        fig.write_html(output / f"{metric}_curve_{tissue}.html")
        fig.write_image(output / f"{metric}_curve_{tissue}.svg", scale=2)
        
    fig.show()


#//////////////////////////////////////////////////////////////////////////////

def plot_violin_and_table(
        selected_tissues, 
        violin_data, 
        auc_summary_table,
        model,
        specific_date,
        metric='ROC',
        metric_score='roc_auc_score',
        threshold=0.5,
        width=1200,
        height=700,
        tissue_colors = {'Breast': 'pink', 'Colon': 'rgb(136, 204, 238)', 'Pancreas': '#FF9E7A'},
        line_colors = {'Breast': '#FF6692', 'Colon': '#636EFA', 'Pancreas': '#EF553B'}
        ):

    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        specs=[[{"type": "violin"}], [{"type": "table"}]]
    )

    # Add violin plots for each tissue
    for tissue in selected_tissues:
        tissue_data = violin_data[violin_data['tissue'] == tissue]
        color = tissue_colors.get(tissue, 'blue') # Default to 'blue' if tissue not found in dictionary
        linecolor = line_colors.get(tissue, 'blue') # Default to 'blue' if tissue not found in dictionary
        fig.add_trace(
            go.Violin(
                y=tissue_data[metric_score],
                x=tissue_data['tissue'],
                meanline_visible=True,
                points='all',
                name=f'{tissue}',
                hovertext='Cell line: ' + tissue_data['cell_line_name'],
                hovertemplate='AUC: %{y:.2f}<extra></extra>, %{hovertext}, (%{x})',
                line_color=linecolor,
                fillcolor=color,
                opacity=0.8
            ),
            row=1, col=1
        )

    # Add threshold line
    fig.add_shape(
        name='threshold_line',
        label=dict(text=f'{metric}_THR = {threshold}', font_size=12, textposition='end'),
        type="line",
        x0=-0.5, x1=len(selected_tissues) - 0.5,
        y0=threshold, y1=threshold,
        line=dict(color="rgb(82, 106, 131)", width=2, dash="6px"),
        row=1, col=1
    )

    # Add table with summary statistics
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Tissue', 'Cell lines', 'Avg', 'Median', 'Max', 'Min', 'Std. Dev.', 'AUC>THR', 'P-value'],
                font=dict(size=12),
                align="center"
            ),
            cells=dict(
                values=[auc_summary_table.index] + [auc_summary_table[col].tolist() for col in auc_summary_table.columns],
                font=dict(size=12),
                align="center")
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=height,
        width=width,
        showlegend=True,
        title_text=f"Violin plot of {metric} AUC Scores by Tissue (model: {model}, run date: {specific_date})",
    )
    fig.show()


#//////////////////////////////////////////////////////////////////////////////

def plot_dots(
        auc_values,
        tissue_dict,
        tissue,
        show_scores=False,
        colorscale='Sunset',
        metric='ROC',
        threshold=0.5,
        height=800,
        width=500
):
    
    #Sort AUC values and cell line names for dots plot
    sorted_indices = sorted(range(len(auc_values)), key=lambda k: auc_values[k], reverse=False)
    sorted_auc_values = [auc_values[i] for i in sorted_indices]
    sorted_cell_lines = [tissue_dict[tissue][i] for i in sorted_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_auc_values,
        y=sorted_cell_lines,
        mode='markers',
        name=f'{metric}_AUC',
        marker=dict(
            size=16,
            color=sorted_auc_values,  # set color equal to a variable
            colorscale= colorscale
            # showscale=True
        )
    ))

    # Add AUC threshold line in x=0.5
    fig.add_shape(
        type='line',
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=len(tissue_dict[tissue]),
        line=dict(
            color='rgb(98, 83, 119)',
            width=3,
            dash='6px'
        )
    )

    # Add a dummy scatter for the threshold legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        name=f'AUC = {threshold}',
        line=dict(
            color='rgb(98, 83, 119)',  # Same color as the threshold line
            width=3,
            dash='dash'
        )
    ))

    if show_scores:
        # Annotations
        annotations = [
            dict(
                x=1.2,
                y=i,
                xref='paper',
                yref='y',
                text=f'AUC={auc:.3f}',
                showarrow=False,
                xanchor='right',
            ) for i, auc in enumerate(sorted_auc_values)
        ]

        fig.update_layout(
            annotations=annotations
        )

    fig.update_layout(
        title=f'{metric} AUC Scores for {tissue} cell line',
        xaxis_title='AUC Value',
        yaxis_title='Cell line',
        # showlegend=True,
        height=height,
        width=width,
        margin=dict(r=150)
    )

    fig.show()

def plot_multi_dots(
    auc_values_dict,
    metric='ROC',
    colorscale_by_auc=False,
    show_scores=False,
    threshold=0.5,
    height=800,
    width=500
):
    
    fig = go.Figure()

    # Group the dataframe by tissue
    grouped = auc_values_dict.groupby('tissue')

    for tissue, group in grouped:
        # Sort the group by AUC values
        group = group.sort_values(by='roc_auc_score')
        sorted_auc_values = group['roc_auc_score'].tolist()
        sorted_cell_lines = group['cell_line_name'].tolist()
        
        if colorscale_by_auc:
            fig.add_trace(go.Scatter(
                x=sorted_auc_values,
                y=sorted_cell_lines,
                legendgroup='AUC Scores',
                legendgrouptitle=dict(text='AUC Scores by tissue'),
                mode='markers',
                name=f'{tissue}',
                marker=dict(
                    size=12,
                    color=sorted_auc_values,  # set color equal to a variable
                    colorscale= 'Sunset'
                ),
            ))

        else:
            fig.add_trace(go.Scatter(
                x=sorted_auc_values,
                y=sorted_cell_lines,
                legendgroup='AUC Scores',
                legendgrouptitle=dict(text='AUC Scores by tissue'),
                name=f'{tissue}',
                mode='markers',
                marker=dict(
                    size=12
                ),
            ))

        if show_scores:
            # Annotations
            annotations = [
                dict(
                    x=1.2,  # Offset to the right
                    y=cell_line,
                    text=f'{auc:.3f}',
                    showarrow=False,
                    xanchor='left'
                ) for auc, cell_line in zip(sorted_auc_values, sorted_cell_lines)
            ]
            fig.update_layout(
                annotations=list(fig.layout.annotations) + annotations
            )

    # Add AUC threshold line in x=0.5
    fig.add_shape(
        type='line',
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=len(auc_values_dict['cell_line_name'].unique()),
        line=dict(
            color='rgb(98, 83, 119)',  # Dark purple
            width=3,
            dash='dash'
        )
    )

    # Add a dummy scatter for the threshold legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        legendgroup='threshold',
        legendgrouptitle=dict(text='Threshold'),
        name=f'AUC = {threshold}',
        line=dict(
            color='rgb(98, 83, 119)',  # Same color as the threshold line
            width=3,
            dash='dash'
        )
    ))

    fig.update_layout(
        title=f'{metric} AUC Scores for Multiple Tissues',
        xaxis_title='AUC Value',
        yaxis_title='Cell line',
        showlegend=True,
        height=height,
        width=width,
        margin=dict(r=150)  # Increased right margin
    )

    fig.show()

#----------------------------------------------------------------------
# WRAPPER
#----------------------------------------------------------------------

def make_roc_plots(
        results_dir: str,
        plots_dir: Optional[str] = None,
        show: bool = False,
        tissue: str = "all",
        width: int = 800,
    height: int = 800,
    plot_sweeps: bool = True
) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """
    High-level entrypoint: load -> prepare -> plot ROC and PR curves.
    
    Args:
        results_dir: Path to directory containing pipeline results (with roc_pr_curves.json)
        plots_dir: Directory to save plot files (if None, uses results_dir/plots)
        show: Whether to display interactive figures
        model: Model name for plot annotation
        specific_date: Run date for plot annotation (if None, shows "Last run date")
        tissue: Tissue name for plot title
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Tuple of (roc_fig, pr_fig) or (None, None) if no data available
    """
    # Load ROC/PR curve data
    roc_inputs = _load_roc_inputs(results_dir)
    
    # Check if we have data
    if not roc_inputs['traces_roc'] or not roc_inputs['traces_pr']:
        logging.warning("No ROC/PR curve data available to plot. "
                       "Ensure the pipeline was run with output_path specified.")
        return None, None
    
    # Prepare data (optional step, currently a pass-through)
    roc_data = _prepare_roc_data(roc_inputs)
    
    # Set up output directory
    if plots_dir is None:
        plots_dir = Path(results_dir) / 'plots'
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curves
    logging.info("Generating ROC curve plot...")
    meta_map = {m.get('cell_line'): m for m in roc_data.get('roc_meta', []) if m.get('cell_line')}
    plot_curves(
        traces=roc_data['traces_roc'],
        auc_score_list=roc_data['rocauc_scores'],
        tissue=tissue,
        metric='ROC',
        width=width,
        height=height,
        output=plots_dir if show else plots_dir,
        meta=meta_map
    )
    
    # Plot PR curves
    logging.info("Generating PR curve plot...")
    plot_curves(
        traces=roc_data['traces_pr'],
        auc_score_list=roc_data['prauc_scores'],
        tissue=tissue,
        metric='PR',
        width=width,
        height=height,
        output=plots_dir if show else plots_dir,
        meta=meta_map
    )

    # Threshold sweep plot (offset vs metric)
    if plot_sweeps and roc_data.get('threshold_sweeps'):
        try:
            plot_threshold_sweeps(
                sweeps=roc_data['threshold_sweeps'],
                plots_dir=plots_dir,
                metric='roc_auc',
                width=width,
                height=height//2,
                show=show
            )
        except Exception as e:
            logging.warning(f"Failed to plot threshold sweeps: {e}")
    
    logging.info(f"ROC/PR curve plots saved to: {plots_dir}")
    
    return None  # Figures are shown/saved by plot_curves


def plot_threshold_sweeps(
    sweeps: list,
    plots_dir: Path,
    metric: str = 'roc_auc',
    width: int = 800,
    height: int = 400,
    show: bool = False
):
    """Plot threshold offset sweeps per cell line.

    Args:
        sweeps: List of {cell_line, sweep: [{offset, threshold, roc_auc, pr_auc, f1_score, balanced_accuracy}]}
        plots_dir: Where to save the plot
        metric: Which metric to show on y-axis ('roc_auc', 'pr_auc', 'f1_score', 'balanced_accuracy')
    """
    metric = metric or 'roc_auc'
    fig = go.Figure()
    for entry in sweeps:
        cl = entry.get('cell_line', 'cell_line')
        points = entry.get('sweep', []) or []
        if not points:
            continue
        offsets = [p.get('offset') for p in points]
        values = [p.get(metric) for p in points]
        thresholds = [p.get('threshold') for p in points]
        # Highlight base offset 0 if present
        marker = dict(size=10)
        fig.add_trace(
            go.Scatter(
                x=offsets,
                y=values,
                mode='lines+markers',
                name=cl,
                text=[f"thr={t}" for t in thresholds],
                hovertemplate="offset=%{x}<br>value=%{y:.3f}<br>%{text}<extra></extra>",
                marker=marker
            )
        )

    fig.update_layout(
        title=f"Threshold sweep ({metric})",
        xaxis_title="Offset (multipliers around base threshold)",
        yaxis_title=metric,
        width=width,
        height=height,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(plots_dir / f"threshold_sweep_{metric}.html")
    try:
        fig.write_image(plots_dir / f"threshold_sweep_{metric}.svg", scale=2)
    except Exception:
        pass
    if show:
        fig.show()
    return fig