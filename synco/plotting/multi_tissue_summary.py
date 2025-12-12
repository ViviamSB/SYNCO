# synco/plotting/multi_tissue_summary.py

"""Multi-tissue summary plotting: aggregate and compare results across tissues.

This module provides functions to load comparison summaries and ROC/PR AUC metrics
from multiple tissue directories, aggregate the data, and create visualizations:
- Ring plots showing per-tissue performance (one ring per tissue + aggregate ring)
- Violin plots showing ROC/PR AUC score distributions across tissues

The plotting follows the three-step pattern:
- load_all_tissue_summaries(cell_fate_dir)
- plot_tissue_rings(comparison_df, plots_dir, ...)
- plot_aggregate_ring(comparison_df, plots_dir, ...)
- plot_roc_pr_violin(roc_auc_df, plots_dir, selected_tissues, ...)
"""

import os
import json
import logging
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import save_fig
from .performance import _draw_ring, _legend_elements


#//////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------
# LOAD
#----------------------------------------------------------------------

def load_all_tissue_summaries(
    cell_fate_dir: str,
    tissues: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load comparison summaries and ROC metrics from all tissue directories.
    
    Args:
        cell_fate_dir: Path to examples/cell_fate/ directory containing tissue folders
        tissues: Optional list of tissue names to load. If None, loads all available tissues.
        
    Returns:
        Tuple of three DataFrames:
        - comparison_df: Per-tissue global metrics (tissue, accuracy, precision, recall, 
                         TP, TN, FP, FN, n_cell_lines, total_comparisons)
        - roc_auc_df: Flattened ROC/PR AUC data (tissue, cell_line, roc_auc_score, pr_auc_score)
        - missing_roc_df: Per-tissue count of missing ROC data (tissue, n_missing, n_total)
    """
    cell_fate_path = Path(cell_fate_dir)
    
    if not cell_fate_path.exists():
        raise FileNotFoundError(f"Cell fate directory not found: {cell_fate_dir}")
    
    # Discover all tissue directories (exclude synco_input)
    tissue_dirs = [d for d in cell_fate_path.iterdir() 
                   if d.is_dir() and d.name != 'synco_input']
    
    # Filter by requested tissues if provided
    if tissues:
        tissue_dirs = [d for d in tissue_dirs if d.name in tissues]
    
    if not tissue_dirs:
        raise ValueError(f"No tissue directories found in {cell_fate_dir}")
    
    logging.info(f"Loading data from {len(tissue_dirs)} tissues...")
    
    comparison_records = []
    roc_auc_records = []
    missing_roc_records = []
    
    for tissue_dir in sorted(tissue_dirs):
        tissue_name = tissue_dir.name
        synco_output = tissue_dir / 'synco_output'
        
        if not synco_output.exists():
            logging.warning(f"Skipping {tissue_name}: synco_output directory not found")
            continue
        
        # Load comparison summary JSON
        summary_json = synco_output / 'cell_line_comparison_summary.json'
        if summary_json.exists():
            try:
                with open(summary_json, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                # Count cell lines from comparison CSV (if available)
                comparison_csv = synco_output / 'cell_line_comparison_results.csv'
                n_cell_lines = 0
                if comparison_csv.exists():
                    comp_df = pd.read_csv(comparison_csv)
                    n_cell_lines = len(comp_df)
                
                # Extract global metrics (handle both with/without % suffix)
                def _get_metric(key_base):
                    """Try key with and without % suffix."""
                    for suffix in ['', ' %']:
                        key = f'Global {key_base}{suffix}'
                        if key in summary_data:
                            val = summary_data[key]
                            # Remove % suffix if present in value
                            if isinstance(val, str) and val.endswith('%'):
                                val = val.rstrip('%')
                            return float(val) if val else 0.0
                    return 0.0
                
                comparison_records.append({
                    'tissue': tissue_name,
                    'Accuracy': _get_metric('Accuracy'),
                    'Precision': _get_metric('Precision'),
                    'Recall': _get_metric('Recall'),
                    'TP': int(summary_data.get('Global True Positives', 0)),
                    'TN': int(summary_data.get('Global True Negatives', 0)),
                    'FP': int(summary_data.get('Global False Positives', 0)),
                    'FN': int(summary_data.get('Global False Negatives', 0)),
                    'Match': int(summary_data.get('Global matches', 0)),
                    'Mismatch': int(summary_data.get('Global mismatches', 0)),
                    'n_cell_lines': n_cell_lines,
                    'total_comparisons': int(summary_data.get('Total comparisons', 0))
                })
                
            except Exception as e:
                logging.warning(f"Failed to load comparison summary for {tissue_name}: {e}")
        else:
            logging.warning(f"Skipping {tissue_name}: comparison summary JSON not found")
        
        # Load ROC metrics CSV
        roc_metrics_csv = synco_output / 'roc_metrics_df.csv'
        if roc_metrics_csv.exists():
            try:
                roc_df = pd.read_csv(roc_metrics_csv)
                
                # Expected columns: cell_line, roc_auc, pr_auc (and possibly others)
                if 'cell_line' in roc_df.columns:
                    # Flatten ROC data with tissue column
                    for _, row in roc_df.iterrows():
                        roc_auc_records.append({
                            'tissue': tissue_name,
                            'cell_line': row.get('cell_line', ''),
                            'roc_auc_score': row.get('roc_auc', np.nan),
                            'pr_auc_score': row.get('pr_auc', np.nan)
                        })
                    
                    # Count missing ROC data
                    n_total = len(roc_df)
                    n_missing_roc = roc_df['roc_auc'].isna().sum() if 'roc_auc' in roc_df.columns else n_total
                    n_missing_pr = roc_df['pr_auc'].isna().sum() if 'pr_auc' in roc_df.columns else n_total
                    
                    missing_roc_records.append({
                        'tissue': tissue_name,
                        'n_missing_roc': int(n_missing_roc),
                        'n_missing_pr': int(n_missing_pr),
                        'n_total': n_total
                    })
                else:
                    logging.warning(f"ROC metrics for {tissue_name} missing 'cell_line' column")
                    
            except Exception as e:
                logging.warning(f"Failed to load ROC metrics for {tissue_name}: {e}")
        else:
            logging.debug(f"No ROC metrics found for {tissue_name}")
    
    # Create DataFrames
    comparison_df = pd.DataFrame(comparison_records)
    roc_auc_df = pd.DataFrame(roc_auc_records)
    missing_roc_df = pd.DataFrame(missing_roc_records)
    
    logging.info(f"Loaded {len(comparison_df)} tissues with comparison data")
    logging.info(f"Loaded {len(roc_auc_df)} cell line ROC/PR AUC records")
    
    return comparison_df, roc_auc_df, missing_roc_df


#----------------------------------------------------------------------
# PLOT - RING PLOTS
#----------------------------------------------------------------------

def plot_tissue_rings(
    comparison_df: pd.DataFrame,
    plots_dir: Optional[str] = None,
    plot_name: str = 'tissue_rings_grid',
    sort_by: str = 'Accuracy',
    ncols: int = 5,
    figsize: Tuple[int, int] = (20, 24),
    center_metric: str = 'recall',
    center_fontsize: int = 14,
    show: bool = False
) -> plt.Figure:
    """Plot grid of ring plots, one per tissue, showing global performance metrics.
    
    Each ring shows Match/Mismatch (outer) and TP/TN/FP/FN (inner) with center metric.
    Text below each ring shows Accuracy, Precision, and number of cell lines.
    
    Args:
        comparison_df: DataFrame with columns [tissue, Accuracy, Precision, Recall, 
                       TP, TN, FP, FN, Match, Mismatch, n_cell_lines]
        plots_dir: Directory to save plot (if None, doesn't save)
        plot_name: Base filename for saved plot
        sort_by: Column name to sort tissues by (default: 'Accuracy')
        ncols: Number of columns in grid
        figsize: Figure size as (width, height)
        center_metric: Metric to display in ring center ('recall', 'accuracy', 'precision')
        center_fontsize: Font size for center metric text
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")
    
    required_cols = {'tissue', 'Accuracy', 'Precision', 'Recall', 'TP', 'TN', 'FP', 'FN', 
                     'Match', 'Mismatch'}
    if not required_cols.issubset(set(comparison_df.columns)):
        raise ValueError(f"comparison_df missing required columns: {required_cols - set(comparison_df.columns)}")
    
    # Sort tissues
    df_sorted = comparison_df.copy()
    if sort_by in df_sorted.columns:
        df_sorted = df_sorted.sort_values(by=sort_by, ascending=False)
    
    n_tissues = len(df_sorted)
    nrows = int(math.ceil(n_tissues / float(ncols)))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    
    outer_colors = ['royalblue', '#D94602']
    inner_colors = ['#458cff', '#6db0ff', '#FA7F2E', '#FDAA65']
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, df_sorted.iterrows())):
        tissue_name = row['tissue']
        
        # Draw ring using centralized helper
        _draw_ring(
            ax, row, 
            title=tissue_name.capitalize(),
            outer_colors=outer_colors,
            inner_colors=inner_colors,
            center_metric=center_metric,
            center_fontsize=center_fontsize,
            title_fontsize=16,
            show_legend=False
        )
        
        # Add Accuracy and Precision text below ring
        accuracy_val = row.get('Accuracy', 0.0)
        precision_val = row.get('Precision', 0.0)
        n_cell_lines = int(row.get('n_cell_lines', 0))
        
        # Position text below the ring (using axis coordinates)
        ax.text(0, -1.2, f"Accuracy: {accuracy_val:.1f}%", 
                ha='center', va='center', fontsize=11, transform=ax.transData)
        ax.text(0, -1.4, f"Precision: {precision_val:.1f}%", 
                ha='center', va='center', fontsize=11, transform=ax.transData)
        ax.text(0, -1.6, f"N cell lines: {n_cell_lines}", 
                ha='center', va='center', fontsize=10, transform=ax.transData,
                style='italic', color='#555555')
    
    # Hide unused axes
    for ax in axes[n_tissues:]:
        ax.axis('off')
    
    # Add single legend for entire figure
    legend_elements = _legend_elements()
    fig.legend(handles=legend_elements, fontsize=11, loc='lower right', 
               bbox_to_anchor=(0.99, 0.02), framealpha=0.9)
    
    plt.suptitle(f'Modelling Performance Across {n_tissues} Tissues', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        save_fig(fig, plots_dir, plot_name, formats=['png', 'html'], scale=2, fig_type='matplotlib')
        logging.info(f"Saved tissue rings grid to {plots_dir}/{plot_name}")
    
    if show:
        try:
            if not mpl.get_backend().lower().startswith('agg'):
                plt.show()
        except Exception:
            pass
    
    return fig


def plot_aggregate_ring(
    comparison_df: pd.DataFrame,
    plots_dir: Optional[str] = None,
    plot_name: str = 'aggregate_ring',
    figsize: Tuple[int, int] = (8, 7),
    center_metric: str = 'recall',
    center_fontsize: int = 20,
    show: bool = False
) -> plt.Figure:
    """Plot single aggregate ring showing combined performance across all tissues.
    
    Aggregates TP/TN/FP/FN across all tissues and recalculates global metrics.
    
    Args:
        comparison_df: DataFrame with columns [tissue, TP, TN, FP, FN, Match, Mismatch, n_cell_lines]
        plots_dir: Directory to save plot (if None, doesn't save)
        plot_name: Base filename for saved plot
        figsize: Figure size as (width, height)
        center_metric: Metric to display in ring center ('recall', 'accuracy', 'precision')
        center_fontsize: Font size for center metric text
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")
    
    # Aggregate metrics across all tissues
    total_tp = int(comparison_df['TP'].sum())
    total_tn = int(comparison_df['TN'].sum())
    total_fp = int(comparison_df['FP'].sum())
    total_fn = int(comparison_df['FN'].sum())
    total_match = int(comparison_df['Match'].sum())
    total_mismatch = int(comparison_df['Mismatch'].sum())
    total_cell_lines = int(comparison_df['n_cell_lines'].sum())
    total_comparisons = int(comparison_df['total_comparisons'].sum())
    n_tissues = len(comparison_df)
    
    # Recalculate global metrics
    total = total_match + total_mismatch
    accuracy = (total_match / total * 100) if total > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    
    # Create aggregate row
    agg_row = pd.Series({
        'tissue': 'All Tissues',
        'TP': total_tp,
        'TN': total_tn,
        'FP': total_fp,
        'FN': total_fn,
        'Match': total_match,
        'Mismatch': total_mismatch,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    outer_colors = ['royalblue', '#D94602']
    inner_colors = ['#458cff', '#6db0ff', '#FA7F2E', '#FDAA65']
    
    # Draw aggregate ring
    _draw_ring(
        ax, agg_row,
        title='',
        outer_colors=outer_colors,
        inner_colors=inner_colors,
        center_metric=center_metric,
        center_fontsize=center_fontsize,
        show_legend=True
    )
    
    # Add metrics text below ring
    fig.text(0.5, 0.18, f"Accuracy: {accuracy:.1f}%", fontsize=13, ha='center', va='center')
    fig.text(0.5, 0.14, f"Precision: {precision:.1f}%", fontsize=13, ha='center', va='center')
    fig.text(0.5, 0.10, f"Total cell lines: {total_cell_lines} (across {n_tissues} tissues)", 
             fontsize=11, ha='center', va='center', style='italic', color='#555555')
    fig.text(0.5, 0.06, f"Total comparisons: {total_comparisons:,}", 
             fontsize=11, ha='center', va='center', style='italic', color='#555555')
    
    plt.title('Modelling Performance Across All Tissues', 
              fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()
    
    # Save
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        save_fig(fig, plots_dir, plot_name, formats=['png', 'html'], scale=2, fig_type='matplotlib')
        logging.info(f"Saved aggregate ring to {plots_dir}/{plot_name}")
    
    if show:
        try:
            if not mpl.get_backend().lower().startswith('agg'):
                plt.show()
        except Exception:
            pass
    
    return fig


#----------------------------------------------------------------------
# PLOT - ROC/PR VIOLIN PLOTS
#----------------------------------------------------------------------

def plot_roc_pr_violin(
    roc_auc_df: pd.DataFrame,
    plots_dir: Optional[str] = None,
    plot_name: str = 'roc_pr_violin',
    selected_tissues: Optional[List[str]] = None,
    metric: str = 'ROC',
    threshold: float = 0.5,
    width: int = 1400,
    height: int = 800,
    tissue_colors: Optional[Dict[str, str]] = None,
    show: bool = False
) -> go.Figure:
    """Plot violin plots showing ROC or PR AUC score distributions across tissues.
    
    Similar to plot_violin_and_table in roc_plots.py, but for multi-tissue comparison.
    Includes summary statistics table below the violin plots.
    
    Args:
        roc_auc_df: DataFrame with columns [tissue, cell_line, roc_auc_score, pr_auc_score]
        plots_dir: Directory to save plot (if None, doesn't save)
        plot_name: Base filename for saved plot
        selected_tissues: List of tissue names to include (None = all tissues)
        metric: 'ROC' or 'PR' to select which AUC metric to plot
        threshold: AUC threshold line to display
        width: Plot width in pixels
        height: Plot height in pixels
        tissue_colors: Optional dict mapping tissue names to colors
        show: Whether to display the plot
        
    Returns:
        plotly Figure object
    """
    if roc_auc_df.empty:
        raise ValueError("roc_auc_df is empty")
    
    required_cols = {'tissue', 'cell_line', 'roc_auc_score', 'pr_auc_score'}
    if not required_cols.issubset(set(roc_auc_df.columns)):
        raise ValueError(f"roc_auc_df missing required columns: {required_cols - set(roc_auc_df.columns)}")
    
    # Select metric column
    metric = metric.upper()
    if metric == 'ROC':
        metric_col = 'roc_auc_score'
        metric_label = 'ROC AUC'
    elif metric == 'PR':
        metric_col = 'pr_auc_score'
        metric_label = 'PR AUC'
    else:
        raise ValueError(f"metric must be 'ROC' or 'PR', got '{metric}'")
    
    # Filter by selected tissues
    df_plot = roc_auc_df.copy()
    if selected_tissues:
        df_plot = df_plot[df_plot['tissue'].isin(selected_tissues)]
        if df_plot.empty:
            raise ValueError(f"No data found for selected tissues: {selected_tissues}")
    
    # Remove NaN values for the selected metric
    df_plot = df_plot[df_plot[metric_col].notna()].copy()
    
    if df_plot.empty:
        raise ValueError(f"No valid {metric_label} data available")
    
    # Get unique tissues (sorted alphabetically)
    tissues = sorted(df_plot['tissue'].unique())
    n_tissues = len(tissues)
    
    # Create summary statistics table
    summary_records = []
    for tissue in tissues:
        tissue_data = df_plot[df_plot['tissue'] == tissue][metric_col]
        n_cell_lines = len(tissue_data)
        avg = tissue_data.mean()
        median = tissue_data.median()
        max_val = tissue_data.max()
        min_val = tissue_data.min()
        std = tissue_data.std()
        above_thr = (tissue_data > threshold).sum()
        
        summary_records.append({
            'Tissue': tissue.capitalize(),
            'Cell lines': n_cell_lines,
            'Avg': f'{avg:.3f}',
            'Median': f'{median:.3f}',
            'Max': f'{max_val:.3f}',
            'Min': f'{min_val:.3f}',
            'Std. Dev.': f'{std:.3f}',
            f'AUC>{threshold}': above_thr,
        })
    
    summary_df = pd.DataFrame(summary_records)
    
    # Create subplots (violin + table)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "violin"}], [{"type": "table"}]]
    )
    
    # Default color palette if not provided
    if tissue_colors is None:
        tissue_colors = {}
    
    # Add violin plots for each tissue
    for tissue in tissues:
        tissue_data = df_plot[df_plot['tissue'] == tissue]
        color = tissue_colors.get(tissue, None)
        
        fig.add_trace(
            go.Violin(
                y=tissue_data[metric_col],
                x=tissue_data['tissue'],
                name=tissue.capitalize(),
                meanline_visible=True,
                points='all',
                hovertext='Cell line: ' + tissue_data['cell_line'],
                hovertemplate='%{hovertext}<br>AUC: %{y:.3f}<extra></extra>',
                marker=dict(color=color) if color else None,
                line=dict(color=color) if color else None,
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=-0.5, x1=n_tissues - 0.5,
        y0=threshold, y1=threshold,
        line=dict(color="rgb(82, 106, 131)", width=2, dash="dash"),
        row=1, col=1
    )
    
    # Add threshold annotation
    fig.add_annotation(
        x=n_tissues - 0.5,
        y=threshold,
        text=f'{metric_label} THR = {threshold}',
        showarrow=False,
        xanchor='left',
        xshift=10,
        font=dict(size=11, color="rgb(82, 106, 131)"),
        row=1, col=1
    )
    
    # Add summary table
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_df.columns),
                font=dict(size=12),
                align="center",
                fill_color='lightgrey'
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                font=dict(size=11),
                align="center"
            )
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=height,
        width=width,
        showlegend=True,
        title_text=f"{metric_label} Score Distribution Across {n_tissues} Tissues",
        title_font=dict(size=16),
    )
    
    fig.update_xaxes(title_text="Tissue", row=1, col=1)
    fig.update_yaxes(title_text=f"{metric_label} Score", row=1, col=1)
    
    # Save
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        save_fig(fig, plots_dir, plot_name, formats=['png', 'html'], scale=2, fig_type='plotly')
        logging.info(f"Saved {metric_label} violin plot to {plots_dir}/{plot_name}")
    
    if show:
        try:
            fig.show()
        except Exception as e:
            logging.warning(f"Could not display figure: {e}")
    
    return fig


#----------------------------------------------------------------------
# WRAPPER
#----------------------------------------------------------------------

def make_multi_tissue_plots(
    cell_fate_dir: str,
    plots_dir: Optional[str] = None,
    tissues: Optional[List[str]] = None,
    sort_by: str = 'Accuracy',
    center_metric: str = 'recall',
    show: bool = False,
    selected_tissues: Optional[List[str]] = None
) -> Dict[str, any]:
    """High-level wrapper: load all tissue data and generate all multi-tissue plots.
    
    Args:
        cell_fate_dir: Path to examples/cell_fate/ directory
        plots_dir: Directory to save plots (if None, uses cell_fate_dir/multi_tissue_plots)
        tissues: Optional list of tissues to include (None = all)
        sort_by: Column to sort tissues by in ring grid ('Accuracy', 'Precision', 'Recall')
        center_metric: Metric to show in ring centers ('recall', 'accuracy', 'precision')
        show: Whether to display plots
        selected_tissues: Tissues for ROC/PR violins (None = all)
        
    Returns:
        Dictionary containing loaded DataFrames and generated figures
    """
    # Load data
    logging.info("Loading multi-tissue data...")
    comparison_df, roc_auc_df, missing_roc_df = load_all_tissue_summaries(
        cell_fate_dir, tissues=tissues
    )
    
    # Set up plots directory
    if plots_dir is None:
        plots_dir = Path(cell_fate_dir) / 'multi_tissue_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {
        'comparison_df': comparison_df,
        'roc_auc_df': roc_auc_df,
        'missing_roc_df': missing_roc_df,
        'figures': {}
    }
    
    # Generate tissue rings grid
    if not comparison_df.empty:
        logging.info("Generating tissue rings grid...")
        fig_grid = plot_tissue_rings(
            comparison_df,
            plots_dir=plots_dir,
            plot_name='tissue_rings_grid',
            sort_by=sort_by,
            center_metric=center_metric,
            show=show
        )
        results['figures']['tissue_rings_grid'] = fig_grid
    
    # Generate aggregate ring
    if not comparison_df.empty:
        logging.info("Generating aggregate ring...")
        fig_agg = plot_aggregate_ring(
            comparison_df,
            plots_dir=plots_dir,
            plot_name='aggregate_ring',
            center_metric=center_metric,
            show=show
        )
        results['figures']['aggregate_ring'] = fig_agg
    
    # Generate ROC violin plot
    if not roc_auc_df.empty:
        logging.info("Generating ROC AUC violin plot...")
        try:
            fig_roc = plot_roc_pr_violin(
                roc_auc_df,
                plots_dir=plots_dir,
                plot_name='roc_auc_violin',
                selected_tissues=selected_tissues,
                metric='ROC',
                show=show
            )
            results['figures']['roc_violin'] = fig_roc
        except Exception as e:
            logging.warning(f"Could not generate ROC violin plot: {e}")
    
    # Generate tissue-level box plots for Accuracy/Recall/Precision
    if not comparison_df.empty:
        logging.info("Generating tissue metric box plots...")
        try:
            figs_perf = plot_tissue_metric_boxplots(
                comparison_df,
                plots_dir=plots_dir,
                plot_name_prefix='tissue_metrics',
                show=show
            )
            results['figures']['tissue_metric_boxplots'] = figs_perf
        except Exception as e:
            logging.warning(f"Could not generate tissue metric box plots: {e}")

    # Generate tissue-level ROC/PR/F1 box & bar plots
    if not roc_auc_df.empty:
        logging.info("Generating tissue ROC/PR/F1 box & bar plots...")
        try:
            figs_rocprf1 = plot_tissue_roc_pr_f1_boxplots(
                roc_auc_df,
                plots_dir=plots_dir,
                plot_name_prefix='tissue_roc_pr_f1',
                show=show
            )
            results['figures']['tissue_roc_pr_f1_boxplots'] = figs_rocprf1
        except Exception as e:
            logging.warning(f"Could not generate tissue ROC/PR/F1 plots: {e}")
        except Exception as e:
            logging.warning(f"Could not generate PR violin plot: {e}")

        # Generate tissue-level box plots for Accuracy/Recall/Precision
        if not comparison_df.empty:
            logging.info("Generating tissue metric box plots...")
            try:
                fig_box = plot_tissue_metric_boxplots(
                    comparison_df,
                    plots_dir=plots_dir,
                    plot_name_prefix='tissue_metrics',
                    show=show
                )
                results['figures']['tissue_metric_boxplots'] = fig_box
            except Exception as e:
                logging.warning(f"Could not generate tissue metric box plots: {e}")
    
    logging.info(f"All multi-tissue plots saved to: {plots_dir}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MULTI-TISSUE SUMMARY")
    print(f"{'='*60}")
    print(f"Tissues analyzed: {len(comparison_df)}")
    print(f"Total cell lines: {comparison_df['n_cell_lines'].sum()}")
    print(f"Total comparisons: {comparison_df['total_comparisons'].sum():,}")
    if not missing_roc_df.empty:
        print(f"\nROC/PR data:")
        print(f"  Total cell lines with ROC data: {len(roc_auc_df[roc_auc_df['roc_auc_score'].notna()])}")
        print(f"  Total cell lines with PR data: {len(roc_auc_df[roc_auc_df['pr_auc_score'].notna()])}")
    print(f"\nPlots saved to: {plots_dir}")
    print(f"{'='*60}\n")
    
    return results


#----------------------------------------------------------------------
# PLOT - TISSUE BOX PLOTS (Accuracy/Recall/Precision)
#----------------------------------------------------------------------
def plot_tissue_metric_boxplots(
    comparison_df: pd.DataFrame,
    plots_dir: Optional[str] = None,
    plot_name_prefix: str = 'tissue_metrics',
    show: bool = False
):
    """Create box plots summarizing tissue-level Accuracy/Recall/Precision.

    Produces two plots similar to classification.py:
    - Box plots for Accuracy, Recall, Precision across tissues with mean/median annotations
    - Bar plot summary by tissue (optional for quick overview)

    Args:
        comparison_df: DataFrame with columns ['tissue','Accuracy','Recall','Precision']
        plots_dir: Directory to save plots
        plot_name_prefix: Base filename prefix
        show: Whether to display figures
    """
    import plotly.graph_objects as go

    req = {'tissue', 'Accuracy', 'Recall', 'Precision'}
    if not req.issubset(set(comparison_df.columns)):
        raise ValueError(f"comparison_df missing required columns: {req - set(comparison_df.columns)}")

    df = comparison_df.copy()
    df = df[['tissue', 'Accuracy', 'Recall', 'Precision']]

    # Box plot for Accuracy/Recall/Precision (aggregated at tissue level)
    fig_box = go.Figure()
    metric_list = ['Accuracy', 'Recall', 'Precision']
    metric_colors = ['#636EFA', "#EEA9FC", "#48CECE"]

    # Build per-metric series
    for metric in metric_list:
        fig_box.add_trace(
            go.Box(
                y=df[metric],
                name=metric,
                marker_color=metric_colors[metric_list.index(metric)],
                boxpoints='all',
                boxmean=True,
                hoverinfo='y+text',
                hovertext=df['tissue']
            )
        )
        # Mean/Median annotations beneath each metric
        mean_value = df[metric].mean()
        median_value = df[metric].median()
        fig_box.add_annotation(
            x=metric,
            yref='paper',
            y=-0.2,
            text=f"Mean: {mean_value:.1f}%" + f"<br>Median: {median_value:.1f}%",
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=metric_colors[metric_list.index(metric)],
            borderwidth=1,
            borderpad=4,
            align='center'
        )

    fig_box.update_layout(
        title_text="Summary of performance metrics across tissues",
        font=dict(size=15),
        height=800,
        width=700,
        margin=dict(l=10, r=10, t=50, b=150)
    )
    fig_box.update_yaxes(title_text='Performance (%)')

    # Optional: grouped bar plot per tissue for quick comparison
    df_bar = df.sort_values(by='Accuracy', ascending=False)
    fig_bar = go.Figure()
    for metric in metric_list:
        fig_bar.add_trace(
            go.Bar(
                x=df_bar['tissue'].str.capitalize(),
                y=df_bar[metric],
                name=metric,
                marker_color=metric_colors[metric_list.index(metric)],
                text=df_bar[metric].round(0),
                textposition='outside'
            )
        )
    fig_bar.update_layout(
        title_text="Tissue-level Accuracy/Recall/Precision",
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1,
        height=500,
        width=1200,
        font=dict(size=14),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1)
    )
    fig_bar.update_xaxes(title_text='Tissue')
    fig_bar.update_yaxes(title_text='Percentage (%)')

    # Save figures
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        save_fig(fig_box, plots_dir, f"{plot_name_prefix}_boxplot", formats=['html','png'], fig_type='plotly')
        save_fig(fig_bar, plots_dir, f"{plot_name_prefix}_barplot", formats=['html','png'], fig_type='plotly')

    if show:
        try:
            fig_box.show()
            fig_bar.show()
        except Exception:
            pass

    return {'box': fig_box, 'bar': fig_bar}


#----------------------------------------------------------------------
# PLOT - TISSUE BOX/BAR FOR ROC/PR/F1
#----------------------------------------------------------------------
def plot_tissue_roc_pr_f1_boxplots(
    roc_auc_df: pd.DataFrame,
    plots_dir: Optional[str] = None,
    plot_name_prefix: str = 'tissue_roc_pr_f1',
    show: bool = False
):
    """Create box plots and grouped bar plots for F1 Score / AUC-ROC / AUC-PR across tissues.

    Expects `roc_auc_df` with columns: ['tissue','cell_line','roc_auc_score','pr_auc_score'] and optionally 'f1_score'.
    Returns a dict with keys {'box': fig_box, 'bar': fig_bar}.
    """
    import plotly.graph_objects as go

    req = {'tissue', 'cell_line'}
    if not req.issubset(set(roc_auc_df.columns)):
        raise ValueError(f"roc_auc_df missing required columns: {req - set(roc_auc_df.columns)}")

    df = roc_auc_df.copy()

    # Normalize metric column names
    rename_map = {}
    for col in df.columns:
        lc = str(col).lower()
        if lc in ('roc_auc', 'auc-roc', 'roc_auc_score'):
            rename_map[col] = 'roc_auc_score'
        if lc in ('pr_auc', 'auc-pr', 'pr_auc_score'):
            rename_map[col] = 'pr_auc_score'
        if lc in ('f1 score', 'f1_score'):
            rename_map[col] = 'f1_score'
    if rename_map:
        df = df.rename(columns=rename_map)

    metrics = []
    labels = []
    colors = []
    # Build available metrics list in consistent order
    if 'f1_score' in df.columns:
        metrics.append('f1_score'); labels.append('F1 Score'); colors.append('#FF6692')
    if 'roc_auc_score' in df.columns:
        metrics.append('roc_auc_score'); labels.append('AUC-ROC'); colors.append('#6EDEDA')
    if 'pr_auc_score' in df.columns:
        metrics.append('pr_auc_score'); labels.append('AUC-PR'); colors.append('#FECB52')

    if not metrics:
        raise ValueError("No ROC/PR/F1 metric columns found in roc_auc_df")

    # Coerce to numeric, drop rows with all metrics missing
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df = df.dropna(subset=metrics, how='all')

    # Box plots
    fig_box = go.Figure()
    for m, label, c in zip(metrics, labels, colors):
        fig_box.add_trace(
            go.Box(
                y=df[m],
                name=label,
                marker_color=c,
                boxpoints='all',
                boxmean=True,
                hoverinfo='y+text',
                hovertext=df.get('tissue', '')
            )
        )
        mean_value = df[m].mean()
        median_value = df[m].median()
        fig_box.add_annotation(
            x=label,
            yref='paper',
            y=-0.2,
            text=f"Mean: {mean_value:.2f}<br>Median: {median_value:.2f}",
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=c,
            borderwidth=1,
            borderpad=4,
            align='center'
        )

    fig_box.update_layout(
        title_text="Summary of ROC/PR/F1 across tissues",
        font=dict(size=15),
        height=800,
        width=700,
        margin=dict(l=10, r=10, t=50, b=150)
    )
    fig_box.update_yaxes(title_text='Score')

    # Grouped bar by tissue (mean per tissue)
    means = df.groupby('tissue')[metrics].mean()
    # Choose sort key: prefer ROC, else first metric present
    sort_key = 'roc_auc_score' if 'roc_auc_score' in means.columns else metrics[0]
    means = means.sort_values(by=sort_key, ascending=False).round(3)

    fig_bar = go.Figure()
    for m, label, c in zip(metrics, labels, colors):
        if m not in means.columns:
            continue
        fig_bar.add_trace(
            go.Bar(
                x=means.index.str.capitalize(),
                y=means[m],
                name=label,
                marker_color=c,
                text=means[m],
                textposition='outside'
            )
        )
    fig_bar.update_layout(
        title_text="Mean F1 / AUC-ROC / AUC-PR by tissue",
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1,
        height=500,
        width=1200,
        font=dict(size=14),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1)
    )
    fig_bar.update_xaxes(title_text='Tissue')
    fig_bar.update_yaxes(title_text='Score')

    # Save figures
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        save_fig(fig_box, plots_dir, f"{plot_name_prefix}_boxplot", formats=['html','png'], fig_type='plotly')
        save_fig(fig_bar, plots_dir, f"{plot_name_prefix}_barplot", formats=['html','png'], fig_type='plotly')

    if show:
        try:
            fig_box.show()
            fig_bar.show()
        except Exception:
            pass

    return {'box': fig_box, 'bar': fig_bar}
