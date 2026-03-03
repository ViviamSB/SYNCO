"""
classification plotting: load -> process -> plot

This module refactors the original top-level plotting script into three
steps consistent with other plotting helpers:
- _load_classification_inputs(results_dir)
- _process_classification_metrics(roc_df, comparison_df)
- make_classification_plots(results_dir, plots_dir, show=False)

The plotting code is largely preserved; it now runs inside functions and
writes outputs to `plots_dir`.
"""

import os
from typing import Optional, Tuple

import pandas as pd
import re
import plotly.graph_objects as go
import logging

from ..utils import save_fig
from .load_results import _load_main_results


def _load_classification_inputs(results_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[object]]:
    """Load roc_metrics and comparison results from a results directory.

    Returns (roc_metrics_df, comparison_results_df) where either may be None.
    """
    results = _load_main_results(results_dir)
    roc_df = results.get('files', {}).get('roc_metrics')
    comparison = results.get('files', {}).get('comparison')

    return roc_df, comparison


def _extract_comparison_tables(comparison: Optional[object]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Given `comparison` which may be a single DataFrame or a dict of DataFrames,
    return a tuple (cell_line_df, combination_df) where either may be None.
    """
    cell_df = None
    combi_df = None
    if comparison is None:
        return None, None

    if isinstance(comparison, pd.DataFrame):
        return comparison.copy(), None

    if isinstance(comparison, dict):
        first_assigned = 0
        for k, v in comparison.items():
            if not isinstance(v, pd.DataFrame):
                continue
            key = str(k).lower()
            if key.startswith('cell'):
                cell_df = v.copy()
            elif key.startswith('inhibitor'):
                combi_df = v.copy()
            else:
                # fallback ordering: first -> cell, second -> combi
                if cell_df is None:
                    cell_df = v.copy()
                elif combi_df is None:
                    combi_df = v.copy()
            first_assigned += 1
        return cell_df, combi_df

    # Unknown type -> return None
    return None, None


def _prepare_combi_df(comparison_comb_df: pd.DataFrame) -> pd.DataFrame:
    """Build combi_match_df by using the provided comparison DataFrame only.

    - sets the DataFrame index to the combination labels (tries to detect
      an appropriate column when the first column has no header),
    - coerces existing 'Accuracy','Recall','Precision' columns to numeric
      (stripping '%' and commas and converting fractions to percentages),
    - fills missing metric values with 0 so plotting functions won't fail.
    """
    df = comparison_comb_df.copy()

    # If the first column was written as an unnamed index (e.g. '' or
    # 'Unnamed: 0'), rename it to 'combination' and set it as the index.
    if isinstance(df.index, pd.RangeIndex) and (not df.columns.empty):
        first_col = df.columns[0]
        fname = str(first_col)
        if fname.strip() == '' or fname.lower().startswith('unnamed'):
            df = df.rename(columns={first_col: 'combination'})
            df['combination'] = df['combination'].astype(str)
            df = df.set_index('combination')
        else:
            df.index = df.index.astype(str)
    else:
        df.index = df.index.astype(str)

    # Return only the index and the expected metric columns. The data is
    # assumed to already be in the correct format.
    return df[['Accuracy', 'Recall', 'Precision']]


def _prepare_cell_df(roc_df: Optional[pd.DataFrame], comparison_cell_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build `classification_metrics_df` from inputs.

    Returns a DataFrame indexed by `cell_line` with columns
    ['Accuracy','Recall','Precision','F1 Score','AUC-ROC','AUC-PR'].
    """
    # Assume both `comparison_cell_df` and `roc_df` are present. Use the
    # comparison table as the base and append roc metrics (F1, AUC-ROC, AUC-PR).
    classification_metrics_df = comparison_cell_df.copy()

    # If the DataFrame was written to CSV with the index as an unnamed
    # first column, move that column back to the index named 'cell_line'.
    if isinstance(classification_metrics_df.index, pd.RangeIndex) and (not classification_metrics_df.columns.empty):
        first_col = classification_metrics_df.columns[0]
        fname = str(first_col)
        if fname.strip() == '' or fname.lower().startswith('unnamed'):
            classification_metrics_df = classification_metrics_df.rename(columns={first_col: 'cell_line'})
            classification_metrics_df['cell_line'] = classification_metrics_df['cell_line'].astype(str)
            classification_metrics_df = classification_metrics_df.set_index('cell_line')
        else:
            classification_metrics_df.index = classification_metrics_df.index.astype(str)
    else:
        classification_metrics_df.index = classification_metrics_df.index.astype(str)

    # Normalize cell-line names for mapping against roc_df and build a
    # lookup dictionary from roc_df for fast assignment.
    def _norm(s: object) -> str:
        if pd.isna(s):
            return ''
        return re.sub(r'[^A-Za-z0-9]', '', str(s).strip()).upper()

    roc_tmp = roc_df.copy()
    roc_tmp['cell_line'] = roc_tmp['cell_line'].astype(str)
    roc_map = {}
    for _, row in roc_tmp.iterrows():
        key = _norm(row.get('cell_line'))
        roc_map.setdefault(key, {})
        roc_map[key]['F1 Score'] = row.get('f1_score') if 'f1_score' in roc_tmp.columns else row.get('F1 Score')
        roc_map[key]['AUC-ROC'] = row.get('roc_auc') if 'roc_auc' in roc_tmp.columns else row.get('AUC-ROC')
        roc_map[key]['AUC-PR'] = row.get('pr_auc') if 'pr_auc' in roc_tmp.columns else row.get('AUC-PR')

    # Map ROC metrics into the classification table, preserving index order.
    mapped_f1 = []
    mapped_roc = []
    mapped_pr = []
    for idx in classification_metrics_df.index.astype(str):
        key = _norm(idx)
        m = roc_map.get(key, {})
        mapped_f1.append(m.get('F1 Score', pd.NA))
        mapped_roc.append(m.get('AUC-ROC', pd.NA))
        mapped_pr.append(m.get('AUC-PR', pd.NA))
    classification_metrics_df['F1 Score'] = mapped_f1
    classification_metrics_df['AUC-ROC'] = mapped_roc
    classification_metrics_df['AUC-PR'] = mapped_pr

    # Ensure Accuracy/Recall/Precision exist; keep ROC/F1 columns as added.
    for c in ['Accuracy', 'Recall', 'Precision']:
        if c not in classification_metrics_df.columns:
            classification_metrics_df[c] = pd.NA

    # Light coercion to numeric for plotting convenience
    for c in ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC-ROC', 'AUC-PR']:
        if c in classification_metrics_df.columns:
            classification_metrics_df[c] = pd.to_numeric(classification_metrics_df[c], errors='coerce')

    # Convert fractional (0..1) Accuracy/Recall/Precision to percentages if needed
    try:
        subset = classification_metrics_df[['Accuracy', 'Recall', 'Precision']]
        if not subset.empty:
            max_val = subset.max().max()
            if pd.notna(max_val) and max_val <= 1.01:
                classification_metrics_df[['Accuracy', 'Recall', 'Precision']] = classification_metrics_df[['Accuracy', 'Recall', 'Precision']] * 100
    except Exception:
        pass

    # Fill NaNs in the three main metrics with zeros so plots render safely
    classification_metrics_df[['Accuracy', 'Recall', 'Precision']] = classification_metrics_df[['Accuracy', 'Recall', 'Precision']].fillna(0)

    return classification_metrics_df


def _plot_by_cell_line(classification_metrics_df: pd.DataFrame, plots_path: str, show: bool = False, return_fig: bool = False):
    """Generate the BY CELL LINE plots and save them to `plots_path`."""
    plots_path = os.path.join(plots_path, '')
    metric_list1 = ['Accuracy', 'Recall', 'Precision']
    metric_list2 = ['F1 Score', 'AUC-ROC', 'AUC-PR']
    metric_colors1 = ['#636EFA', "#EEA9FC", "#48CECE"]
    metric_colors2 = ['#FF6692', "#6EDEDA", '#FECB52']

    classification_metrics_scaled_df = classification_metrics_df.copy()
    figs = []

    # BAR - Accuracy/Recall/Precision
    fig = go.Figure()
    classification_metrics_scaled_df = classification_metrics_scaled_df.sort_values(by=metric_list1[0], ascending=False)
    for metric in metric_list1:
        fig.add_trace(go.Bar(x=classification_metrics_scaled_df.index, y=classification_metrics_scaled_df[metric], name=metric,
                             marker_color=metric_colors1[metric_list1.index(metric)], text=classification_metrics_scaled_df[metric].round(0), textposition='outside'))
    fig.update_layout(title_text="Classification Metrics by cell line", barmode='group', bargap=0.3, bargroupgap=0.1, height=450, width=1400, font=dict(size=14), margin=dict(l=10, r=10, t=40, b=10))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1, font=dict(size=14)))
    fig.update_xaxes(title_text='Cell Line')
    fig.update_yaxes(title_text='Percentage (%)')
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, f"sum_{metric_list1}_barplot", formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    # HEATMAP
    cm_df = classification_metrics_df.sort_values(by='Accuracy', ascending=True)
    fig = go.Figure(data=go.Heatmap(z=cm_df[['Accuracy', 'Recall', 'Precision']].values, x=['Accuracy', 'Recall', 'Precision'], y=cm_df.index, colorscale='RdBu', colorbar=dict(title='Score'), zmin=0, zmax=100))
    fig.update_layout(title_text="Classification metrics by cell line", height=600, width=500, font=dict(size=14), margin=dict(l=150, r=50, t=100, b=50))
    if show:
        fig.show()
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, 'heatmap_classification_metrics', formats=['html', 'png'], fig_type='plotly')

    # BAR - F1/AUC
    fig = go.Figure()
    classification_metrics_scaled_df = classification_metrics_scaled_df.sort_values(by=metric_list2[0], ascending=False)
    for metric in metric_list2:
        fig.add_trace(go.Bar(x=classification_metrics_scaled_df.index, y=classification_metrics_scaled_df[metric], name=metric, marker_color=metric_colors2[metric_list2.index(metric)], text=classification_metrics_scaled_df[metric].round(1), textposition='outside'))
    fig.update_layout(title_text=f"Summary of classification metrics by cell line", barmode='group', bargap=0.3, bargroupgap=0.1, height=500, width=1500)
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, f"sum_{metric_list2}_barplot", formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    # HEATMAP - F1/AUC (exclude cell lines with missing metric values)
    cm_df_f1 = classification_metrics_df.dropna(subset=['F1 Score', 'AUC-ROC', 'AUC-PR']).sort_values(by='F1 Score', ascending=True)
    if not cm_df_f1.empty:
        fig = go.Figure(data=go.Heatmap(z=cm_df_f1[['F1 Score', 'AUC-ROC', 'AUC-PR']].values, x=['F1 Score', 'AUC-ROC', 'AUC-PR'], y=cm_df_f1.index, colorscale='RdBu', colorbar=dict(title='Score'), zmin=0, zmax=1))
        fig.update_layout(title_text="F1/AUC metrics by cell line", height=600, width=500, font=dict(size=14), margin=dict(l=150, r=50, t=100, b=50))
        if show:
            fig.show()
        if return_fig:
            figs.append((fig, 'plotly'))
        else:
            save_fig(fig, plots_path, 'heatmap_f1_auc_metrics', formats=['html', 'png'], fig_type='plotly')

    # BOX PLOTS for Accuracy/Recall/Precision
    fig = go.Figure()
    classification_metrics_scaled_df = classification_metrics_scaled_df.sort_values(by=metric_list1[0], ascending=False)
    for metric in metric_list1:
        fig.add_trace(go.Box(y=classification_metrics_scaled_df[metric], name=metric, marker_color=metric_colors1[metric_list1.index(metric)], boxpoints='all', boxmean=True, hoverinfo='y+text', hovertext=classification_metrics_scaled_df[metric].index))
    for metric in metric_list1:
        mean_value = classification_metrics_scaled_df[metric].mean().round(0)
        median_value = classification_metrics_scaled_df[metric].median().round(0)
        std_value = classification_metrics_scaled_df[metric].std().round(2)
        n_cell_lines = classification_metrics_scaled_df[metric].notna().sum()
        fig.add_annotation(x=metric, yref='paper', y=-0.08, text=f"n ={n_cell_lines}", showarrow=False, font=dict(size=13, color='#666666'), align='center')
        fig.add_annotation(x=metric, yref='paper', y=-0.2, text=f"Mean: {mean_value:.0f}%" + f"<br>Median: {median_value:.0f}%" + f"<br>Std: {std_value:.2f}", showarrow=False, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor=metric_colors1[metric_list1.index(metric)], borderwidth=1, borderpad=4, align='center')
    fig.update_layout(title_text="Summary of performance metrics across cell lines",
        font=dict(size=15), height=800, width=600, margin=dict(l=10, r=10, t=50, b=150))
    fig.update_yaxes(title_text='Performance (%)')
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, 'box_acuracy', formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    # BOX PLOTS for F1/AUC
    fig = go.Figure()
    classification_metrics_scaled_df = classification_metrics_scaled_df.sort_values(by=metric_list2[0], ascending=False)
    for metric in metric_list2:
        # Count valid (non-NaN) data points for this metric
        n_cell_lines = classification_metrics_scaled_df[metric].notna().sum()
        fig.add_trace(go.Box(y=classification_metrics_scaled_df[metric], name=metric, marker_color=metric_colors2[metric_list2.index(metric)], boxpoints='all', boxmean=True, hovertext=classification_metrics_scaled_df[metric].index, hoverinfo='y+text'))
    for metric in metric_list2:
        mean_value = classification_metrics_scaled_df[metric].mean()
        median_value = classification_metrics_scaled_df[metric].median()
        std_value = classification_metrics_scaled_df[metric].std()

        fig.add_annotation(x=metric, yref='paper', y=-0.08, text=f"n ={n_cell_lines}", showarrow=False, font=dict(size=13, color='#666666'), align='center')
        fig.add_annotation(x=metric, y=-0.2, yref='paper', text=f"Mean: {mean_value:.2f}" + f"<br>Median: {median_value:.2f}" + f"<br>Std: {std_value:.2f}", showarrow=False, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor=metric_colors2[metric_list2.index(metric)], borderwidth=1, borderpad=4, align='center')
    fig.update_layout(title_text=f"Summary of classification metrics", font=dict(size=15), height=800, width=600, margin=dict(t=50, b=150, l=10, r=10))
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, 'box_AUCscores', formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    if return_fig:
        return figs


def _plot_by_combination(combi_match_df: pd.DataFrame, plots_path: str, show: bool = False, return_fig: bool = False):
    """Generate BY COMBINATION plots given `combi_match_df`."""
    plots_path = os.path.join(plots_path, '')
    metric_list1 = ['Accuracy', 'Recall', 'Precision']
    metric_colors1 = ['#636EFA', "#EEA9FC", "#48CECE"]
    figs = []

    classification_metrics_combi_scaled_df = combi_match_df.copy().sort_values(by=metric_list1[0], ascending=False)
    fig = go.Figure()
    for metric in metric_list1:
        fig.add_trace(go.Bar(y=classification_metrics_combi_scaled_df[metric], x=classification_metrics_combi_scaled_df.index, name=metric, orientation='v', marker_color=metric_colors1[metric_list1.index(metric)], text=classification_metrics_combi_scaled_df[metric].round(0), textposition='outside'))
    fig.update_layout(title_text="Classification Metrics by inhibitor group combination", barmode='group', bargap=0.3, bargroupgap=0.1, height=500, width=1500, font=dict(size=14))
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, f"sum_{metric_list1}_barplot_combination", formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    # Heatmap for combinations
    classification_metrics_combi_scaled_df = classification_metrics_combi_scaled_df.sort_values(by='Accuracy', ascending=True)
    fig = go.Figure(data=go.Heatmap(
        z=classification_metrics_combi_scaled_df[['Accuracy', 'Recall', 'Precision']].values,
        x=['Accuracy', 'Recall', 'Precision'],
        y=classification_metrics_combi_scaled_df.index,
        colorscale='RdBu',
        colorbar=dict(title='Score'),
        zmin=0,
        zmax=100
    ))
    fig.update_layout(title_text="Predictive performance by combination", height=600, width=500, font=dict(size=14), margin=dict(l=150, r=50, t=100, b=50))
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, 'heatmap_classification_metrics_combi', formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    # Box plots for combinations
    fig = go.Figure()
    for metric in metric_list1:
        mean_value = classification_metrics_combi_scaled_df[metric].mean()
        median_value = classification_metrics_combi_scaled_df[metric].median()
        std_value = classification_metrics_combi_scaled_df[metric].std()
        fig.add_trace(go.Box(y=classification_metrics_combi_scaled_df[metric], name=metric, marker_color=metric_colors1[metric_list1.index(metric)], boxpoints='all', boxmean=True, hoverinfo='y+text', hovertext=classification_metrics_combi_scaled_df[metric].index))
        fig.add_annotation(x=metric, yref='paper', y=-0.2, text=f"Mean: {mean_value:.0f}%" + f"<br>Median: {median_value:.0f}%" + f"<br>Std: {std_value:.2f}", showarrow=False, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor=metric_colors1[metric_list1.index(metric)], borderwidth=1, borderpad=4, align='center')
    fig.update_layout(title_text="Summary of metrics across combinations",
        font=dict(size=15), height=800, width=650, margin=dict(l=10, r=10, t=50, b=150))
    fig.update_yaxes(title_text='Performance (%)')
    if return_fig:
        figs.append((fig, 'plotly'))
    else:
        save_fig(fig, plots_path, 'box_acuracy_combination', formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    if return_fig:
        return figs


def make_classification_plots(results_dir: str, plots_dir: Optional[str] = None, show: bool = False, analysis_type: Optional[str] = None, debug: bool = False, return_fig: bool = False):
    """High-level entrypoint: load -> process -> plot classification metrics.

    - `results_dir`: folder containing `roc_metrics_df.csv` and comparison results
    - `plots_dir`: output folder for plots (defaults to results_dir/plots)
    """
    roc_df, comparison = _load_classification_inputs(results_dir)

    # Extract candidate comparison tables (handles single DataFrame or mapping)
    comp_cell_df, comp_comb_df = _extract_comparison_tables(comparison)

    mode = analysis_type.lower() if analysis_type is not None else None

    if mode in ('combination', 'combi', 'comb', 'inhibitor_group', 'inhibitor_combination'):
        # Combination-only: build combi_match_df from the combination table
        classification_metrics_df = None
        combi_match_df = _prepare_combi_df(comp_comb_df) if isinstance(comp_comb_df, pd.DataFrame) else pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision'])

    elif mode in ('cell_line', 'cellline', 'cell', 'cell_lines'):
        # Cell-line only: build cell-line metrics (may use roc_df)
        classification_metrics_df = _prepare_cell_df(roc_df, comp_cell_df)
        combi_match_df = None
    else:
        # Auto-detect: build both if possible
        classification_metrics_df = _prepare_cell_df(roc_df, comp_cell_df) if isinstance(comp_cell_df, pd.DataFrame) else None
        combi_match_df = _prepare_combi_df(comp_comb_df) if isinstance(comp_comb_df, pd.DataFrame) else None

    if debug:
        logger = logging.getLogger(__name__)
        if classification_metrics_df is None:
            print('\n[classification] DEBUG: classification_metrics_df is None')
            logger.info('classification_metrics_df is None')
        else:
            print('\n[classification] DEBUG: classification_metrics_df shape:', getattr(classification_metrics_df, 'shape', None))
            try:
                print(classification_metrics_df.head(10).to_string())
            except Exception:
                print(classification_metrics_df.head(10))
            print('\n[classification] DEBUG: classification_metrics_df dtypes:')
            try:
                print(classification_metrics_df.dtypes.to_string())
            except Exception:
                print(classification_metrics_df.dtypes)
            logger.info('classification_metrics_df shape=%s', getattr(classification_metrics_df, 'shape', None))
            logger.info('classification_metrics_df columns=%s', list(classification_metrics_df.columns))

        if combi_match_df is None:
            print('\n[classification] DEBUG: combi_match_df is None')
            logger.info('combi_match_df is None')
        else:
            print('\n[classification] DEBUG: combi_match_df shape:', getattr(combi_match_df, 'shape', None))
            try:
                print(combi_match_df.head(10).to_string())
            except Exception:
                print(combi_match_df.head(10))
            print('\n[classification] DEBUG: combi_match_df dtypes:')
            try:
                print(combi_match_df.dtypes.to_string())
            except Exception:
                print(combi_match_df.dtypes)
            logger.info('combi_match_df shape=%s', getattr(combi_match_df, 'shape', None))
            logger.info('combi_match_df columns=%s', list(combi_match_df.columns))

    if not return_fig:
        if plots_dir is None:
            plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

    # Decide which plots to make based on `analysis_type` or available files
    want_cell = False
    want_comb = False
    if analysis_type is None:
        want_cell = classification_metrics_df is not None and not classification_metrics_df.empty
        want_comb = combi_match_df is not None and not combi_match_df.empty
    else:
        at = analysis_type.lower()
        if at in ('cell_line', 'cellline', 'cell', 'cell_lines'):
            want_cell = True
        if at in ('combination', 'combi', 'comb', 'inhibitor_group', 'inhibitor_combination'):
            want_comb = True

    figs = []
    if want_cell:
        cell_figs = _plot_by_cell_line(classification_metrics_df, plots_dir, show=show, return_fig=return_fig)
        if return_fig and cell_figs:
            figs.extend(cell_figs)
    if want_comb:
        combi_figs = _plot_by_combination(combi_match_df, plots_dir, show=show, return_fig=return_fig)
        if return_fig and combi_figs:
            figs.extend(combi_figs)

    if return_fig:
        return figs

    return {
        'classification_metrics_df': classification_metrics_df,
        'combi_match_df': combi_match_df,
        'plots_dir': plots_dir
    }
    



