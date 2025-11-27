"""ROC metrics plotting helpers

Follow the project's load -> prepare -> plot workflow. Exposes a single
entrypoint `make_roc_plots(results_dir, plots_dir, show)` which mirrors the
style used by other plotting modules.
"""
import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from ..utils import save_fig
from .load_results import _load_main_results


def _prepare_roc_metrics(results: dict) -> Optional[pd.DataFrame]:
    """Extract and clean `roc_metrics` DataFrame from results dict.

    Returns None when no ROC metrics file is present.
    """
    if not isinstance(results, dict) or 'files' not in results:
        raise ValueError('results must be the dict returned by _load_main_results')

    df = results['files'].get('roc_metrics')
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        raise ValueError('roc_metrics entry must be a pandas DataFrame')

    df = df.copy()
    # Drop any unnamed index / placeholder columns
    unnamed = [c for c in df.columns if str(c).strip() == '']
    if unnamed:
        df = df.drop(columns=unnamed)

    # Ensure expected column exists
    if 'cell_line' not in df.columns:
        raise ValueError('roc_metrics dataframe must contain a `cell_line` column')

    # Set index and coerce numeric types
    df['cell_line'] = df['cell_line'].astype(str)
    df = df.set_index('cell_line')

    numeric_cols = ['roc_auc', 'pr_auc', 'f1_score', 'n_positive', 'n_negative', 'pred_min']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def plot_roc_bar(df: pd.DataFrame, plots_dir: str, plot_name: str = 'roc_pr_f1_bar', width: int = 1200, height: int = 600, show: bool = False):
    df_plot = df[['roc_auc', 'pr_auc', 'f1_score']].copy()
    df_plot = df_plot.dropna(how='all')

    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96']
    for i, col in enumerate(df_plot.columns):
        fig.add_trace(go.Bar(
            x=df_plot.index,
            y=df_plot[col],
            name=col,
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(title='ROC / PR / F1 by cell line', barmode='group', xaxis_tickangle=45, height=height, width=width)

    # Save via helper
    save_fig(fig, plots_dir, plot_name, formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    return fig


def plot_predmin_vs_roc(df: pd.DataFrame, plots_dir: str, plot_name: str = 'predmin_vs_roc', width: int = 900, height: int = 600, show: bool = False):
    if 'pred_min' not in df.columns or 'roc_auc' not in df.columns:
        return None

    scatter = df.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scatter['pred_min'],
        y=scatter['roc_auc'],
        mode='markers+text',
        text=scatter['cell_line'],
        textposition='top center'
    ))
    fig.update_layout(title='pred_min vs roc_auc', xaxis_title='pred_min', yaxis_title='roc_auc', height=height, width=width)

    save_fig(fig, plots_dir, plot_name, formats=['html', 'png'], fig_type='plotly')
    if show:
        fig.show()

    return fig


def make_roc_plots(results_dir: str, plots_dir: Optional[str] = None, show: bool = False):
    results = _load_main_results(results_dir)
    df = _prepare_roc_metrics(results)
    if df is None:
        print('No roc_metrics.csv found in results dir:', results_dir)
        return None

    if plots_dir is None:
        plots_dir = os.path.join(results.get('results_dir', results_dir), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig1 = plot_roc_bar(df, plots_dir, show=show)
    fig2 = plot_predmin_vs_roc(df, plots_dir, show=show)

    return {
        'roc_bar': fig1,
        'predmin_scatter': fig2,
        'roc_metrics_df': df,
        'plots_dir': plots_dir,
    }


__all__ = [
    'make_roc_plots',
]
