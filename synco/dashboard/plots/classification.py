"""
classification.py – Plotly-native classification metric plots for the SYNCO dashboard.

Supported filters: cell_line (by_cell_line), combination (by_combination)
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go

from synco.dashboard.plots._data import (
    METRIC_COLORS,
    apply_filters,
    check_empty,
    load,
    roc_metrics,
    comparison,
    normalise_comparison_df,
    _extract_cell_line_df,
    _extract_combination_df,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_cell_line_df(r: dict) -> pd.DataFrame | None:
    """Return a tidy DataFrame indexed by cell_line with all available metrics."""
    comp = comparison(r)
    roc  = roc_metrics(r)

    cell_df = _extract_cell_line_df(comp)
    if cell_df is None:
        return None

    cell_df = normalise_comparison_df(cell_df, "cell_line")
    cell_df = cell_df.set_index("cell_line")

    # Coerce numeric
    for col in ["Accuracy", "Recall", "Precision"]:
        if col in cell_df.columns:
            cell_df[col] = pd.to_numeric(cell_df[col], errors="coerce")

    # Join F1 / AUC metrics from roc_metrics_df
    if roc is not None:
        roc_cols = {}
        col_map = {
            "f1_score": "F1 Score",
            "roc_auc":  "AUC-ROC",
            "pr_auc":   "AUC-PR",
        }
        for src, dst in col_map.items():
            if src in roc.columns:
                roc_cols[dst] = roc.set_index("cell_line")[src]
        if roc_cols:
            roc_df = pd.DataFrame(roc_cols)
            cell_df = cell_df.join(roc_df, how="left")

    return cell_df.reset_index()


def _horizontal_bar(df: pd.DataFrame, id_col: str, metrics: list[str],
                    title: str, sort_by: str | None = None) -> go.Figure:
    """Build a grouped horizontal bar chart: id_col on y, metrics on x."""
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True)

    fig = go.Figure()
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.isna().all():
            continue
        fig.add_trace(go.Bar(
            x=values,
            y=df[id_col].astype(str),
            name=metric,
            orientation="h",
            marker_color=METRIC_COLORS.get(metric),
            hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Score",
        yaxis_title=id_col.replace("_", " ").title(),
        xaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(300, 30 * len(df) + 120),
        margin=dict(l=180, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_by_cell_line(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Classification metrics (Accuracy/Recall/Precision + F1/AUC) per cell line.

    Filterable by: cell_line
    """
    r   = load(results_dir)
    df  = _build_cell_line_df(r)
    if df is None or df.empty:
        return []

    # Apply cell_line filter
    if filters and filters.get("cell_line"):
        df = df[df["cell_line"] == filters["cell_line"]]
        check_empty(df, "cell_line filter")

    figs: list[go.Figure] = []

    # Figure 1: Accuracy / Recall / Precision
    fig1 = _horizontal_bar(
        df, "cell_line",
        metrics=["Accuracy", "Recall", "Precision"],
        title="Classification Metrics by Cell Line",
        sort_by="Accuracy",
    )
    figs.append(fig1)

    # Figure 2: F1 / AUC-ROC / AUC-PR  (only if at least one non-NaN column)
    auc_metrics = ["F1 Score", "AUC-ROC", "AUC-PR"]
    auc_available = [m for m in auc_metrics if m in df.columns
                     and pd.to_numeric(df[m], errors="coerce").notna().any()]
    if auc_available:
        fig2 = _horizontal_bar(
            df, "cell_line",
            metrics=auc_metrics,
            title="F1 / AUC Metrics by Cell Line",
            sort_by="F1 Score",
        )
        figs.append(fig2)

    return figs


def plot_by_combination(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Classification metrics (Accuracy/Recall/Precision) per drug combination.

    Filterable by: combination
    """
    r    = load(results_dir)
    comp = comparison(r)
    df   = _extract_combination_df(comp)
    if df is None or df.empty:
        return []

    df = normalise_comparison_df(df, "combination")
    # set combination as regular column (not index)
    if "combination" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "combination":
            df = df.rename(columns={df.columns[0]: "combination"})

    for col in ["Accuracy", "Recall", "Precision"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply combination filter
    if filters and filters.get("combination"):
        df = df[df["combination"] == filters["combination"]]
        check_empty(df, "combination filter")

    fig = _horizontal_bar(
        df, "combination",
        metrics=["Accuracy", "Recall", "Precision"],
        title="Classification Metrics by Drug Combination",
        sort_by="Accuracy",
    )
    return [fig]
