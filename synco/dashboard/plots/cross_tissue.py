"""
cross_tissue.py – Plotly cross-tissue aggregate plots for the SYNCO dashboard.

Functions that already return Plotly figures in multi_tissue_summary.py are
called directly. Matplotlib ring plots (plot_tissue_rings, plot_aggregate_ring)
are re-implemented here as Plotly donuts.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synco.dashboard.plots._data import RING_COLORS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tissue directory scanning
# ---------------------------------------------------------------------------

def _scan_tissue_dirs(cell_fate_dir: str) -> list[Path]:
    """Return list of synco_output Paths under *cell_fate_dir*."""
    from synco.dashboard.callbacks.pipeline_cb import _scan_multi_tissue_root
    return _scan_multi_tissue_root(Path(cell_fate_dir))


# ---------------------------------------------------------------------------
# Plotly port: tissue rings
# ---------------------------------------------------------------------------

def _tissue_ring_grid(comparison_df: pd.DataFrame) -> go.Figure:
    """Grid of TP/TN/FP/FN donut subplots, one per tissue."""
    if comparison_df is None or comparison_df.empty:
        return go.Figure()

    df = comparison_df.reset_index() if comparison_df.index.name else comparison_df.copy()
    if "tissue" not in df.columns:
        df.insert(0, "tissue", df.index.astype(str))

    n = len(df)
    n_cols = min(5, n)
    n_rows = math.ceil(n / n_cols)

    col_aliases = {
        "TP": ("TP", "True Positive", "True Positives"),
        "TN": ("TN", "True Negative", "True Negatives"),
        "FP": ("FP", "False Positive", "False Positives"),
        "FN": ("FN", "False Negative", "False Negatives"),
    }

    def _get(row, aliases):
        for a in aliases:
            if a in row.index and not pd.isna(row[a]):
                return float(row[a])
        return 0.0

    acc_vals = pd.to_numeric(df.get("Accuracy", pd.Series([float("nan")] * n)), errors="coerce")
    subtitles = [
        f"{str(df.iloc[i]['tissue'])}<br><sub>Acc={acc_vals.iloc[i]:.2f}</sub>"
        if not pd.isna(acc_vals.iloc[i])
        else str(df.iloc[i]["tissue"])
        for i in range(n)
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
        subplot_titles=subtitles,
    )

    for i, (_, row) in enumerate(df.iterrows()):
        r_idx, c_idx = divmod(i, n_cols)
        tp = _get(row, col_aliases["TP"])
        tn = _get(row, col_aliases["TN"])
        fp = _get(row, col_aliases["FP"])
        fn = _get(row, col_aliases["FN"])

        fig.add_trace(
            go.Pie(
                values=[tp, tn, fp, fn],
                labels=["TP", "TN", "FP", "FN"],
                hole=0.45,
                marker_colors=[RING_COLORS["TP"], RING_COLORS["TN"],
                               RING_COLORS["FP"], RING_COLORS["FN"]],
                name=str(row["tissue"]),
                showlegend=(i == 0),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
                textposition="inside",
            ),
            row=r_idx + 1, col=c_idx + 1,
        )

    fig.update_layout(
        title=dict(text="Performance Rings by Tissue", x=0.5),
        height=220 * n_rows + 80,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )
    return fig


def _aggregate_ring(comparison_df: pd.DataFrame) -> go.Figure:
    """Single nested donut: outer = Match/Mismatch, inner = TP/TN/FP/FN (summed across tissues)."""
    if comparison_df is None or comparison_df.empty:
        return go.Figure()

    col_aliases = {
        "TP": ("TP", "True Positive", "True Positives"),
        "TN": ("TN", "True Negative", "True Negatives"),
        "FP": ("FP", "False Positive", "False Positives"),
        "FN": ("FN", "False Negative", "False Negatives"),
    }

    def _sum(aliases):
        for a in aliases:
            if a in comparison_df.columns:
                return float(pd.to_numeric(comparison_df[a], errors="coerce").fillna(0).sum())
        return 0.0

    tp = _sum(col_aliases["TP"])
    tn = _sum(col_aliases["TN"])
    fp = _sum(col_aliases["FP"])
    fn = _sum(col_aliases["FN"])
    match    = tp + tn
    mismatch = fp + fn
    total    = match + mismatch
    recall   = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    accuracy = match / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[match, mismatch],
        labels=["Match", "Mismatch"],
        hole=0.65,
        marker_colors=[RING_COLORS["Match"], RING_COLORS["Mismatch"]],
        domain={"x": [0, 1], "y": [0, 1]},
        name="Outer",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        showlegend=True,
        textposition="outside",
    ))
    fig.add_trace(go.Pie(
        values=[tp, tn, fp, fn],
        labels=["TP", "TN", "FP", "FN"],
        hole=0.4,
        marker_colors=[RING_COLORS["TP"], RING_COLORS["TN"], RING_COLORS["FP"], RING_COLORS["FN"]],
        domain={"x": [0.175, 0.825], "y": [0.175, 0.825]},
        name="Inner",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        showlegend=True,
        textposition="inside",
    ))

    centre = (
        f"<b>Recall</b><br>{recall:.3f}"
        f"<br><br><b>Accuracy</b><br>{accuracy:.3f}"
        f"<br><br><b>Precision</b><br>{precision:.3f}"
    ) if not math.isnan(recall) else "No data"

    fig.update_layout(
        title=dict(text="Aggregate Performance Ring (All Tissues)", x=0.5),
        annotations=[dict(text=centre, x=0.5, y=0.5, font_size=12,
                          showarrow=False, align="center")],
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_summaries(cell_fate_dir: str):
    from synco.plotting.multi_tissue_summary import load_all_tissue_summaries
    return load_all_tissue_summaries(cell_fate_dir)


def plot_tissue_metric_boxplots(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Accuracy/Recall/Precision box plots aggregated across tissues (already Plotly)."""
    from synco.plotting.multi_tissue_summary import plot_tissue_metric_boxplots as _fn
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    result = _fn(comparison_df, plots_dir=None)
    if isinstance(result, dict):
        return [fig for fig in result.values() if fig is not None]
    if result is not None:
        return [result]
    return []


def plot_tissue_rings(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Grid of TP/TN/FP/FN donut rings per tissue (Plotly re-implementation)."""
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    return [_tissue_ring_grid(comparison_df)]


def plot_aggregate_ring(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Single nested donut ring aggregating all tissues (Plotly re-implementation)."""
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    return [_aggregate_ring(comparison_df)]


def plot_roc_pr_violin(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """ROC/PR AUC violin + table per tissue (already Plotly)."""
    from synco.plotting.multi_tissue_summary import plot_roc_pr_violin as _fn
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []
    fig = _fn(roc_auc_df, plots_dir=None)
    return [fig] if fig is not None else []


def plot_tissue_roc_pr_detail(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Detailed ROC/PR/F1 summary per tissue: box, bar, heatmap (already Plotly)."""
    from synco.plotting.multi_tissue_summary import plot_tissue_roc_pr_f1 as _detail
    comparison_df, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    figs = []
    if roc_auc_df is not None and not roc_auc_df.empty:
        try:
            result = _detail(roc_auc_df, plots_dir=None)
            if isinstance(result, dict):
                figs += [f for f in result.values() if f is not None]
            elif result is not None:
                figs.append(result)
        except Exception:
            logger.exception("plot_tissue_roc_pr_f1 failed")
    return figs


def _iter_tissue_figs(cell_fate_dir: str, fn, filters: dict | None) -> list[go.Figure]:
    """Call *fn(results_dir, filters)* for every tissue directory; collect figures."""
    tissue_dirs = _scan_tissue_dirs(cell_fate_dir)
    all_figs: list[go.Figure] = []
    for td in tissue_dirs:
        tissue_name = td.parent.name
        try:
            figs = fn(str(td), filters=filters)
            for fig in (figs or []):
                # Prepend tissue name to the figure title
                old_title = (fig.layout.title.text or "") if fig.layout.title else ""
                new_title = f"[{tissue_name}] {old_title}" if old_title else f"[{tissue_name}]"
                fig.update_layout(title=dict(text=new_title))
                all_figs.append(fig)
        except Exception:
            logger.exception("Failed to render %s for tissue %s", fn.__name__, tissue_name)
    return all_figs


def plot_exp_distributions_all(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Experimental distributions for every tissue."""
    from synco.dashboard.plots.distributions import plot_experimental
    return _iter_tissue_figs(cell_fate_dir, plot_experimental, filters)


def plot_pred_distributions_all(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Predicted distributions for every tissue."""
    from synco.dashboard.plots.distributions import plot_predicted
    return _iter_tissue_figs(cell_fate_dir, plot_predicted, filters)


def plot_profiles_all(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Profile categories for every tissue."""
    from synco.dashboard.plots.profiles import plot_profile_categories
    return _iter_tissue_figs(cell_fate_dir, plot_profile_categories, filters)
