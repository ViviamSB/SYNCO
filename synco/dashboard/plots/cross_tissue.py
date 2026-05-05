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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

_CT_RING_COLORS = {
    "Match":    "royalblue",
    "Mismatch": "#D94602",
    "TP":       "#458cff",
    "TN":       "#6db0ff",
    "FP":       "#FA7F2E",
    "FN":       "#FDAA65",
}


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
    """Grid of TP/TN/FP/FN donut subplots, one per tissue.

    Counter-clockwise order: Match data (TP, TN) first then Mismatch (FP, FN).
    """
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
                direction="counterclockwise",
                sort=False,
                rotation=90,
                marker_colors=[
                    _CT_RING_COLORS["TP"], _CT_RING_COLORS["TN"],
                    _CT_RING_COLORS["FP"], _CT_RING_COLORS["FN"],
                ],
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
    """Single nested donut: outer = Match/Mismatch, inner = TP/TN/FP/FN.

    Both rings are centered. Counter-clockwise order with Match (TP, TN) first.
    Metrics are annotated in the center hole.
    """
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
    match       = tp + tn
    mismatch    = fp + fn
    total       = match + mismatch
    recall      = tp / (tp + fn)       if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp)       if (tn + fp) > 0 else float("nan")
    accuracy    = match / total        if total > 0     else float("nan")
    precision   = tp / (tp + fp)       if (tp + fp) > 0 else float("nan")
    bal_acc     = (recall + specificity) / 2 \
        if not (math.isnan(recall) or math.isnan(specificity)) else float("nan")

    def _fmt(v):
        return f"{v:.3f}" if not math.isnan(v) else "–"

    fig = go.Figure()
    # Outer ring — centered, counter-clockwise
    fig.add_trace(go.Pie(
        values=[match, mismatch],
        labels=["Match", "Mismatch"],
        hole=0.65,
        marker_colors=[_CT_RING_COLORS["Match"], _CT_RING_COLORS["Mismatch"]],
        domain={"x": [0.05, 0.95], "y": [0.05, 0.95]},
        name="Outer",
        direction="counterclockwise",
        sort=False,
        rotation=90,
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        showlegend=True,
        textposition="outside",
    ))
    # Inner ring — centered within outer, same direction
    fig.add_trace(go.Pie(
        values=[tp, tn, fp, fn],
        labels=["TP", "TN", "FP", "FN"],
        hole=0.4,
        marker_colors=[
            _CT_RING_COLORS["TP"], _CT_RING_COLORS["TN"],
            _CT_RING_COLORS["FP"], _CT_RING_COLORS["FN"],
        ],
        domain={"x": [0.28, 0.72], "y": [0.23, 0.77]},
        name="Inner",
        direction="counterclockwise",
        sort=False,
        rotation=90,
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        showlegend=True,
        textposition="inside",
    ))

    # Metrics annotation in the center hole
    center_text = (
        f"<b>Acc</b>: {_fmt(accuracy)}"
        f"<br><b>Rec</b>: {_fmt(recall)}"
        f"<br><b>Prec</b>: {_fmt(precision)}"
        f"<br><b>Bal</b>: {_fmt(bal_acc)}"
    ) if not math.isnan(accuracy) else "No data"

    fig.update_layout(
        title=dict(text="Aggregate Performance Ring (All Tissues)", x=0.5),
        annotations=[dict(
            text=center_text,
            x=0.5, y=0.5,
            font_size=11,
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
        )],
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Summary data loader
# ---------------------------------------------------------------------------

def _load_summaries(cell_fate_dir: str):
    from synco.plotting.multi_tissue_summary import load_all_tissue_summaries
    return load_all_tissue_summaries(cell_fate_dir)


# ---------------------------------------------------------------------------
# Cross-tissue experimental data loader (all tissues combined)
# ---------------------------------------------------------------------------

def _load_all_experimental(cell_fate_dir: str) -> pd.DataFrame:
    """Concatenate experimental DataFrames from all tissue directories.

    Adds a ``tissue`` column containing the tissue folder name.
    """
    from synco.plotting.load_results import _load_main_results
    tissue_dirs = _scan_tissue_dirs(cell_fate_dir)
    frames = []
    for td in tissue_dirs:
        tissue_name = td.parent.name
        try:
            r = _load_main_results(str(td))
            df = r["files"].get("experimental")
            if df is not None and not df.empty:
                df = df.copy()
                df["tissue"] = tissue_name
                frames.append(df)
        except Exception:
            logger.warning("Could not load experimental data for tissue '%s'", tissue_name)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Cross-tissue cell-line classification data loader
# ---------------------------------------------------------------------------

def _load_all_cell_line_cls(cell_fate_dir: str) -> pd.DataFrame:
    """Concatenate cell_line_comparison_results.csv from all tissue directories.

    Adds a ``tissue`` column. Returns DataFrame with columns including
    cell_line, Accuracy, Recall, Precision (and tissue).
    """
    tissue_dirs = _scan_tissue_dirs(cell_fate_dir)
    frames = []
    for td in tissue_dirs:
        tissue_name = td.parent.name
        csv_path = Path(td) / "cell_line_comparison_results.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df["tissue"] = tissue_name
            frames.append(df)
        except Exception:
            logger.warning("Could not load cell-line comparison for tissue '%s'", tissue_name)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Classification split plots – tissue-level (1.a box, 1.b violin, 1.c bar)
# ---------------------------------------------------------------------------

def plot_tissue_cls_boxes(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Accuracy / Recall / Precision box plots across tissues (tissue-level data points)."""
    from synco.plotting.multi_tissue_summary import plot_tissue_metric_boxplots as _fn
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    result = _fn(comparison_df, plots_dir=None)
    if isinstance(result, dict):
        fig = result.get("box")
        return [fig] if fig is not None else []
    return [result] if result is not None else []


def plot_tissue_cls_violins(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Accuracy / Recall / Precision violin plots across tissues (tissue-level data points)."""
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []

    df = comparison_df.copy()
    if "tissue" not in df.columns:
        df.insert(0, "tissue", df.index.astype(str))

    metrics = [m for m in ("Accuracy", "Recall", "Precision") if m in df.columns]
    if not metrics:
        return []

    palette = {"Accuracy": "#636EFA", "Recall": "#EF553B", "Precision": "#00CC96"}
    fig = go.Figure()
    tissues = sorted(df["tissue"].unique())
    for metric in metrics:
        color = palette.get(metric, "#aaa")
        for tissue in tissues:
            sub = pd.to_numeric(df.loc[df["tissue"] == tissue, metric], errors="coerce").dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Violin(
                y=sub,
                x=[tissue] * len(sub),
                name=metric,
                legendgroup=metric,
                showlegend=(tissue == tissues[0]),
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.7,
                line_color="black",
            ))
    fig.update_layout(
        title=dict(text="Metric Violin Plots — Cross-Tissue (tissue-level)", x=0.5),
        yaxis_title="Score",
        xaxis_title="Tissue",
        violinmode="group",
        height=420,
        margin=dict(l=40, r=20, t=60, b=80),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return [fig]


def plot_tissue_cls_bars(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Accuracy / Recall / Precision bar plots across tissues."""
    from synco.plotting.multi_tissue_summary import plot_tissue_metric_boxplots as _fn
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    result = _fn(comparison_df, plots_dir=None)
    if isinstance(result, dict):
        fig = result.get("bar")
        return [fig] if fig is not None else []
    return []


# ---------------------------------------------------------------------------
# Classification split plots – cell-line level (cross-cell-lines view)
# ---------------------------------------------------------------------------

def plot_tissue_cls_boxes_cl(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Accuracy / Recall / Precision box plots with individual cell lines as data points."""
    df_all = _load_all_cell_line_cls(cell_fate_dir)
    if df_all.empty:
        return []

    metrics = [m for m in ("Accuracy", "Recall", "Precision") if m in df_all.columns]
    if not metrics:
        return []

    # Normalise percentage columns to 0–1 range if stored as percentages
    for m in metrics:
        col = pd.to_numeric(df_all[m], errors="coerce")
        df_all[m] = col / 100.0 if col.max(skipna=True) > 1.5 else col

    df_melt = df_all.melt(
        id_vars=["tissue"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    ).dropna(subset=["Score"])

    palette = {"Accuracy": "#636EFA", "Recall": "#EF553B", "Precision": "#00CC96"}

    fig = go.Figure()
    for metric in metrics:
        sub = df_melt[df_melt["Metric"] == metric]
        fig.add_trace(go.Box(
            x=sub["tissue"],
            y=sub["Score"],
            name=metric,
            marker_color=palette.get(metric, "#aaa"),
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ))
    fig.update_layout(
        title=dict(text="Metric Box Plots — Cross-Cell-Lines (cell-line data points)", x=0.5),
        yaxis_title="Score",
        xaxis_title="Tissue",
        boxmode="group",
        height=420,
        margin=dict(l=40, r=20, t=60, b=80),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return [fig]


def plot_tissue_cls_violins_cl(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Accuracy / Recall / Precision violin plots with individual cell lines as data points."""
    df_all = _load_all_cell_line_cls(cell_fate_dir)
    if df_all.empty:
        return []

    metrics = [m for m in ("Accuracy", "Recall", "Precision") if m in df_all.columns]
    if not metrics:
        return []

    for m in metrics:
        col = pd.to_numeric(df_all[m], errors="coerce")
        df_all[m] = col / 100.0 if col.max(skipna=True) > 1.5 else col

    palette = {"Accuracy": "#636EFA", "Recall": "#EF553B", "Precision": "#00CC96"}
    tissues = sorted(df_all["tissue"].unique())

    fig = go.Figure()
    for metric in metrics:
        color = palette.get(metric, "#aaa")
        for tissue in tissues:
            sub = pd.to_numeric(df_all.loc[df_all["tissue"] == tissue, metric], errors="coerce").dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Violin(
                y=sub,
                x=[tissue] * len(sub),
                name=metric,
                legendgroup=metric,
                showlegend=(tissue == tissues[0]),
                box_visible=True,
                meanline_visible=True,
                points="all",
                fillcolor=color,
                opacity=0.65,
                line_color="black",
            ))
    fig.update_layout(
        title=dict(text="Metric Violin Plots — Cross-Cell-Lines (cell-line data points)", x=0.5),
        yaxis_title="Score",
        xaxis_title="Tissue",
        violinmode="group",
        height=450,
        margin=dict(l=40, r=20, t=60, b=80),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return [fig]


# ---------------------------------------------------------------------------
# Performance ring public functions
# ---------------------------------------------------------------------------

def plot_tissue_rings(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Grid of TP/TN/FP/FN donut rings per tissue."""
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    return [_tissue_ring_grid(comparison_df)]


def plot_aggregate_ring(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Single nested donut ring aggregating all tissues."""
    comparison_df, _, _ = _load_summaries(cell_fate_dir)
    if comparison_df is None or comparison_df.empty:
        return []
    return [_aggregate_ring(comparison_df)]


# ---------------------------------------------------------------------------
# ROC/PR split plots
# ---------------------------------------------------------------------------

def _make_metric_violin(roc_auc_df: pd.DataFrame, metric_col: str,
                        title: str, color: str) -> go.Figure:
    """Per-tissue violin for a single AUC metric column."""
    df = roc_auc_df.dropna(subset=[metric_col]).copy()
    if df.empty:
        return go.Figure()

    if "tissue" not in df.columns:
        df["tissue"] = "all"
    tissues = sorted(df["tissue"].unique())

    fig = go.Figure()
    for tissue in tissues:
        sub = df[df["tissue"] == tissue][metric_col]
        fig.add_trace(go.Violin(
            y=sub,
            x=[tissue] * len(sub),
            name=tissue,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.7,
            line_color="black",
            showlegend=False,
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title=metric_col.replace("_", " ").title(),
        xaxis_title="Tissue",
        height=380,
        margin=dict(l=40, r=20, t=60, b=80),
        template="plotly_white",
    )
    return fig


def plot_roc_pr_boxes_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """F1 / AUC-ROC / AUC-PR box plots across tissues."""
    from synco.plotting.multi_tissue_summary import plot_tissue_roc_pr_f1 as _detail
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []
    try:
        result = _detail(roc_auc_df, plots_dir=None)
        if isinstance(result, dict):
            fig = result.get("box")
            return [fig] if fig is not None else []
        return [result] if result is not None else []
    except Exception:
        logger.exception("plot_roc_pr_boxes_ct failed")
        return []


def plot_roc_pr_metric_violins_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """F1 / AUC-ROC / AUC-PR aggregated violin plots — Cross-Tissue view.

    Shows 3 violins (one per metric). Each data point is a tissue-level mean,
    so the distribution captures variability *across tissues*.
    """
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []

    df = roc_auc_df.copy()
    if "tissue" not in df.columns:
        df["tissue"] = "all"

    metric_conf = [
        ("f1_score",      "F1 Score", "#8B687F"),
        ("roc_auc_score", "AUC-ROC",  "#00B8AB"),
        ("pr_auc_score",  "AUC-PR",   "#048B8B"),
    ]
    present = [(col, label, color) for col, label, color in metric_conf if col in df.columns]
    if not present:
        return []

    # Aggregate to one value per tissue per metric
    agg = df.groupby("tissue")[[col for col, _, _ in present]].mean().reset_index()

    fig = go.Figure()
    for col, label, color in present:
        vals = pd.to_numeric(agg[col], errors="coerce").dropna()
        fig.add_trace(go.Violin(
            y=vals,
            x=[label] * len(vals),
            name=label,
            box_visible=True,
            meanline_visible=True,
            points="all",
            fillcolor=color,
            opacity=0.6,
            line_color="#03312E",
            hovertext=agg.loc[vals.index, "tissue"],
            hovertemplate="<b>%{hovertext}</b><br>Score: %{y:.3f}<extra></extra>",
        ))

    # Annotation of n, mean, median, std in a box at the bottom of the plot
    annotations = []
    for col, label, color in present:
        vals = pd.to_numeric(agg[col], errors="coerce").dropna()
        n = len(vals)
        mean = vals.mean()
        median = vals.median()
        std = vals.std()
        annotations.append(
            f"<b>{label}</b><br>n={n}<br>mean={mean:.3f}<br>median={median:.3f}<br>std={std:.3f}"
        )

    # Add annotation for each metric, positioned below the x-axis, centered under the corresponding violin
    for i, annotation in enumerate(annotations):
        fig.add_annotation(
            text=annotation,
            x=i, y=-0.3,
            xref="x", yref="paper",
            showarrow=False,
            align="center",
            font=dict(size=11, color="#333"),
        )
    fig.update_layout(
        title=dict(text="ROC / PR / F1 Score Distribution — Cross-Tissue", x=0.5),
        yaxis_title="Score",
        # xaxis_title="Metric",
        height=500,
        margin=dict(l=40, r=20, t=60, b=100),
        # template="plotly_white",
        showlegend=False,
    )
    return [fig]


def plot_roc_pr_metric_violins_cl_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """F1 / AUC-ROC / AUC-PR aggregated violin plots — Cross-Cell-Lines view.

    Shows 3 violins (one per metric). Each data point is an individual cell line
    across all tissues, so the distribution captures variability *across cell lines*.
    """
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []

    df = roc_auc_df.copy()
    if "tissue" not in df.columns:
        df["tissue"] = "all"

    metric_conf = [
        ("f1_score",      "F1 Score", "#5DCDFA"),
        ("roc_auc_score", "AUC-ROC",  "#00B8AB"),
        ("pr_auc_score",  "AUC-PR",   "#048B8B"),
    ]
    present = [(col, label, color) for col, label, color in metric_conf if col in df.columns]
    if not present:
        return []

    cl_col = "cell_line" if "cell_line" in df.columns else None

    fig = go.Figure()
    for col, label, color in present:
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        hover = (df.loc[mask, "tissue"] + " / " + df.loc[mask, cl_col]) \
            if cl_col else df.loc[mask, "tissue"]
        fig.add_trace(go.Violin(
            y=vals[mask],
            x=[label] * mask.sum(),
            name=label,
            box_visible=True,
            meanline_visible=True,
            points="all",
            fillcolor=color,
            opacity=0.6,
            line_color="#03312E",
            hovertext=hover,
            hovertemplate="<b>%{hovertext}</b><br>Score: %{y:.3f}<extra></extra>",
        ))

    # Annotation of n, mean, median, std in a box at the bottom of the plot
    annotations = []
    for col, label, color in present:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        n = len(vals)
        mean = vals.mean()
        median = vals.median()
        std = vals.std()
        annotations.append(
            f"n={n}<br>mean={mean:.2f}<br>median={median:.2f}<br>std={std:.2f}"
        )
    
    # Add annotation for each metric, positioned below the x-axis, centered under the corresponding violin
    for i, annotation in enumerate(annotations):
        fig.add_annotation(
            text=annotation,
            x=i, y=-0.18,
            xref="x", yref="paper",
            showarrow=False,
            align="center",
            font=dict(size=13, color="#333"),
        )

    fig.update_layout(
        title=dict(text="ROC / PR / F1 Score Distribution — Cross-Cell-Lines", x=0.5),
        yaxis_title="Score",
        # xaxis_title="Metric",
        height=700,
        margin=dict(l=40, r=20, t=60, b=100),
        # template="plotly_white",
        showlegend=False,
    )
    return [fig]


def plot_roc_pr_heatmap_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """AUC score heatmap per tissue."""
    from synco.plotting.multi_tissue_summary import plot_tissue_roc_pr_f1 as _detail
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []
    try:
        result = _detail(roc_auc_df, plots_dir=None)
        if isinstance(result, dict):
            fig = result.get("heatmap")
            return [fig] if fig is not None else []
        return []
    except Exception:
        logger.exception("plot_roc_pr_heatmap_ct failed")
        return []


def plot_roc_pr_bars_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """F1 / AUC bar plots across tissues."""
    from synco.plotting.multi_tissue_summary import plot_tissue_roc_pr_f1 as _detail
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []
    try:
        result = _detail(roc_auc_df, plots_dir=None)
        if isinstance(result, dict):
            fig = result.get("bar")
            return [fig] if fig is not None else []
        return []
    except Exception:
        logger.exception("plot_roc_pr_bars_ct failed")
        return []


def plot_roc_violin_roc_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """AUC-ROC violin per tissue (simple per-tissue distribution)."""
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty or "roc_auc_score" not in roc_auc_df.columns:
        return []
    return [_make_metric_violin(roc_auc_df, "roc_auc_score",
                                "AUC-ROC Distribution Across Tissues", "#FFA15A")]


def plot_roc_violin_pr_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """AUC-PR violin per tissue (simple per-tissue distribution)."""
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty or "pr_auc_score" not in roc_auc_df.columns:
        return []
    return [_make_metric_violin(roc_auc_df, "pr_auc_score",
                                "AUC-PR Distribution Across Tissues", "#19D3F3")]


def plot_roc_pr_violin_table_roc_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """AUC-ROC violin + summary statistics table across tissues (classic multi-tissue view)."""
    from synco.plotting.multi_tissue_summary import plot_roc_pr_violin as _fn
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []
    try:
        fig = _fn(roc_auc_df, plots_dir=None, metric="ROC")
        return [fig] if fig is not None else []
    except Exception:
        logger.exception("plot_roc_pr_violin_table_roc_ct failed")
        return []


def plot_roc_pr_violin_table_pr_ct(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """AUC-PR violin + summary statistics table across tissues (classic multi-tissue view)."""
    from synco.plotting.multi_tissue_summary import plot_roc_pr_violin as _fn
    _, roc_auc_df, _ = _load_summaries(cell_fate_dir)
    if roc_auc_df is None or roc_auc_df.empty:
        return []
    try:
        fig = _fn(roc_auc_df, plots_dir=None, metric="PR")
        return [fig] if fig is not None else []
    except Exception:
        logger.exception("plot_roc_pr_violin_table_pr_ct failed")
        return []


# ---------------------------------------------------------------------------
# Distribution summary plots
# ---------------------------------------------------------------------------

def plot_exp_dist_by_tissue(
    cell_fate_dir: str, filters: dict | None = None, threshold: float = 0.0
) -> list[go.Figure]:
    """Synergy scores across tissues: tissues on y-axis, scores on x-axis.

    Each point is a (cell_line × combination) observation, coloured by cell line.
    """
    df_all = _load_all_experimental(cell_fate_dir)
    if df_all.empty or "synergy" not in df_all.columns:
        return []

    if filters:
        from synco.dashboard.plots._data import apply_filters
        df_all = apply_filters(df_all, {k: v for k, v in filters.items()
                                         if k in ("drug", "profile", "combination")})
    if df_all.empty:
        return []

    tissues  = sorted(df_all["tissue"].unique())
    cl_col   = "cell_line" if "cell_line" in df_all.columns else None

    fig = go.Figure()
    if cl_col:
        cell_lines = sorted(df_all[cl_col].dropna().unique())
        palette = {cl: px.colors.qualitative.Plotly[i % 10] for i, cl in enumerate(cell_lines)}
        for cl in cell_lines:
            sub = df_all[df_all[cl_col] == cl]
            fig.add_trace(go.Scatter(
                x=sub["synergy"],
                y=sub["tissue"],
                mode="markers",
                name=cl,
                marker=dict(size=5, opacity=0.55, color=palette[cl]),
                hovertemplate=(
                    "<b>%{y}</b><br>Synergy: %{x:.3f}"
                    f"<br>Cell line: {cl}<extra></extra>"
                ),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df_all["synergy"],
            y=df_all["tissue"],
            mode="markers",
            marker=dict(size=5, opacity=0.5, color="#636EFA"),
        ))

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"threshold={threshold}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text="Experimental Synergy Score Distribution by Tissue", x=0.5),
        xaxis_title="Synergy Score",
        yaxis_title="Tissue",
        height=max(350, 60 * len(tissues) + 100),
        margin=dict(l=160, r=20, t=60, b=60),
        template="plotly_white",
        legend=dict(title="Cell line", orientation="v", x=1.01, y=1, xanchor="left"),
    )
    return [fig]


def plot_exp_dist_by_combo(
    cell_fate_dir: str, filters: dict | None = None, threshold: float = 0.0
) -> list[go.Figure]:
    """Synergy scores with drug combinations on y-axis, coloured by tissue."""
    df_all = _load_all_experimental(cell_fate_dir)
    if df_all.empty or "synergy" not in df_all.columns:
        return []

    combo_col = next(
        (c for c in ("inhibitor_combination", "drug_combination") if c in df_all.columns),
        None,
    )
    if combo_col is None:
        return []

    if filters:
        from synco.dashboard.plots._data import apply_filters
        df_all = apply_filters(df_all, {k: v for k, v in filters.items()
                                         if k in ("drug", "profile")})
    if df_all.empty:
        return []

    tissues = sorted(df_all["tissue"].unique())
    palette = {t: px.colors.qualitative.Plotly[i % 10] for i, t in enumerate(tissues)}

    fig = go.Figure()
    for tissue in tissues:
        sub = df_all[df_all["tissue"] == tissue]
        fig.add_trace(go.Scatter(
            x=sub["synergy"],
            y=sub[combo_col],
            mode="markers",
            name=tissue,
            marker=dict(size=5, opacity=0.55, color=palette[tissue]),
            hovertemplate=(
                "<b>%{y}</b><br>Synergy: %{x:.3f}"
                f"<br>Tissue: {tissue}<extra></extra>"
            ),
        ))

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"threshold={threshold}",
        annotation_position="top right",
    )
    combos = sorted(df_all[combo_col].dropna().unique())
    fig.update_layout(
        title=dict(text="Experimental Synergy Score Distribution by Combination", x=0.5),
        xaxis_title="Synergy Score",
        yaxis_title="Drug Combination",
        height=max(350, 30 * len(combos) + 120),
        margin=dict(l=220, r=20, t=60, b=60),
        template="plotly_white",
        legend=dict(title="Tissue", orientation="v", x=1.01, y=1, xanchor="left"),
    )
    return [fig]


def plot_exp_synergy_counts(
    cell_fate_dir: str, filters: dict | None = None
) -> list[go.Figure]:
    """Per-tissue stacked bar plots of synergistic vs non-synergistic inhibitor counts.

    Uses make_experimental_distribution_plots per tissue to produce the same
    stacked-bar layout as the single-tissue view.
    """
    from synco.plotting.exp_distributions import make_experimental_distribution_plots
    tissue_dirs = _scan_tissue_dirs(cell_fate_dir)
    figs: list[go.Figure] = []
    for td in tissue_dirs:
        tissue_name = td.parent.name
        try:
            result = make_experimental_distribution_plots(
                str(td), plots_dir=None, return_fig=True
            )
            if result:
                # result[0] = (fig_synergy_counts, 'plotly')
                fig = result[0][0]
                old_title = (fig.layout.title.text or "") if fig.layout.title else ""
                new_title = f"[{tissue_name}] {old_title}" if old_title else f"[{tissue_name}]"
                fig.update_layout(title=dict(text=new_title, x=0.5))
                figs.append(fig)
        except Exception:
            logger.exception("Failed synergy stacked bars for tissue '%s'", tissue_name)
    return figs


# ---------------------------------------------------------------------------
# Per-tissue iteration helpers (distributions / profiles)
# ---------------------------------------------------------------------------

def _iter_tissue_figs(cell_fate_dir: str, fn, filters: dict | None) -> list[go.Figure]:
    """Call *fn(results_dir, filters)* for every tissue directory; collect figures."""
    tissue_dirs = _scan_tissue_dirs(cell_fate_dir)
    all_figs: list[go.Figure] = []
    for td in tissue_dirs:
        tissue_name = td.parent.name
        try:
            figs = fn(str(td), filters=filters)
            for fig in (figs or []):
                old_title = (fig.layout.title.text or "") if fig.layout.title else ""
                new_title = f"[{tissue_name}] {old_title}" if old_title else f"[{tissue_name}]"
                fig.update_layout(title=dict(text=new_title))
                all_figs.append(fig)
        except Exception:
            logger.exception("Failed to render %s for tissue %s", fn.__name__, tissue_name)
    return all_figs


def plot_exp_distributions_all(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Experimental distributions for every tissue (per-tissue detail view)."""
    from synco.dashboard.plots.distributions import plot_experimental
    return _iter_tissue_figs(cell_fate_dir, plot_experimental, filters)


def plot_pred_distributions_all(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Predicted distributions per tissue — one violin per inhibitor group / mechanism."""
    from synco.plotting.pred_distributions import make_pred_distribution_plots
    tissue_dirs = _scan_tissue_dirs(cell_fate_dir)
    all_figs: list[go.Figure] = []
    for td in tissue_dirs:
        tissue_name = td.parent.name
        try:
            result = make_pred_distribution_plots(str(td), plots_dir=None, return_fig=True)
            for fig, _ in (result or []):
                old_title = (fig.layout.title.text or "") if fig.layout.title else ""
                new_title = f"[{tissue_name}] {old_title}" if old_title else f"[{tissue_name}]"
                fig.update_layout(title=dict(text=new_title))
                all_figs.append(fig)
        except Exception:
            logger.exception("Failed predicted distributions for tissue '%s'", tissue_name)
    return all_figs


def plot_profiles_all(cell_fate_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Profile categories for every tissue."""
    from synco.dashboard.plots.profiles import plot_profile_categories
    return _iter_tissue_figs(cell_fate_dir, plot_profile_categories, filters)
