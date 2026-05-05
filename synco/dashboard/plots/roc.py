"""
roc.py – Plotly ROC/PR curve and threshold sweep plots for the SYNCO dashboard.

The ROC/PR curves are **rehydrated** from pre-built go.Scatter trace objects
stored in the roc_pr_curves.json output file.

Supported filters: cell_line (both functions)
"""
from __future__ import annotations

import logging
import statistics

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from synco.dashboard.plots._data import (
    load,
    roc_traces,
    roc_metrics,
    check_empty,
)
from synco.dashboard.plot_registry import NoFilterMatchError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_cell_line(trace_name: str, cell_line: str) -> bool:
    """Return True if *trace_name* belongs to *cell_line*."""
    return trace_name.lower().startswith(cell_line.lower())


def _filter_trace_list(trace_list: list, cell_line: str | None) -> list:
    """Keep only (auc, trace) tuples matching *cell_line*; keep all if None."""
    if not cell_line:
        return trace_list
    filtered = [(auc, t) for auc, t in trace_list
                if _match_cell_line(t.name or "", cell_line)]
    return filtered


def _add_diagonal(fig: go.Figure, row=None, col=None) -> None:
    kw = dict(row=row, col=col) if row is not None else {}
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey", width=1),
            showlegend=False,
            hoverinfo="skip",
        ),
        **kw,
    )


def _add_reference_line(fig: go.Figure, row=None, col=None) -> None:
    """Horizontal reference line at 0.5 for PR curves (no-skill baseline)."""
    kw = dict(row=row, col=col) if row is not None else {}
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0.5, 0.5],
            mode="lines",
            line=dict(dash="dot", color="grey", width=1),
            showlegend=False,
            hoverinfo="skip",
        ),
        **kw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    results_dir: str, filters: dict | None = None
) -> list[go.Figure]:
    """ROC and PR curves from the pre-computed roc_pr_curves.json.

    Filterable by: cell_line
    """
    r = load(results_dir)
    rt = roc_traces(r)
    if rt is None:
        logger.warning("No roc_traces found in %s", results_dir)
        return []

    cell_line = (filters or {}).get("cell_line")
    traces_roc = _filter_trace_list(rt.get("traces_roc") or [], cell_line)
    traces_pr  = _filter_trace_list(rt.get("traces_pr")  or [], cell_line)

    if not traces_roc and not traces_pr:
        raise NoFilterMatchError()

    figs: list[go.Figure] = []

    # ── Figure 1: ROC curves ────────────────────────────────────────────────
    if traces_roc:
        sorted_roc = sorted(traces_roc, key=lambda x: x[0], reverse=True)
        auc_scores = [auc for auc, _ in sorted_roc]

        fig_roc = go.Figure()
        _add_diagonal(fig_roc)
        for _auc, trace in sorted_roc:
            fig_roc.add_trace(trace)

        avg = float(np.mean(auc_scores)) if auc_scores else float("nan")
        med = float(np.median(auc_scores)) if auc_scores else float("nan")
        fig_roc.add_annotation(
            x=0.98, y=0.05, xanchor="right", yanchor="bottom",
            text=f"Avg AUC: {avg:.3f} | Median: {med:.3f}",
            showarrow=False, font_size=11, bgcolor="rgba(255,255,255,0.7)",
        )
        fig_roc.update_layout(
            title="ROC Curves by Cell Line",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1.02]),
            yaxis=dict(range=[0, 1.02]),
            height=480,
            template="plotly_white",
            legend=dict(orientation="v", x=1.02, y=1),
        )
        figs.append(fig_roc)

    # ── Figure 2: PR curves ─────────────────────────────────────────────────
    if traces_pr:
        sorted_pr = sorted(traces_pr, key=lambda x: x[0], reverse=True)
        pr_scores = [auc for auc, _ in sorted_pr]

        fig_pr = go.Figure()
        _add_reference_line(fig_pr)
        for _auc, trace in sorted_pr:
            fig_pr.add_trace(trace)

        avg_pr = float(np.mean(pr_scores)) if pr_scores else float("nan")
        med_pr = float(np.median(pr_scores)) if pr_scores else float("nan")
        fig_pr.add_annotation(
            x=0.98, y=0.95, xanchor="right", yanchor="top",
            text=f"Avg AUC-PR: {avg_pr:.3f} | Median: {med_pr:.3f}",
            showarrow=False, font_size=11, bgcolor="rgba(255,255,255,0.7)",
        )
        fig_pr.update_layout(
            title="Precision-Recall Curves by Cell Line",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1.02]),
            yaxis=dict(range=[0, 1.02]),
            height=480,
            template="plotly_white",
            legend=dict(orientation="v", x=1.02, y=1),
        )
        figs.append(fig_pr)

    return figs


def plot_threshold_sweeps(
    results_dir: str, filters: dict | None = None
) -> list[go.Figure]:
    """AUC vs classification threshold offset per cell line.

    Filterable by: cell_line
    """
    r = load(results_dir)
    rt = roc_traces(r)
    if rt is None:
        return []

    sweeps = rt.get("threshold_sweeps") or []
    if not sweeps:
        return []

    cell_line = (filters or {}).get("cell_line")
    if cell_line:
        sweeps = [s for s in sweeps if s.get("cell_line", "").lower() == cell_line.lower()]
        if not sweeps:
            raise NoFilterMatchError()

    # Build one line per metric per cell line, using updatemenus dropdown to switch
    metrics = [
        ("roc_auc",          "ROC AUC"),
        ("pr_auc",           "PR AUC"),
        ("f1_score",         "F1 Score"),
        ("balanced_accuracy","Balanced Accuracy"),
    ]

    # Create traces for all metrics; visibility toggled by dropdown
    fig = go.Figure()
    n_metrics = len(metrics)
    n_cell_lines = len(sweeps)

    for m_idx, (m_key, m_label) in enumerate(metrics):
        for s in sweeps:
            cl  = s.get("cell_line", "?")
            pts = s.get("sweep", [])
            if not pts:
                continue
            xs = [p.get("offset", float("nan")) for p in pts]
            ys = [p.get(m_key, float("nan")) for p in pts]
            thresholds = [p.get("threshold", float("nan")) for p in pts]

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=cl,
                legendgroup=cl,
                showlegend=(m_idx == 0),
                visible=(m_idx == 0),
                hovertemplate=(
                    f"<b>{cl}</b><br>"
                    f"Offset: %{{x}}<br>"
                    f"{m_label}: %{{y:.3f}}<br>"
                    f"Threshold: {thresholds}<extra></extra>"
                ),
                customdata=thresholds,
            ))

    # Dropdown buttons to switch metric
    buttons = []
    for m_idx, (_, m_label) in enumerate(metrics):
        vis = []
        for i in range(n_metrics):
            vis += [i == m_idx] * n_cell_lines
        buttons.append(dict(
            label=m_label,
            method="update",
            args=[{"visible": vis}, {"yaxis.title.text": m_label}],
        ))

    fig.update_layout(
        title="Threshold Sweep per Cell Line",
        xaxis_title="Threshold Offset",
        yaxis_title="ROC AUC",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=1.0, y=1.15,
            showactive=True,
        )],
        height=480,
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return [fig]


# ---------------------------------------------------------------------------
# AUC metric summary views
# ---------------------------------------------------------------------------

def plot_auc_bars(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Horizontal grouped bar chart of AUC-ROC / AUC-PR / F1 per cell line.

    Sorted by AUC-ROC descending. Filterable by: cell_line.
    """
    from plotly.subplots import make_subplots

    r   = load(results_dir)
    df  = roc_metrics(r)
    if df is None or df.empty:
        return []

    # normalise column names
    col_map = {}
    for src, dst in (("roc_auc", "AUC-ROC"), ("pr_auc", "AUC-PR"),
                     ("f1_score", "F1 Score"), ("roc_auc_score", "AUC-ROC"),
                     ("pr_auc_score", "AUC-PR")):
        if src in df.columns and dst not in col_map:
            col_map[src] = dst
    df = df.rename(columns=col_map)

    cl_col = next((c for c in ("cell_line", "Cell Line") if c in df.columns), None)
    if cl_col is None:
        return []

    if filters and filters.get("cell_line"):
        df = df[df[cl_col] == filters["cell_line"]]
        check_empty(df, "cell_line filter")

    metrics = [m for m in ("AUC-ROC", "AUC-PR", "F1 Score") if m in df.columns]
    if not metrics:
        return []

    sort_col = metrics[0]
    df = df.sort_values(sort_col, ascending=True)

    palette = {"AUC-ROC": "#FFA15A", "AUC-PR": "#19D3F3", "F1 Score": "#AB63FA"}
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(
            y=df[cl_col].astype(str),
            x=pd.to_numeric(df[m], errors="coerce"),
            name=m,
            orientation="h",
            marker_color=palette.get(m, "#aaa"),
        ))
    fig.update_layout(
        title=dict(text="AUC-ROC / AUC-PR / F1 Score per Cell Line", x=0.5),
        xaxis_title="Score",
        yaxis_title="Cell Line",
        barmode="group",
        height=max(300, 30 * len(df) + 120),
        margin=dict(l=160, r=20, t=60, b=60),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return [fig]


def plot_auc_summary(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """3 aggregated violins (F1, AUC-ROC, AUC-PR) across cell lines + summary stats table.

    Filterable by: cell_line.
    """
    from plotly.subplots import make_subplots

    r   = load(results_dir)
    df  = roc_metrics(r)
    if df is None or df.empty:
        return []

    col_map = {}
    for src, dst in (("roc_auc", "AUC-ROC"), ("pr_auc", "AUC-PR"),
                     ("f1_score", "F1 Score"), ("roc_auc_score", "AUC-ROC"),
                     ("pr_auc_score", "AUC-PR")):
        if src in df.columns and dst not in col_map:
            col_map[src] = dst
    df = df.rename(columns=col_map)

    cl_col = next((c for c in ("cell_line", "Cell Line") if c in df.columns), None)

    if filters and filters.get("cell_line") and cl_col:
        df = df[df[cl_col] == filters["cell_line"]]
        check_empty(df, "cell_line filter")

    metric_conf = [
        ("F1 Score", "#AB63FA"),
        ("AUC-ROC",  "#FFA15A"),
        ("AUC-PR",   "#19D3F3"),
    ]
    present = [(label, color) for label, color in metric_conf if label in df.columns]
    if not present:
        return []

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        specs=[[{"type": "violin"}], [{"type": "table"}]],
    )

    for label, color in present:
        vals = pd.to_numeric(df[label], errors="coerce").dropna()
        hover = df.loc[vals.index, cl_col].astype(str) if cl_col else None
        fig.add_trace(
            go.Violin(
                y=vals,
                x=[label] * len(vals),
                name=label,
                box_visible=True,
                meanline_visible=True,
                points="all",
                fillcolor=color,
                opacity=0.7,
                line_color="black",
                hovertext=hover,
                hovertemplate=(
                    "<b>%{hovertext}</b><br>Score: %{y:.3f}<extra></extra>"
                    if hover is not None else "%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Summary statistics table
    summary_rows = []
    for label, _ in present:
        vals = pd.to_numeric(df[label], errors="coerce").dropna()
        summary_rows.append({
            "Metric": label,
            "n": len(vals),
            "Mean": f"{vals.mean():.3f}",
            "Median": f"{vals.median():.3f}",
            "Std": f"{vals.std():.3f}",
            "Min": f"{vals.min():.3f}",
            "Max": f"{vals.max():.3f}",
            ">0.5": int((vals > 0.5).sum()),
        })
    tbl = pd.DataFrame(summary_rows)
    fig.add_trace(
        go.Table(
            header=dict(values=list(tbl.columns), font_size=12,
                        align="center", fill_color="lightgrey"),
            cells=dict(values=[tbl[c] for c in tbl.columns],
                       font_size=11, align="center"),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(text="AUC / F1 Score Distribution Across Cell Lines", x=0.5),
        yaxis_title="Score",
        height=640,
        margin=dict(l=40, r=20, t=60, b=20),
        template="plotly_white",
    )
    return [fig]

