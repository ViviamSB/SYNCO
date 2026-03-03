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
import plotly.graph_objects as go

from synco.dashboard.plots._data import (
    load,
    roc_traces,
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
