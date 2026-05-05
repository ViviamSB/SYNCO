"""
distributions.py – Plotly-native experimental and predicted synergy distribution
plots for the SYNCO dashboard.

Supported filters:
  plot_experimental  → cell_line, combination, drug, profile
  plot_predicted     → cell_line, drug, profile
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from synco.dashboard.plots._data import (
    apply_filters,
    check_empty,
    dicts,
    experimental,
    load,
    predictions,
)

logger = logging.getLogger(__name__)

# Prediction metadata columns (not cell lines)
_PRED_META = frozenset({
    "Perturbation", "PD_A", "PD_B",
    "drug_name_A", "drug_name_B", "drug_combination",
    "node_targets_A", "node_targets_B",
    "inhibitor_group_A", "inhibitor_group_B", "inhibitor_combination",
    "targets_A", "targets_B", "target_combination",
})

_PALETTE = [
    "#16B7D3", "#71C715", "#FC7299", "#F09138", "#636EFA",
    "#FF97FF", "#BD7EF7", "#72B7B2", "#FF6F61", "#0C40A0",
    "#FFDE4D", "#DD4477", "#757575",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _enrich_mechanisms(df: pd.DataFrame, d: dict) -> pd.DataFrame:
    """Add Mechanism_A, Mechanism_B, mech_combination columns from PD_mechanism_dict."""
    pm = d.get("PD_mechanism_dict") or {}

    def _mech(k):
        val = pm.get(k) if pm else None
        if isinstance(val, dict):
            return val.get("Mechanism")
        return val if isinstance(val, str) else None

    df = df.copy()
    df["Mechanism_A"] = df["PD_A"].map(_mech) if "PD_A" in df.columns else None
    df["Mechanism_B"] = df["PD_B"].map(_mech) if "PD_B" in df.columns else None

    if pm:
        df["mech_combination"] = (
            df["Mechanism_A"].fillna("Unknown") + " + " + df["Mechanism_B"].fillna("Unknown")
        )
    elif "inhibitor_combination" in df.columns:
        df["mech_combination"] = df["inhibitor_combination"]
    else:
        df["mech_combination"] = "Unknown"

    return df


def _color_map(categories: list) -> dict:
    return {c: _PALETTE[i % (len(_PALETTE) - 1)] for i, c in enumerate(sorted(set(categories)))}


# ---------------------------------------------------------------------------
# Experimental distributions
# ---------------------------------------------------------------------------

def plot_experimental(
    results_dir: str, filters: dict | None = None, threshold: float = 0.0
) -> list[go.Figure]:
    """Experimental synergy distribtion plots.

    Figure 1 – Horizontal stacked bar: n synergies / total per inhibitor combination,
               colour-coded by mechanism combination.
    Figure 2 – 3-panel: (a) histogram, (b) scatter synergy vs cell line, (c) bar n_synergies
               per cell line.

    Filterable by: cell_line, combination, drug, profile
    """
    r   = load(results_dir)
    df  = experimental(r)
    d   = dicts(r)

    if df is None or df.empty:
        return []

    df = df.dropna(subset=["PD_A", "PD_B"]).copy() if "PD_A" in df.columns else df.copy()
    df = _enrich_mechanisms(df, d)

    # Apply filters
    df = apply_filters(
        df, filters,
        cell_line_col="cell_line",
        combi_col="inhibitor_combination",
        drug_cols=("drug_name_A", "drug_name_B"),
        profile_cols=("mech_combination", "Mechanism_A", "Mechanism_B"),
    )
    check_empty(df, "current filters")

    df["synergy"] = pd.to_numeric(df["synergy"], errors="coerce")
    df["synergy_binary"] = (df["synergy"] >= threshold).astype(int)

    mech_colors = _color_map(df["mech_combination"].dropna().unique().tolist())

    figs: list[go.Figure] = []

    # ── Figure 1: Synergy counts per inhibitor combination ──────────────────
    if "inhibitor_combination" in df.columns:
        summ = (
            df.groupby("inhibitor_combination")
            .agg(
                n_syn=("synergy_binary", "sum"),
                total=("synergy_binary", "count"),
                mech=("mech_combination", "first"),
            )
            .reset_index()
        )
        summ["n_non_syn"] = summ["total"] - summ["n_syn"]
        summ["pct"] = (summ["n_syn"] / summ["total"] * 100).round(1)
        summ = summ.sort_values("pct", ascending=True)

        fig1 = go.Figure()
        for mech_val, grp in summ.groupby("mech"):
            color = mech_colors.get(str(mech_val), "#757575")
            fig1.add_trace(go.Bar(
                x=grp["n_syn"],
                y=grp["inhibitor_combination"].astype(str),
                orientation="h",
                name=str(mech_val),
                marker_color=color,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Synergies: %{x}<br>"
                    f"Mechanism: {mech_val}<extra></extra>"
                ),
            ))

        # Add non-synergy as a grey trace
        fig1.add_trace(go.Bar(
            x=summ["n_non_syn"],
            y=summ["inhibitor_combination"].astype(str),
            orientation="h",
            name="Non-synergistic",
            marker_color="#d3d3d3",
            hovertemplate="<b>%{y}</b><br>Non-synergistic: %{x}<extra></extra>",
        ))

        # Add percentage annotations
        for _, row in summ.iterrows():
            fig1.add_annotation(
                x=row["total"] + 0.5, y=str(row["inhibitor_combination"]),
                text=f"{row['pct']}%",
                showarrow=False, xanchor="left", font_size=10,
            )

        fig1.update_layout(
            barmode="stack",
            title="Synergy Counts by Inhibitor Combination",
            xaxis_title="Count",
            yaxis_title="Inhibitor Combination",
            height=max(350, 28 * len(summ) + 100),
            margin=dict(l=200, r=80, t=60, b=40),
            template="plotly_white",
            legend=dict(orientation="v", x=1.02, y=1),
        )
        figs.append(fig1)

    # ── Figure 2: Distribution panels ───────────────────────────────────────
    has_cell_line = "cell_line" in df.columns and df["cell_line"].notna().any()
    n_cols = 3 if has_cell_line else 1

    fig2 = make_subplots(
        rows=1, cols=n_cols,
        column_widths=[0.4, 0.4, 0.2] if n_cols == 3 else [1.0],
        subplot_titles=(
            ["Synergy Distribution", "Synergy vs Cell Line", "Synergies / Cell Line"]
            if n_cols == 3 else ["Synergy Distribution"]
        ),
    )

    # Panel (a): Histogram
    fig2.add_trace(
        go.Histogram(
            x=df["synergy"].dropna(),
            nbinsx=30,
            marker_color="#636EFA",
            opacity=0.8,
            name="Synergy",
            showlegend=False,
        ),
        row=1, col=1,
    )
    mean_val = float(df["synergy"].mean())
    fig2.add_vline(x=threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Threshold={threshold}", row=1, col=1)
    fig2.add_vline(x=mean_val, line_dash="dot", line_color="blue",
                   annotation_text=f"Mean={mean_val:.2f}", row=1, col=1)

    if has_cell_line:
        # Panel (b): Scatter synergy vs cell line
        cl_order = (
            df.groupby("cell_line")["synergy_binary"].mean()
            .sort_values().index.tolist()
        )
        for mech_val, sub in df.groupby("mech_combination"):
            color = mech_colors.get(str(mech_val), "#757575")
            fig2.add_trace(
                go.Scatter(
                    x=pd.to_numeric(sub["synergy"], errors="coerce"),
                    y=sub["cell_line"].astype(str),
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.6),
                    name=str(mech_val),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Synergy: %{x:.3f}<br>"
                        f"Mechanism: {mech_val}"
                        + ("<br>Drug A: %{customdata[0]}<br>Drug B: %{customdata[1]}"
                           if "drug_name_A" in sub.columns else "")
                        + "<extra></extra>"
                    ),
                    customdata=(
                        sub[["drug_name_A", "drug_name_B"]].values
                        if "drug_name_A" in sub.columns else None
                    ),
                    showlegend=False,
                ),
                row=1, col=2,
            )

        # Panel (c): Bar n_synergies per cell line
        cl_counts = (
            df[df["synergy_binary"] == 1]
            .groupby("cell_line").size()
            .reindex(cl_order, fill_value=0)
            .reset_index()
            .rename(columns={0: "n_syn"})
        )
        cl_total = df.groupby("cell_line").size().reindex(cl_order, fill_value=0)
        cl_counts["pct"] = (cl_counts["n_syn"] / cl_total.values * 100).round(1)

        fig2.add_trace(
            go.Bar(
                x=cl_counts["n_syn"],
                y=cl_counts["cell_line"].astype(str),
                orientation="h",
                marker_color="#ef553b",
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Synergies: %{x}<br>%{customdata:.1f}%<extra></extra>",
                customdata=cl_counts["pct"],
            ),
            row=1, col=3,
        )

    fig2.update_layout(
        title="Synergy Score Distribution",
        height=max(380, 25 * df["cell_line"].nunique() + 120) if has_cell_line else 380,
        template="plotly_white",
        margin=dict(l=100, r=20, t=80, b=40),
    )
    figs.append(fig2)

    return figs


# ---------------------------------------------------------------------------
# Predicted distributions
# ---------------------------------------------------------------------------

def plot_predicted(
    results_dir: str, filters: dict | None = None
) -> list[go.Figure]:
    """Predicted synergy distribution per inhibitor group.

    Delegates to make_pred_distribution_plots for violin per inhibitor group
    and mechanism summary table.
    """
    try:
        from synco.plotting.pred_distributions import make_pred_distribution_plots
    except ImportError:
        logger.warning("synco.plotting.pred_distributions not available")
        return []

    try:
        result = make_pred_distribution_plots(results_dir, plots_dir=None, return_fig=True)
        return [fig for fig, _ in (result or [])]
    except Exception:
        logger.exception("make_pred_distribution_plots failed for %s", results_dir)
        return []


def plot_predicted_by_inhibitor_group(
    results_dir: str, filters: dict | None = None
) -> list[go.Figure]:
    """Violin plots of predicted synergy grouped by inhibitor group pair.

    Always groups by inhibitor_group_A | inhibitor_group_B regardless of
    whether a mechanism dictionary is present.
    Filterable by: cell_line, drug, profile
    """
    r    = load(results_dir)
    pred = predictions(r)

    if pred is None or pred.empty:
        return []

    meta_cols = [c for c in pred.columns if c in _PRED_META]
    cl_cols   = [c for c in pred.columns if c not in _PRED_META]
    if not cl_cols or "inhibitor_group_A" not in pred.columns:
        return []

    melt = pred.melt(id_vars=meta_cols, value_vars=cl_cols,
                     var_name="cell_line", value_name="synergy")
    melt["synergy"] = pd.to_numeric(melt["synergy"], errors="coerce") * -1

    ig_b = melt["inhibitor_group_B"] if "inhibitor_group_B" in melt.columns \
           else melt["inhibitor_group_A"]
    melt["group"] = melt["inhibitor_group_A"].astype(str) + " | " + ig_b.astype(str)

    melt = apply_filters(
        melt, filters,
        cell_line_col="cell_line",
        combi_col="inhibitor_combination",
        drug_cols=("drug_name_A", "drug_name_B"),
        profile_cols=("group", "inhibitor_group_A", "inhibitor_group_B"),
    )
    check_empty(melt, "current filters")

    group_order = (
        melt.groupby("group")["synergy"]
        .median().sort_values(ascending=False).index.tolist()
    )
    colors = _color_map(group_order)

    fig = go.Figure()
    for g in group_order:
        sub = melt[melt["group"] == g]["synergy"].dropna()
        color = colors.get(g, "#757575")
        fig.add_trace(go.Violin(
            y=sub,
            name=g,
            box_visible=True,
            points="all",
            jitter=0.3,
            pointpos=0,
            marker_size=3,
            line_color=color,
            fillcolor=color,
            opacity=0.5,
            hovertemplate=f"<b>{g}</b><br>Synergy: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text="Predicted Synergy by Inhibitor Group", x=0.5),
        height=500,
        template="plotly_white",
        margin=dict(l=40, r=20, t=80, b=80),
        yaxis_title="Predicted Synergy Score",
        xaxis_title="Inhibitor Group Pair",
    )
    return [fig]
