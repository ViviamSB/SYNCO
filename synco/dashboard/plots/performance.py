"""
performance.py – Plotly-native ring (donut) plots for the SYNCO dashboard.

Supported filters:
  plot_ring_summary       → combination, drug, profile
  plot_cell_line_rings    → cell_line
  plot_combination_rings  → combination, drug, profile
"""
from __future__ import annotations

import logging
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synco.dashboard.plots._data import (
    RING_COLORS,
    check_empty,
    load,
    comparison,
    normalise_comparison_df,
    _extract_cell_line_df,
    _extract_combination_df,
    _extract_valid_inhibitor_combis,
)
from synco.dashboard.plot_registry import NoFilterMatchError

logger = logging.getLogger(__name__)

# Normalised column aliases
_TP_COLS = ("True Positive", "True Positives", "TP")
_TN_COLS = ("True Negative", "True Negatives", "TN")
_FP_COLS = ("False Positive", "False Positives", "FP")
_FN_COLS = ("False Negative", "False Negatives", "FN")


def _find_col(df: pd.DataFrame, aliases: tuple) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    return None


def _prepare_ring_rows(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Normalise a comparison DataFrame for ring plotting.

    Returns a DataFrame with columns: [id_col, TP, TN, FP, FN, Recall, Accuracy, Precision]
    """
    df = df.copy()
    # Ensure id_col is a column (not index)
    if id_col not in df.columns:
        df = df.reset_index()
        if df.columns[0] != id_col:
            df = df.rename(columns={df.columns[0]: id_col})

    rename = {}
    for src in _TP_COLS[1:]:
        if src in df.columns and "True Positive" not in df.columns:
            rename[src] = "True Positive"
    for src in _TN_COLS[1:]:
        if src in df.columns and "True Negative" not in df.columns:
            rename[src] = "True Negative"
    for src in _FP_COLS[1:]:
        if src in df.columns and "False Positive" not in df.columns:
            rename[src] = "False Positive"
    for src in _FN_COLS[1:]:
        if src in df.columns and "False Negative" not in df.columns:
            rename[src] = "False Negative"
    df = df.rename(columns=rename)

    for col in ("True Positive", "True Negative", "False Positive", "False Negative",
                "Accuracy", "Recall", "Precision"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ---------------------------------------------------------------------------
# Nested donut (aggregate ring)
# ---------------------------------------------------------------------------

def _aggregate_nested_donut(
    tp: float, tn: float, fp: float, fn: float,
    title: str = "Aggregate Performance Ring",
) -> go.Figure:
    """Build a nested donut: outer = Match/Mismatch, inner = TP/TN/FP/FN."""
    match    = tp + tn
    mismatch = fp + fn
    total    = match + mismatch
    recall   = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    accuracy = match / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    fig = go.Figure()

    # Outer ring: Match / Mismatch
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

    # Inner ring: TP / TN / FP / FN
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

    # Centre annotation
    centre_text = (
        f"<b>Recall</b><br>{recall:.3f}"
        f"<br><br><b>Accuracy</b><br>{accuracy:.3f}"
        f"<br><br><b>Precision</b><br>{precision:.3f}"
    ) if not math.isnan(recall) else "No data"

    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=[dict(
            text=centre_text,
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False,
            align="center",
        )],
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Grid of mini donuts
# ---------------------------------------------------------------------------

def _donut_grid(df: pd.DataFrame, id_col: str, title: str, n_cols: int = 5) -> go.Figure:
    """Build a grid of TP/TN/FP/FN donut subplots, one per row in *df*."""
    n = len(df)
    if n == 0:
        return go.Figure()

    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)
    labels = df[id_col].astype(str).tolist()

    # Subtitle: Accuracy below each donut
    acc_vals = pd.to_numeric(df.get("Accuracy", pd.Series([float("nan")] * n)), errors="coerce")
    acc_labels = [f"{labels[i]}<br><sub>Acc={acc_vals.iloc[i]:.2f}</sub>"
                  if not pd.isna(acc_vals.iloc[i]) else labels[i]
                  for i in range(n)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
        subplot_titles=acc_labels,
    )

    for i, row in df.iterrows():
        r_idx, c_idx = divmod(list(df.index).index(i), n_cols)
        tp = float(row.get("True Positive", 0) or 0)
        tn = float(row.get("True Negative", 0) or 0)
        fp = float(row.get("False Positive", 0) or 0)
        fn = float(row.get("False Negative", 0) or 0)

        fig.add_trace(
            go.Pie(
                values=[tp, tn, fp, fn],
                labels=["TP", "TN", "FP", "FN"],
                hole=0.45,
                marker_colors=[RING_COLORS["TP"], RING_COLORS["TN"],
                               RING_COLORS["FP"], RING_COLORS["FN"]],
                showlegend=(i == df.index[0]),   # legend only on first
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
                textposition="inside",
                name=str(row[id_col]),
            ),
            row=r_idx + 1, col=c_idx + 1,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=200 * n_rows + 100,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.05,
            xanchor="center", x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _filter_by_combis(df: pd.DataFrame, id_col: str, valid_combis: set) -> pd.DataFrame:
    """Keep only rows whose id_col value is in *valid_combis*."""
    if not valid_combis:
        return df
    return df[df[id_col].isin(valid_combis)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_ring_summary(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Single aggregate donut showing overall TP/TN/FP/FN.

    Filterable by: combination, drug, profile
    """
    r = load(results_dir)
    comp = comparison(r)
    df = _extract_combination_df(comp) or _extract_cell_line_df(comp)
    if df is None or df.empty:
        return []

    # Apply combination/drug/profile filters via predictions
    if filters and any(filters.get(k) for k in ("combination", "drug", "profile")):
        valid_combis, matched = _extract_valid_inhibitor_combis(r, filters)
        if not matched:
            raise NoFilterMatchError()
        # Filter combination df
        combo_df = _extract_combination_df(comp)
        if combo_df is not None and not combo_df.empty:
            combo_df = normalise_comparison_df(combo_df, "inhibitor_combination")
            combo_df = _prepare_ring_rows(combo_df, "inhibitor_combination")
            combo_df = _filter_by_combis(combo_df, "inhibitor_combination", valid_combis)
            check_empty(combo_df, "combination/drug/profile filter")
            df = combo_df
        else:
            df = normalise_comparison_df(df, "label")
    else:
        df = normalise_comparison_df(df, df.columns[0])

    df = _prepare_ring_rows(df, df.columns[0])

    tp = df["True Positive"].sum() if "True Positive" in df.columns else 0
    tn = df["True Negative"].sum() if "True Negative" in df.columns else 0
    fp = df["False Positive"].sum() if "False Positive" in df.columns else 0
    fn = df["False Negative"].sum() if "False Negative" in df.columns else 0

    fig = _aggregate_nested_donut(tp, tn, fp, fn, title="Aggregate Performance Ring")
    return [fig]


def plot_cell_line_rings(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Grid of donuts, one per cell line.

    Filterable by: cell_line
    """
    r    = load(results_dir)
    comp = comparison(r)
    df   = _extract_cell_line_df(comp)
    if df is None or df.empty:
        return []

    df = normalise_comparison_df(df, "cell_line")
    df = _prepare_ring_rows(df, "cell_line")

    if filters and filters.get("cell_line"):
        df = df[df["cell_line"] == filters["cell_line"]]
        check_empty(df, "cell_line filter")

    fig = _donut_grid(df, "cell_line", "Performance by Cell Line", n_cols=5)
    return [fig]


def plot_combination_rings(results_dir: str, filters: dict | None = None) -> list[go.Figure]:
    """Grid of donuts, one per drug combination.

    Filterable by: combination, drug, profile
    """
    r    = load(results_dir)
    comp = comparison(r)
    df   = _extract_combination_df(comp)
    if df is None or df.empty:
        return []

    df = normalise_comparison_df(df, "inhibitor_combination")
    df = _prepare_ring_rows(df, "inhibitor_combination")

    if filters and any(filters.get(k) for k in ("combination", "drug", "profile")):
        valid_combis, matched = _extract_valid_inhibitor_combis(r, filters)
        if not matched:
            raise NoFilterMatchError()
        df = _filter_by_combis(df, "inhibitor_combination", valid_combis)
        check_empty(df, "combination/drug/profile filter")

    fig = _donut_grid(df, "inhibitor_combination",
                      "Performance by Drug Combination", n_cols=4)
    return [fig]
