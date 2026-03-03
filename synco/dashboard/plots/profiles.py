"""
profiles.py – Plotly parallel-categories plots for drug and combination profiles.

Supported filters: drug, combination, profile
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go

from synco.dashboard.plots._data import (
    check_empty,
    dicts,
    experimental,
    load,
)

logger = logging.getLogger(__name__)

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


def _color_seq(values: list) -> list[str]:
    uniq = sorted(set(v for v in values if v is not None))
    cmap = {v: _PALETTE[i % (len(_PALETTE) - 1)] for i, v in enumerate(uniq)}
    return [cmap.get(v, "#757575") for v in values]


def _build_profilecat_df(df: pd.DataFrame, d: dict) -> pd.DataFrame | None:
    """Build a per-drug dimension DataFrame (compound, PD, InhibitorGroup, Mechanism)."""
    drug_cols = [c for c in ("drug_name_A", "drug_name_B") if c in df.columns]
    pd_cols   = [c for c in ("PD_A", "PD_B") if c in df.columns]
    if not drug_cols or not pd_cols:
        return None

    pd_inh  = d.get("PD_inhibitors_dict") or {}
    pd_mech = d.get("PD_mechanism_dict")   or {}

    def _inh(k):
        v = pd_inh.get(k)
        if isinstance(v, dict):
            return v.get("InhibitorGroup") or v.get("inhibitor_group")
        return v if isinstance(v, str) else None

    def _mech(k):
        v = pd_mech.get(k)
        if isinstance(v, dict):
            return v.get("Mechanism")
        return v if isinstance(v, str) else None

    rows = []
    for drug_col, pd_col in zip(drug_cols, pd_cols):
        sub = df[[drug_col, pd_col]].dropna().drop_duplicates()
        sub.columns = ["compound", "PD"]
        sub["InhibitorGroup"] = sub["PD"].map(_inh).fillna(sub["PD"])
        sub["Mechanism"]      = sub["PD"].map(_mech).fillna("Unknown")
        rows.append(sub)

    if not rows:
        return None

    result = pd.concat(rows, ignore_index=True).drop_duplicates("compound")
    return result


def _build_combicat_df(df: pd.DataFrame, d: dict) -> pd.DataFrame | None:
    """Build a per-combination dimension DataFrame."""
    required = {"drug_name_A", "drug_name_B"}
    if not required.issubset(df.columns):
        return None

    pd_mech = d.get("PD_mechanism_dict") or {}

    def _mech(k):
        v = pd_mech.get(k)
        if isinstance(v, dict):
            return v.get("Mechanism")
        return v if isinstance(v, str) else None

    combi = df.copy()
    if "drug_combination" not in combi.columns:
        combi["drug_combination"] = combi["drug_name_A"] + " + " + combi["drug_name_B"]

    if "PD_A" in combi.columns and "PD_B" in combi.columns:
        combi["PD_combination"] = combi["PD_A"] + " + " + combi["PD_B"]
        combi["Mechanism_A"] = combi["PD_A"].map(_mech).fillna("Unknown")
        combi["Mechanism_B"] = combi["PD_B"].map(_mech).fillna("Unknown")
        combi["mech_combination"] = combi["Mechanism_A"] + " + " + combi["Mechanism_B"]
    else:
        combi["PD_combination"] = "Unknown"
        combi["mech_combination"] = "Unknown"

    if "inhibitor_combination" not in combi.columns:
        combi["inhibitor_combination"] = combi["drug_combination"]

    keep = ["drug_combination", "PD_combination", "inhibitor_combination", "mech_combination"]
    keep = [c for c in keep if c in combi.columns]
    result = combi[keep].drop_duplicates()
    return result


def _parcats_fig(df: pd.DataFrame, dims: list[str], color_col: str | None,
                 title: str) -> go.Figure:
    """Build a go.Parcats figure from *df* using *dims* as category dimensions."""
    if df.empty or not dims:
        return go.Figure()

    dimensions = []
    for dim in dims:
        if dim not in df.columns:
            continue
        dimensions.append(go.parcats.Dimension(
            values=df[dim].astype(str).tolist(),
            label=dim.replace("_", " ").title(),
        ))

    if not dimensions:
        return go.Figure()

    colors = _color_seq(df[color_col].tolist()) if color_col and color_col in df.columns else "#636EFA"

    fig = go.Figure(go.Parcats(
        dimensions=dimensions,
        line=dict(color=colors, colorscale="Plasma", shape="hspline"),
        hoveron="color",
        hoverinfo="count+probability",
        labelfont=dict(size=12),
        tickfont=dict(size=10),
        arrangement="freeform",
    ))
    fig.update_layout(
        title=title,
        height=450,
        margin=dict(l=100, r=100, t=60, b=40),
        template="plotly_white",
    )
    return fig


def plot_profile_categories(
    results_dir: str, filters: dict | None = None
) -> list[go.Figure]:
    """Parallel-categories plots for drug profiles and combination profiles.

    Figure 1 – Drug profiles:   Compound → InhibitorGroup → PD → Mechanism
    Figure 2 – Combi profiles:  Drug Combination → PD Combination → Inhibitor Combination → Mech Combination

    Filterable by: drug, combination, profile
    """
    r  = load(results_dir)
    df = experimental(r)
    d  = dicts(r)

    if df is None or df.empty:
        return []

    df = df.copy()

    figs: list[go.Figure] = []
    active_filters = filters or {}

    # ── Figure 1: Drug profiles ─────────────────────────────────────────────
    profile_df = _build_profilecat_df(df, d)
    if profile_df is not None and not profile_df.empty:
        # Apply filters
        if active_filters.get("drug"):
            profile_df = profile_df[profile_df["compound"] == active_filters["drug"]]
        if active_filters.get("profile"):
            pf = active_filters["profile"]
            mask = (
                (profile_df.get("InhibitorGroup", pd.Series()) == pf) |
                (profile_df.get("Mechanism", pd.Series()) == pf)
            )
            profile_df = profile_df[mask]

        if not profile_df.empty:
            has_mech = profile_df["Mechanism"].notna().any() and (profile_df["Mechanism"] != "Unknown").any()
            dims = ["compound", "InhibitorGroup", "PD"] + (["Mechanism"] if has_mech else [])
            dims = [d for d in dims if d in profile_df.columns]
            fig1 = _parcats_fig(
                profile_df, dims,
                color_col="Mechanism" if has_mech else "InhibitorGroup",
                title="Drug Profile Categories",
            )
            figs.append(fig1)

    # ── Figure 2: Combination profiles ─────────────────────────────────────
    combi_df = _build_combicat_df(df, d)
    if combi_df is not None and not combi_df.empty:
        # Apply filters
        if active_filters.get("combination"):
            combi_df = combi_df[
                (combi_df.get("drug_combination", pd.Series()) == active_filters["combination"]) |
                (combi_df.get("inhibitor_combination", pd.Series()) == active_filters["combination"])
            ]
        if active_filters.get("profile"):
            pf = active_filters["profile"]
            mask = combi_df.get("mech_combination", pd.Series("")).str.contains(pf, na=False)
            combi_df = combi_df[mask]

        if not combi_df.empty:
            has_mech = (
                "mech_combination" in combi_df.columns and
                (combi_df["mech_combination"] != "Unknown + Unknown").any()
            )
            dims = (
                ["drug_combination", "PD_combination", "inhibitor_combination", "mech_combination"]
                if has_mech
                else ["drug_combination", "PD_combination", "inhibitor_combination"]
            )
            dims = [d for d in dims if d in combi_df.columns]
            fig2 = _parcats_fig(
                combi_df, dims,
                color_col="mech_combination" if has_mech else "inhibitor_combination",
                title="Combination Profile Categories",
            )
            figs.append(fig2)

    if not figs:
        check_empty(pd.DataFrame(), "drug/combination/profile filter")

    return figs
