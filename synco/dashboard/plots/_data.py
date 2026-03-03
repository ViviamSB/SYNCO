"""
_data.py – Shared data-loading helpers and filter utilities for the
           synco.dashboard.plots package.

All functions work purely in-memory; nothing is written to disk.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from synco.plotting.load_results import _load_main_results
from synco.dashboard.plot_registry import NoFilterMatchError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
RING_COLORS = {
    "TP":       "#2ca02c",   # green
    "TN":       "#1f77b4",   # blue
    "FP":       "#d62728",   # red
    "FN":       "#ff7f0e",   # orange
    "Match":    "#2ca02c",
    "Mismatch": "#d62728",
}

METRIC_COLORS = {
    "Accuracy":  "#636efa",
    "Recall":    "#ef553b",
    "Precision": "#00cc96",
    "F1 Score":  "#ab63fa",
    "AUC-ROC":   "#ffa15a",
    "AUC-PR":    "#19d3f3",
}

# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load(results_dir: str) -> dict:
    """Load all result artefacts from *results_dir*."""
    return _load_main_results(results_dir)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def experimental(r: dict) -> pd.DataFrame | None:
    df = r["files"].get("experimental")
    if df is not None:
        return df
    # Fallback: experimental data may live in synco_shared/ (2 levels above results_dir)
    results_dir = r.get("results_dir", "")
    if results_dir:
        import os
        import glob as _glob
        shared = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "synco_shared")
        for cand in sorted(_glob.glob(os.path.join(shared, "experimental_full_*.csv"))):
            try:
                loaded = pd.read_csv(cand)
                logger.info("Loaded experimental data from synco_shared: %s", cand)
                return loaded
            except Exception:
                pass
    return None


def predictions(r: dict) -> pd.DataFrame | None:
    return r["files"].get("predictions")


def comparison(r: dict):
    """Return the comparison DataFrame or dict-of-DataFrames."""
    return r["files"].get("comparison")


def roc_metrics(r: dict) -> pd.DataFrame | None:
    return r["files"].get("roc_metrics")


def roc_traces(r: dict) -> dict | None:
    return r.get("roc_traces")


def dicts(r: dict) -> dict:
    return r.get("dicts") or {}


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    filters: dict | None,
    *,
    cell_line_col: str = "cell_line",
    combi_col: str = "inhibitor_combination",
    drug_cols: tuple = ("drug_name_A", "drug_name_B"),
    profile_cols: tuple = (
        "mech_combination", "Mechanism_A", "Mechanism_B",
        "inhibitor_group_A", "inhibitor_group_B",
    ),
) -> pd.DataFrame:
    """Subset *df* by any active filter values. Returns the (possibly empty) result."""
    if not filters:
        return df

    for key, value in filters.items():
        if not value:
            continue

        if key == "cell_line" and cell_line_col in df.columns:
            df = df[df[cell_line_col] == value]

        elif key == "combination" and combi_col in df.columns:
            df = df[df[combi_col] == value]

        elif key == "drug":
            mask = pd.Series(False, index=df.index)
            for col in drug_cols:
                if col in df.columns:
                    mask = mask | (df[col] == value)
            df = df[mask]

        elif key == "profile":
            mask = pd.Series(False, index=df.index)
            for col in profile_cols:
                if col in df.columns:
                    mask = mask | (df[col] == value)
            df = df[mask]

    return df


def check_empty(df: pd.DataFrame, context: str = "filters") -> None:
    """Raise *NoFilterMatchError* if *df* is empty."""
    if df is None or len(df) == 0:
        raise NoFilterMatchError(
            f"No data matches the current {context} settings."
        )


# ---------------------------------------------------------------------------
# Comparison DataFrame helpers
# ---------------------------------------------------------------------------

_KNOWN_COMBI_KEYS = {"inhibitor_combination", "combination", "inhibitor_group"}
_KNOWN_CELL_KEYS  = {"cell_line", "cell"}


def _extract_cell_line_df(comp: Any) -> pd.DataFrame | None:
    """Return the cell-line comparison DataFrame from *comp* (dict or DataFrame)."""
    if comp is None:
        return None
    if isinstance(comp, pd.DataFrame):
        return comp
    if isinstance(comp, dict):
        for key in comp:
            if any(k in key.lower() for k in ("cell_line", "cell")):
                return comp[key]
        # fallback: first key
        return next(iter(comp.values()), None)
    return None


def _extract_combination_df(comp: Any) -> pd.DataFrame | None:
    """Return the combination comparison DataFrame from *comp*."""
    if comp is None:
        return None
    if isinstance(comp, pd.DataFrame):
        # single file — may be either type; return as-is
        return comp
    if isinstance(comp, dict):
        for key in comp:
            if any(k in key.lower() for k in ("inhibitor", "combination", "combi")):
                return comp[key]
    return None


# ---------------------------------------------------------------------------
# Inhibitor-combination filter helper (used by performance.py)
# ---------------------------------------------------------------------------

def _vals(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _extract_valid_inhibitor_combis(r: dict, filters: dict | None) -> tuple[set, bool]:
    """Return (valid_combi_set, matched_bool) from predictions filtered by *filters*."""
    preds = r.get("files", {}).get("predictions")
    if preds is None:
        return set(), False

    combi_vals   = _vals((filters or {}).get("combination"))
    drug_vals    = _vals((filters or {}).get("drug"))
    profile_vals = _vals((filters or {}).get("profile"))

    mask = pd.Series(True, index=preds.index)

    if combi_vals and "drug_combination" in preds.columns:
        mask &= preds["drug_combination"].isin(combi_vals)
    if drug_vals:
        dm = pd.Series(False, index=preds.index)
        for col in ("drug_name_A", "drug_name_B"):
            if col in preds.columns:
                dm |= preds[col].isin(drug_vals)
        mask &= dm
    if profile_vals:
        pm = pd.Series(False, index=preds.index)
        for col in ("inhibitor_group_A", "inhibitor_group_B"):
            if col in preds.columns:
                pm |= preds[col].isin(profile_vals)
        mask &= pm

    if not mask.any():
        return set(), False

    preds_f = preds[mask]
    valid: set[str] = set()
    if "inhibitor_combination" in preds_f.columns:
        valid = set(preds_f["inhibitor_combination"].dropna().astype(str))
    return valid, True


def normalise_comparison_df(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Ensure the first column is named *id_col* and is a regular column (not index)."""
    df = df.copy()
    if df.index.name and df.index.name != "RangeIndex":
        df = df.reset_index()
    if df.columns[0] != id_col:
        df = df.rename(columns={df.columns[0]: id_col})
    # Normalise TP/FP column name variants
    rename_map = {
        "True Positives": "True Positive",
        "True Negatives": "True Negative",
        "False Positives": "False Positive",
        "False Negatives": "False Negative",
    }
    df = df.rename(columns=rename_map)
    return df
