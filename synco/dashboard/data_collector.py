"""
data_collector.py – Centralised data loading for the SYNCO dashboard.

Architecture
------------
``collect_all_data`` (or ``load_or_build_cache``) reads every per-tissue
synco_output directory, loads the existing pipeline CSVs and JSON dicts,
and assembles a single ``DataBundle`` dataclass.  No new data is computed
here – the pipeline (``converge_synergies`` / ``compare_synergies``) already
wrote everything to disk.

Key files loaded per tissue
---------------------------
- ``experimental_full_df.csv``          (falls back to synco_shared/)
- ``predictions_full_df.csv``
- ``cell_line_comparison_results.csv``
- ``inhibitor_combination_comparison_results.csv``
- ``cell_line_pair_details.csv``        (written by compare_synergies when
- ``inhibitor_combination_pair_details.csv``  return_pair_details=True)
- ``roc_metrics_df.csv``
- JSON dictionary files (PD_mechanism_dict, etc.)

Multi-tissue root structure expected by ``_scan_multi_tissue_root``:
  <root>/
    <tissue_A>/synco_output/
    <tissue_B>/synco_output/
    synco_shared/              ← optional shared experimental data & dicts
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR_NAME = ".synco_cache"
_CACHE_FILE_NAME = "bundle.pkl"
_CACHE_VERSION = 3


# ---------------------------------------------------------------------------
# DataBundle
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Aggregated SYNCO pipeline outputs across all tissues."""

    experimental_df: Optional[pd.DataFrame] = None
    cell_line_comparison: Optional[pd.DataFrame] = None
    inhibitor_comparison: Optional[pd.DataFrame] = None
    cell_line_pair_details: Optional[pd.DataFrame] = None
    inhibitor_pair_details: Optional[pd.DataFrame] = None
    roc_metrics: Optional[pd.DataFrame] = None
    shared_dicts: dict = field(default_factory=dict)
    tissues: list = field(default_factory=list)
    source_dirs: dict = field(default_factory=dict)   # {tissue_label: results_dir_str}
    root_dir: str = ""
    metadata: dict = field(default_factory=dict)
    cache_version: int = _CACHE_VERSION

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter_pair_details(
        self, filters: Optional[dict]
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Return (cell_line_pair_df, inhibitor_pair_df) filtered to *filters*.

        Supported filter keys (value: str or list[str]):
          - ``"combination"``  – match ``inhibitor_combination``
          - ``"drug"``         – match ``drug_name_A`` or ``drug_name_B``
          - ``"profile"``      – match ``inhibitor_group_A`` or ``inhibitor_group_B``
          - ``"tissue"``       – match ``tissue``
        """
        if not filters or not any(
            filters.get(k) for k in ("combination", "drug", "profile", "tissue")
        ):
            return self.cell_line_pair_details, self.inhibitor_pair_details

        def _vals(v):
            if v is None:
                return []
            return [v] if isinstance(v, str) else list(v)

        combo_vals   = _vals(filters.get("combination"))
        drug_vals    = _vals(filters.get("drug"))
        profile_vals = _vals(filters.get("profile"))
        tissue_vals  = _vals(filters.get("tissue"))

        def _apply(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return df
            mask = pd.Series(True, index=df.index)
            if combo_vals and "inhibitor_combination" in df.columns:
                mask &= df["inhibitor_combination"].isin(combo_vals)
            if drug_vals:
                dm = pd.Series(False, index=df.index)
                for col in ("drug_name_A", "drug_name_B"):
                    if col in df.columns:
                        dm |= df[col].isin(drug_vals)
                if dm.any():
                    mask &= dm
            if profile_vals:
                pm = pd.Series(False, index=df.index)
                for col in ("inhibitor_group_A", "inhibitor_group_B"):
                    if col in df.columns:
                        pm |= df[col].isin(profile_vals)
                if pm.any():
                    mask &= pm
            if tissue_vals and "tissue" in df.columns:
                mask &= df["tissue"].isin(tissue_vals)
            result = df[mask]
            return result if not result.empty else df

        return _apply(self.cell_line_pair_details), _apply(self.inhibitor_pair_details)

    # ------------------------------------------------------------------
    # Dynamic ring computation
    # ------------------------------------------------------------------

    def derive_ring_df(
        self,
        pair_df: pd.DataFrame,
        group_by: str = "cell_line",
    ) -> pd.DataFrame:
        """Compute TP/TN/FP/FN ring metrics from a pair_details DataFrame.

        *pair_df* must have a ``confusion_matrix_value`` column with values
        "True Positive", "True Negative", "False Positive", "False Negative".
        Returns a DataFrame with one row per *group_by* value and columns:
        TP, TN, FP, FN, Total, Match, Mismatch, Accuracy, Recall, Precision.
        """
        if pair_df is None or pair_df.empty or group_by not in pair_df.columns:
            return pd.DataFrame()

        counts = (
            pair_df.groupby([group_by, "confusion_matrix_value"])
            .size()
            .unstack(fill_value=0)
        )
        for col in ("True Positive", "True Negative", "False Positive", "False Negative"):
            if col not in counts.columns:
                counts[col] = 0

        ring = pd.DataFrame({
            "True Positive":  counts["True Positive"],
            "True Negative":  counts["True Negative"],
            "False Positive": counts["False Positive"],
            "False Negative": counts["False Negative"],
        })
        ring["Total"]    = ring.sum(axis=1)
        ring["Match"]    = ring["True Positive"] + ring["True Negative"]
        ring["Mismatch"] = ring["False Positive"] + ring["False Negative"]
        ring["Match %"]    = ring.apply(
            lambda r: r["Match"] / r["Total"] * 100 if r["Total"] else 0.0, axis=1
        )
        ring["Mismatch %"] = ring.apply(
            lambda r: r["Mismatch"] / r["Total"] * 100 if r["Total"] else 0.0, axis=1
        )
        ring["Accuracy"] = ring.apply(
            lambda r: (r["True Positive"] + r["True Negative"]) / r["Total"] * 100
            if r["Total"] else 0.0, axis=1
        )
        ring["Recall"] = ring.apply(
            lambda r: r["True Positive"] / (r["True Positive"] + r["False Negative"]) * 100
            if (r["True Positive"] + r["False Negative"]) > 0 else 0.0, axis=1
        )
        ring["Precision"] = ring.apply(
            lambda r: r["True Positive"] / (r["True Positive"] + r["False Positive"]) * 100
            if (r["True Positive"] + r["False Positive"]) > 0 else 0.0, axis=1
        )
        ring.index.name = group_by
        return ring.reset_index()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True if at least one core DataFrame was loaded."""
        return any(
            df is not None and not df.empty
            for df in [
                self.experimental_df, self.cell_line_comparison,
                self.inhibitor_comparison, self.roc_metrics,
            ]
        )

    @property
    def has_pair_details(self) -> bool:
        return (
            (self.cell_line_pair_details is not None and not self.cell_line_pair_details.empty)
            or (self.inhibitor_pair_details is not None and not self.inhibitor_pair_details.empty)
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXPECTED_OUTPUT_FILES = ["roc_metrics_df.csv", "roc_pr_curves.json"]


def _is_synco_output_dir(path: Path) -> bool:
    return any((path / f).exists() for f in _EXPECTED_OUTPUT_FILES)


def _scan_tissue_dirs(root: Path) -> list[tuple[str, Path]]:
    """Return [(tissue_label, results_dir_path), ...] sorted by tissue label.

    Expects the structure:  <root>/<tissue>/synco_output/
    """
    found = []
    try:
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            candidate = subdir / "synco_output"
            if candidate.is_dir() and _is_synco_output_dir(candidate):
                found.append((subdir.name, candidate))
    except PermissionError:
        pass
    return found


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
    return None


def _read_json_safe(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
    return None


_JSON_DICT_FILES = {
    "PD_mechanism_dict":       "PD_mechanism_dict.json",
    "PD_inhibitors_dict":      "PD_inhibitors_dict.json",
    "mechanism_PD_dict":       "mechanism_PD_dict.json",
    "Drugnames_PD_dict":       "Drugnames_PD_dict.json",
    "PD_drugnames_dict":       "PD_drugnames_dict.json",
    "inhibitorgroups_dict":    "inhibitorgroups_dict.json",
    "Drugnames_inhibitor_dict":"Drugnames_inhibitor_dict.json",
    "PD_targets_dict":         "PD_targets_dict.json",
    "Drugnames_mechanism_dict":"Drugnames_mechanism_dict.json",
    "mechanism_drugnames_dict":"mechanism_drugnames_dict.json",
}


def _load_dicts_from_dir(directory: Path, existing: Optional[dict] = None) -> dict:
    """Load JSON dicts from *directory*; skip keys already present in *existing*."""
    dicts = dict(existing or {})
    for key, fname in _JSON_DICT_FILES.items():
        if dicts.get(key):
            continue
        loaded = _read_json_safe(directory / fname)
        if loaded is not None:
            dicts[key] = loaded
    return dicts


# ---------------------------------------------------------------------------
# Single-tissue loader
# ---------------------------------------------------------------------------

def collect_tissue_data(results_dir: str, tissue_label: Optional[str] = None) -> dict:
    """Load all existing pipeline CSVs for a single synco_output directory.

    Returns a dict with keys:
      - ``"experimental_df"``
      - ``"cell_line_comparison"``
      - ``"inhibitor_comparison"``
      - ``"cell_line_pair_details"``
      - ``"inhibitor_pair_details"``
      - ``"roc_metrics"``
      - ``"dicts"``
      - ``"tissue"``   (tissue_label)
      - ``"results_dir"``
      - ``"missing"``  (list of file names not found)

    All DataFrame values may be ``None`` if the file does not exist.
    A ``"tissue"`` column is added to each non-None DataFrame when
    *tissue_label* is provided.
    """
    rdir = Path(results_dir)
    missing: list[str] = []

    def _load_and_tag(path_or_none, fname_hint=""):
        df = _read_csv_safe(path_or_none) if path_or_none else None
        if df is None:
            missing.append(fname_hint)
        elif tissue_label and "tissue" not in df.columns:
            df["tissue"] = tissue_label
        return df

    # ── Core CSVs ──────────────────────────────────────────────────────────
    experimental_df = _load_and_tag(rdir / "experimental_full_df.csv", "experimental_full_df.csv")

    # Shared experimental fallback (lives two levels up in synco_shared/)
    if experimental_df is None:
        shared_dir = rdir.parent.parent / "synco_shared"
        for fname in [
            "experimental_full_df.csv",
            "experimental_drug_names_synergies_df.csv",
        ]:
            cand = shared_dir / fname
            if cand.exists():
                experimental_df = _read_csv_safe(cand)
                if experimental_df is not None:
                    # Restrict to tissue-specific cell lines if possible
                    preds = _read_csv_safe(rdir / "predictions_full_df.csv")
                    if preds is not None and "cell_line" in experimental_df.columns:
                        _pred_meta = {
                            "Perturbation", "PD_A", "PD_B",
                            "drug_name_A", "drug_name_B", "drug_combination",
                            "node_targets_A", "node_targets_B",
                            "inhibitor_group_A", "inhibitor_group_B", "inhibitor_combination",
                            "targets_A", "targets_B", "target_combination",
                        }
                        tissue_cls = [c for c in preds.columns if c not in _pred_meta]
                        if tissue_cls:
                            experimental_df = experimental_df[
                                experimental_df["cell_line"].isin(tissue_cls)
                            ]
                    if tissue_label and "tissue" not in experimental_df.columns:
                        experimental_df["tissue"] = tissue_label
                    missing_hint = "experimental_full_df.csv"
                    if missing_hint in missing:
                        missing.remove(missing_hint)
                    logger.info("Loaded experimental data from synco_shared: %s", cand)
                    break

    # Comparison results
    cell_line_comparison      = _load_and_tag(rdir / "cell_line_comparison_results.csv",               "cell_line_comparison_results.csv")
    inhibitor_comparison      = _load_and_tag(rdir / "inhibitor_combination_comparison_results.csv",    "inhibitor_combination_comparison_results.csv")

    # Pair details (written by compare_synergies when return_pair_details=True)
    cell_line_pair_details    = _load_and_tag(rdir / "cell_line_pair_details.csv",                    "cell_line_pair_details.csv")
    inhibitor_pair_details    = _load_and_tag(rdir / "inhibitor_combination_pair_details.csv",         "inhibitor_combination_pair_details.csv")

    roc_metrics               = _load_and_tag(rdir / "roc_metrics_df.csv",                            "roc_metrics_df.csv")

    # ── JSON dicts ─────────────────────────────────────────────────────────
    dicts = _load_dicts_from_dir(rdir)
    # Fallback: synco_shared/
    shared_dir = rdir.parent.parent / "synco_shared"
    if shared_dir.is_dir():
        dicts = _load_dicts_from_dir(shared_dir, existing=dicts)

    return {
        "experimental_df":           experimental_df,
        "cell_line_comparison":      cell_line_comparison,
        "inhibitor_comparison":      inhibitor_comparison,
        "cell_line_pair_details":    cell_line_pair_details,
        "inhibitor_pair_details":    inhibitor_pair_details,
        "roc_metrics":               roc_metrics,
        "dicts":                     dicts,
        "tissue":                    tissue_label,
        "results_dir":               str(rdir),
        "missing":                   missing,
    }


# ---------------------------------------------------------------------------
# Multi-tissue collector
# ---------------------------------------------------------------------------

def _concat_safe(frames: list[pd.DataFrame]) -> Optional[pd.DataFrame]:
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return None
    return pd.concat(valid, ignore_index=True)


def collect_all_data(
    cell_fate_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> DataBundle:
    """Collect and stack SYNCO pipeline outputs across all tissues.

    Pass *cell_fate_dir* (multi-tissue root) to load all tissues, or
    *results_dir* alone for a single synco_output directory.
    """
    if not cell_fate_dir and not results_dir:
        raise ValueError("Provide at least one of cell_fate_dir or results_dir.")

    # ── Build list of (tissue_label, results_dir_path) ─────────────────────
    tissue_list: list[tuple[str, Path]] = []

    if cell_fate_dir:
        root = Path(cell_fate_dir)
        tissue_list = _scan_tissue_dirs(root)
        if not tissue_list:
            logger.warning("No tissue sub-directories found in %s; treating as single output.", root)
            if _is_synco_output_dir(root):
                tissue_list = [(root.name, root)]

    if not tissue_list and results_dir:
        rdir = Path(results_dir)
        # Derive tissue label from directory structure (<root>/<tissue>/synco_output)
        if rdir.name == "synco_output":
            label = rdir.parent.name
        else:
            label = rdir.name
        tissue_list = [(label, rdir)]

    root_dir = str(Path(cell_fate_dir) if cell_fate_dir else Path(results_dir))

    # ── Load per tissue ────────────────────────────────────────────────────
    exp_frames:       list[pd.DataFrame] = []
    cl_comp_frames:   list[pd.DataFrame] = []
    ic_comp_frames:   list[pd.DataFrame] = []
    cl_pair_frames:   list[pd.DataFrame] = []
    ic_pair_frames:   list[pd.DataFrame] = []
    roc_frames:       list[pd.DataFrame] = []
    merged_dicts:     dict = {}
    source_dirs:      dict = {}
    all_missing:      dict = {}

    for tissue_label, tdir in tissue_list:
        logger.info("Loading tissue '%s' from %s", tissue_label, tdir)
        td = collect_tissue_data(str(tdir), tissue_label=tissue_label)
        source_dirs[tissue_label] = td["results_dir"]
        all_missing[tissue_label] = td["missing"]

        for frames, key in [
            (exp_frames,     "experimental_df"),
            (cl_comp_frames, "cell_line_comparison"),
            (ic_comp_frames, "inhibitor_comparison"),
            (cl_pair_frames, "cell_line_pair_details"),
            (ic_pair_frames, "inhibitor_pair_details"),
            (roc_frames,     "roc_metrics"),
        ]:
            val = td.get(key)
            if val is not None:
                frames.append(val)

        # Merge dicts (first tissue wins for each key)
        for k, v in td["dicts"].items():
            if v and k not in merged_dicts:
                merged_dicts[k] = v

    tissues = [label for label, _ in tissue_list]

    bundle = DataBundle(
        experimental_df          = _concat_safe(exp_frames),
        cell_line_comparison     = _concat_safe(cl_comp_frames),
        inhibitor_comparison     = _concat_safe(ic_comp_frames),
        cell_line_pair_details   = _concat_safe(cl_pair_frames),
        inhibitor_pair_details   = _concat_safe(ic_pair_frames),
        roc_metrics              = _concat_safe(roc_frames),
        shared_dicts             = merged_dicts,
        tissues                  = tissues,
        source_dirs              = source_dirs,
        root_dir                 = root_dir,
        metadata                 = {"missing_files": all_missing},
    )
    return bundle


# ---------------------------------------------------------------------------
# Pickle cache
# ---------------------------------------------------------------------------

def _cache_path(root_dir: str) -> Path:
    return Path(root_dir) / _CACHE_DIR_NAME / _CACHE_FILE_NAME


def _dir_fingerprint(root_dir: str) -> str:
    """Lightweight fingerprint: modification times of all CSV files under root_dir."""
    import hashlib
    h = hashlib.md5()
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in sorted(filenames):
                if fname.endswith(".csv") or fname.endswith(".json"):
                    fpath = os.path.join(dirpath, fname)
                    try:
                        h.update(str(os.path.getmtime(fpath)).encode())
                    except OSError:
                        pass
    except Exception:
        pass
    return h.hexdigest()


def load_or_build_cache(
    cell_fate_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    force: bool = False,
) -> DataBundle:
    """Return a cached DataBundle, rebuilding if the cache is stale or absent.

    The cache is stored at ``<root_dir>/.synco_cache/bundle.pkl``.
    It is invalidated when any CSV or JSON file under *root_dir* changes
    (checked via modification-time fingerprint) or when *force=True*.
    """
    root_dir = str(Path(cell_fate_dir) if cell_fate_dir else Path(results_dir))
    cp = _cache_path(root_dir)
    fingerprint = _dir_fingerprint(root_dir)

    if not force and cp.exists():
        try:
            with open(cp, "rb") as fh:
                cached: DataBundle = pickle.load(fh)
            if (
                getattr(cached, "cache_version", 0) == _CACHE_VERSION
                and cached.metadata.get("fingerprint") == fingerprint
            ):
                logger.info("Returning cached DataBundle from %s", cp)
                return cached
            else:
                logger.info("Cache stale or version mismatch – rebuilding.")
        except Exception as exc:
            logger.warning("Failed to load cache: %s", exc)

    bundle = collect_all_data(cell_fate_dir=cell_fate_dir, results_dir=results_dir)
    bundle.metadata["fingerprint"] = fingerprint

    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as fh:
            pickle.dump(bundle, fh)
        logger.info("DataBundle cached at %s", cp)
    except Exception as exc:
        logger.warning("Could not write cache: %s", exc)

    return bundle


# ---------------------------------------------------------------------------
# Metadata for dcc.Store
# ---------------------------------------------------------------------------

def bundle_to_metadata(bundle: DataBundle) -> dict:
    """Serialise bundle provenance to a JSON-compatible dict for dcc.Store.

    The store must be lightweight (no DataFrames), so we only record
    paths, shapes, and column names.
    """
    def _df_info(df: Optional[pd.DataFrame]) -> Optional[dict]:
        if df is None or df.empty:
            return None
        return {
            "rows":    int(len(df)),
            "columns": list(df.columns),
        }

    dataframes = {
        "experimental_df":         _df_info(bundle.experimental_df),
        "cell_line_comparison":    _df_info(bundle.cell_line_comparison),
        "inhibitor_comparison":    _df_info(bundle.inhibitor_comparison),
        "cell_line_pair_details":  _df_info(bundle.cell_line_pair_details),
        "inhibitor_pair_details":  _df_info(bundle.inhibitor_pair_details),
        "roc_metrics":             _df_info(bundle.roc_metrics),
    }

    warnings: list[str] = []
    missing = bundle.metadata.get("missing_files", {})
    for tissue, files in missing.items():
        for f in files:
            if "pair_details" in f:
                warnings.append(
                    f"{tissue}: {f} missing – pair_details-based ring plots unavailable."
                )

    return {
        "ready":           bundle.ready,
        "has_pair_details": bundle.has_pair_details,
        "root_dir":        bundle.root_dir,
        "tissues":         bundle.tissues,
        "source_dirs":     bundle.source_dirs,
        "dataframes":      dataframes,
        "dicts_loaded":    [k for k, v in bundle.shared_dicts.items() if v],
        "missing_files":   missing,
        "warnings":        warnings,
    }
