"""
pipeline_cb.py – Callbacks for SYNCO pipeline execution and result loading.

Background execution
--------------------
The pipeline runs in a ``threading.Thread`` to avoid blocking the Dash
server.  A module-level ``_state`` dict acts as a lightweight inter-thread
communication channel (acceptable for single-user local deployment).
A ``dcc.Interval`` component on the Setup page polls ``_state`` every
second and updates the status badge until the run finishes.

Config assembly
---------------
``_build_config_from_form`` translates raw form-field values into the nested
dict expected by ``synco.build_pipeline_config``.
"""

import json
import logging
import threading
from pathlib import Path

import dash
from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from synco import build_pipeline_config, run_pipeline
from synco.dashboard.components.config_form import FORM_FIELD_IDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory structure detection helpers
# ---------------------------------------------------------------------------

_EXPECTED_FILES = ["roc_metrics_df.csv", "roc_pr_curves.json"]


def _is_single_output_dir(path: Path) -> bool:
    """True if *path* directly contains at least one expected synco output file."""
    return any((path / f).exists() for f in _EXPECTED_FILES)


def _scan_multi_tissue_root(path: Path) -> list:
    """
    Return a sorted list of ``<tissue>/synco_output`` Paths found directly
    inside *path*, filtered to those that contain expected output files.
    """
    found = []
    try:
        for subdir in sorted(path.iterdir()):
            if not subdir.is_dir():
                continue
            candidate = subdir / "synco_output"
            if candidate.is_dir() and _is_single_output_dir(candidate):
                found.append(candidate)
    except PermissionError:
        pass
    return found


# ---------------------------------------------------------------------------
# Shared state for background thread
# ---------------------------------------------------------------------------
_state: dict = {
    "status":  "idle",   # "idle" | "running" | "done" | "error"
    "message": "",
    "results_dir": None,
}
_lock = threading.Lock()


def _set_state(**kwargs):
    with _lock:
        _state.update(kwargs)


def _get_state() -> dict:
    with _lock:
        return dict(_state)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _parse_cell_lines(raw: str | None) -> list | str | None:
    """Convert the free-text cell-lines field to the format synco expects."""
    if not raw or not raw.strip():
        return []
    raw = raw.strip()
    # If it looks like a CSV filename, pass as string
    if raw.endswith(".csv") or "/" in raw or "\\" in raw:
        return raw
    # Otherwise treat as comma-separated list
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_offsets(raw: str | None) -> list | None:
    """Convert comma-separated threshold-offset string to list of floats."""
    if not raw or not raw.strip():
        return None
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return None


def _build_config_from_form(
    base, pipeline_runs, input_path, output,
    cell_lines_raw, prediction_method, verbose,
    threshold, analysis_mode, duplicate_strategy,
    threshold_offsets_raw, roc_bootstrap_n, roc_bootstrap_ci,
    oc_enabled,
    oc_profiles, oc_exp_full, oc_pred_full,
    oc_syn_pred, oc_compare, oc_roc,
    advance_json_raw,
) -> dict:
    """Assemble the synco config dict from Setup-page form values."""

    advance = {}
    if advance_json_raw and advance_json_raw.strip():
        try:
            advance = json.loads(advance_json_raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in advance field; ignoring.")

    return {
        "paths": {
            "base":          base or "",
            "pipeline_runs": pipeline_runs or "",
            "input":         input_path or "",
            "output":        output or "",
        },
        "general": {
            "cell_lines":        _parse_cell_lines(cell_lines_raw),
            "verbose":           bool(verbose),
        },
        "compare": {
            "prediction_method":  prediction_method or "DrugLogics",
            "threshold":          float(threshold) if threshold is not None else 0.5,
            "analysis_mode":      analysis_mode or "cell_line",
            "duplicate_strategy": duplicate_strategy or "mean",
            "threshold_offsets":  _parse_offsets(threshold_offsets_raw),
            "roc_bootstrap_n":    int(roc_bootstrap_n) if roc_bootstrap_n else None,
            "roc_bootstrap_ci":   float(roc_bootstrap_ci) if roc_bootstrap_ci else 0.95,
        },
        "output_control": {
            "enabled":                    bool(oc_enabled),
            "write_profiles":             bool(oc_profiles),
            "write_experimental_full_df": bool(oc_exp_full),
            "write_predictions_full_df":  bool(oc_pred_full),
            "write_synergy_predictions":  bool(oc_syn_pred),
            "write_compare_outputs":      bool(oc_compare),
            "write_roc_outputs":          bool(oc_roc),
        },
        "advance": advance,
    }


def _validate_paths(base, pipeline_runs, input_path, output) -> str | None:
    """Return an error message string, or None if all paths are present."""
    missing = []
    for name, val in [("Base", base), ("Pipeline runs", pipeline_runs),
                      ("Input", input_path), ("Output", output)]:
        if not val or not val.strip():
            missing.append(name)
    if missing:
        return f"Required path(s) missing: {', '.join(missing)}"
    return None


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _run_pipeline_background(config: dict):
    """Target function for the background thread."""
    _set_state(status="running", message="Pipeline running…")
    try:
        final_config = build_pipeline_config(config)
        run_pipeline(final_config, verbose=final_config.get("general", {}).get("verbose", False))
        results_dir = final_config.get("paths", {}).get("output", "")
        _set_state(status="done", message="Pipeline complete!", results_dir=results_dir)
    except Exception as exc:
        logger.exception("Pipeline failed")
        _set_state(status="error", message=str(exc))


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_pipeline_callbacks(app: dash.Dash) -> None:
    """Attach all pipeline-related callbacks to *app*."""

    # ------------------------------------------------------------------
    # 1.  Start pipeline run
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-pipeline-status", "data"),
        Output("store-config",          "data"),
        Output("poll-interval",         "disabled"),
        Output("run-alert",             "children"),
        Output("run-alert",             "color"),
        Output("run-alert",             "is_open"),
        Input("btn-run-pipeline",       "n_clicks"),
        State(FORM_FIELD_IDS["base"],             "value"),
        State(FORM_FIELD_IDS["pipeline_runs"],    "value"),
        State(FORM_FIELD_IDS["input"],            "value"),
        State(FORM_FIELD_IDS["output"],           "value"),
        State(FORM_FIELD_IDS["cell_lines"],       "value"),
        State(FORM_FIELD_IDS["prediction_method"],"value"),
        State(FORM_FIELD_IDS["verbose"],          "value"),
        State(FORM_FIELD_IDS["threshold"],        "value"),
        State(FORM_FIELD_IDS["analysis_mode"],    "value"),
        State(FORM_FIELD_IDS["duplicate_strategy"],"value"),
        State(FORM_FIELD_IDS["threshold_offsets"],"value"),
        State(FORM_FIELD_IDS["roc_bootstrap_n"],  "value"),
        State(FORM_FIELD_IDS["roc_bootstrap_ci"], "value"),
        State(FORM_FIELD_IDS["output_control_enabled"], "value"),
        State(FORM_FIELD_IDS["write_profiles"],            "value"),
        State(FORM_FIELD_IDS["write_experimental_full_df"],"value"),
        State(FORM_FIELD_IDS["write_predictions_full_df"], "value"),
        State(FORM_FIELD_IDS["write_synergy_predictions"], "value"),
        State(FORM_FIELD_IDS["write_compare_outputs"],     "value"),
        State(FORM_FIELD_IDS["write_roc_outputs"],         "value"),
        State(FORM_FIELD_IDS["advance_json"],     "value"),
        prevent_initial_call=True,
    )
    def start_pipeline(
        n_clicks,
        base, pipeline_runs, input_path, output,
        cell_lines_raw, prediction_method, verbose,
        threshold, analysis_mode, duplicate_strategy,
        threshold_offsets_raw, roc_bootstrap_n, roc_bootstrap_ci,
        oc_enabled,
        oc_profiles, oc_exp_full, oc_pred_full,
        oc_syn_pred, oc_compare, oc_roc,
        advance_json_raw,
    ):
        if not n_clicks:
            raise PreventUpdate

        err = _validate_paths(base, pipeline_runs, input_path, output)
        if err:
            return (
                no_update, no_update, True,
                err, "danger", True,
            )

        config = _build_config_from_form(
            base, pipeline_runs, input_path, output,
            cell_lines_raw, prediction_method, verbose,
            threshold, analysis_mode, duplicate_strategy,
            threshold_offsets_raw, roc_bootstrap_n, roc_bootstrap_ci,
            oc_enabled,
            oc_profiles, oc_exp_full, oc_pred_full,
            oc_syn_pred, oc_compare, oc_roc,
            advance_json_raw,
        )

        _set_state(status="idle", message="", results_dir=None)
        thread = threading.Thread(
            target=_run_pipeline_background,
            args=(config,),
            daemon=True,
        )
        thread.start()

        status_data = {"status": "running", "message": "Pipeline running…"}
        return (
            status_data, config, False,          # enable interval
            "Pipeline started. See status below.", "info", True,
        )

    # ------------------------------------------------------------------
    # 2.  Poll pipeline status (driven by dcc.Interval)
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-pipeline-status", "data",   allow_duplicate=True),
        Output("store-results-dir",     "data",   allow_duplicate=True),
        Output("poll-interval",         "disabled", allow_duplicate=True),
        Input("poll-interval",          "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_pipeline(n_intervals):
        state = _get_state()
        status = state["status"]

        if status == "done":
            results = {"results_dir": state.get("results_dir")}
            return (
                {"status": "done", "message": state["message"]},
                results,
                True,   # disable interval
            )
        if status == "error":
            return (
                {"status": "error", "message": state["message"]},
                no_update,
                True,   # disable interval
            )
        # Still running
        return (
            {"status": "running", "message": state["message"]},
            no_update,
            False,
        )

    # ------------------------------------------------------------------
    # 3.  Load existing results directory (auto-detects structure)
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-results-dir",   "data",  allow_duplicate=True),
        Output("store-cell-fate-dir", "data",  allow_duplicate=True),
        Output("load-alert",          "children"),
        Output("load-alert",          "color"),
        Output("load-alert",          "is_open"),
        Input("btn-load-results",     "n_clicks"),
        State("input-load-path",      "value"),
        prevent_initial_call=True,
    )
    def load_existing_results(n_clicks, load_path):
        if not n_clicks:
            raise PreventUpdate

        if not load_path or not load_path.strip():
            return (
                no_update, no_update,
                "Please enter a path to the results directory.",
                "warning", True,
            )

        p = Path(load_path.strip())
        if not p.exists() or not p.is_dir():
            return (
                no_update, no_update,
                f"Directory not found: {p}",
                "danger", True,
            )

        # ------------------------------------------------------------------
        # Auto-detect structure
        # ------------------------------------------------------------------
        if _is_single_output_dir(p):
            # Single synco_output/ folder – use as results_dir directly.
            results_data   = {"results_dir": str(p)}
            # Also try the grandparent as a potential multi-tissue root.
            grandparent    = p.parent.parent
            tissue_dirs    = _scan_multi_tissue_root(grandparent)
            if tissue_dirs:
                cell_fate_data = {"cell_fate_dir": str(grandparent)}
                msg = (
                    f"Loaded single output: {p.name}  |  "
                    f"Multi-tissue root auto-detected ({len(tissue_dirs)} tissues): "
                    f"{grandparent.name}"
                )
            else:
                cell_fate_data = no_update
                msg = f"Loaded results from: {p}"
            return results_data, cell_fate_data, msg, "success", True

        tissue_dirs = _scan_multi_tissue_root(p)
        if tissue_dirs:
            # Multi-tissue root – store as cell_fate_dir; leave results_dir unset
            # until the user picks a tissue on the Explorer page.
            cell_fate_data = {"cell_fate_dir": str(p)}
            tissue_names   = [t.parent.name for t in tissue_dirs]
            msg = (
                f"Multi-tissue root detected ({len(tissue_dirs)} tissues: "
                f"{', '.join(tissue_names[:5])}"
                + (" …" if len(tissue_names) > 5 else "")
                + "). Select a tissue in Explorer for single-tissue plots."
            )
            return no_update, cell_fate_data, msg, "success", True

        return (
            no_update, no_update,
            f"No recognisable SYNCO output found in {p}. "
            "Expected roc_metrics_df.csv / roc_pr_curves.json directly or in "
            "<tissue>/synco_output/ sub-directories.",
            "warning", True,
        )
