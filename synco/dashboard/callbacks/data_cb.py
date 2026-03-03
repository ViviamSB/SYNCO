"""
data_cb.py – Callbacks for the Data page.

Callbacks
---------
1. ``collect_data``   – triggered by "Collect Data" / "Reload" buttons
                        OR automatically when navigating to /data.
                        Calls ``load_or_build_cache``, writes metadata to
                        ``store-data``, and updates the summary panel.
2. ``export_data``    – triggered by "Save" button.
                        Reloads the cached DataBundle and writes CSVs.
"""

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from synco.dashboard.data_collector import (
    DataBundle,
    bundle_to_metadata,
    load_or_build_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary content builder  (kept here to avoid importing the pages module,
# which would cause Dash to register the page under two different module
# names and raise a duplicate-path error)
# ---------------------------------------------------------------------------

def build_summary_content(metadata: dict) -> html.Div:
    """Build a card with a DataFrame summary table from bundle metadata."""
    dataframes   = metadata.get("dataframes", {})
    dicts_loaded = metadata.get("dicts_loaded", [])
    warnings     = metadata.get("warnings", [])
    tissues      = metadata.get("tissues", [])

    # ── DataFrame table ────────────────────────────────────────────────────
    header = html.Thead(html.Tr([
        html.Th("DataFrame"),
        html.Th("Rows"),
        html.Th("Columns"),
        html.Th("Column names"),
    ]))
    rows = []
    for name, info in dataframes.items():
        if info is None:
            rows.append(html.Tr([
                html.Td(html.Code(name)),
                html.Td(html.Span("—", className="text-muted")),
                html.Td(html.Span("—", className="text-muted")),
                html.Td(html.Span("not available", className="text-muted fst-italic")),
            ], className="table-secondary"))
        else:
            cols_text = ", ".join(str(c) for c in info["columns"])
            rows.append(html.Tr([
                html.Td(html.Code(name)),
                html.Td(f"{info['rows']:,}"),
                html.Td(str(len(info["columns"]))),
                html.Td(
                    html.Small(cols_text, className="text-muted"),
                    style={"maxWidth": "480px", "wordBreak": "break-word"},
                ),
            ]))

    df_table = dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        size="sm",
        className="mb-0",
    )

    # ── Dictionaries ───────────────────────────────────────────────────────
    dict_badges = (
        [dbc.Badge(k, color="info", className="me-1") for k in sorted(dicts_loaded)]
        if dicts_loaded
        else [html.Span("None loaded", className="text-muted fst-italic")]
    )

    # ── Tissues ────────────────────────────────────────────────────────────
    tissue_badges = (
        [dbc.Badge(t, color="secondary", className="me-1") for t in tissues]
        if tissues
        else [html.Span("—", className="text-muted")]
    )

    # ── Warnings ───────────────────────────────────────────────────────────
    warn_items = (
        [dbc.Alert(w, color="warning", className="mb-1 py-2") for w in warnings]
        if warnings
        else []
    )

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody([
                    html.H6([html.I(className="bi bi-geo-alt me-2"), "Tissues"], className="card-title mb-2"),
                    html.Div(tissue_badges),
                ]),
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H6([html.I(className="bi bi-table me-2"), "DataFrames"], className="card-title mb-2"),
                    df_table,
                ]),
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H6([html.I(className="bi bi-book me-2"), "Loaded dictionaries"], className="card-title mb-2"),
                    html.Div(dict_badges),
                ]),
                className="mb-3",
            ),
            html.Div(warn_items) if warn_items else html.Div(),
        ]
    )


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_data_callbacks(app: dash.Dash) -> None:
    """Register all Data-page callbacks on *app*."""

    # ------------------------------------------------------------------
    # 1. Collect (or reload) all pipeline data
    #    Triggered by: Collect button, Reload button, or URL change to /data
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-data",             "data"),
        Output("data-summary-container", "children"),
        Output("data-status-badge",      "children"),
        Output("data-alert",             "children"),
        Output("data-alert",             "color"),
        Output("data-alert",             "is_open"),
        Output("data-export-section",    "style"),
        Output("data-nav-hint",          "children"),
        Input("btn-collect-data",  "n_clicks"),
        Input("btn-reload-data",   "n_clicks"),
        Input("url",               "pathname"),
        State("store-results-dir",   "data"),
        State("store-cell-fate-dir", "data"),
        prevent_initial_call=True,
    )
    def collect_data(n_collect, n_reload, pathname, results_data, cell_fate_data):
        triggered_id = dash.ctx.triggered_id

        # URL-triggered: only proceed when navigating to /data
        if triggered_id == "url" and pathname != "/data":
            raise PreventUpdate

        results_dir   = (results_data   or {}).get("results_dir")
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")

        if not results_dir and not cell_fate_dir:
            return (
                no_update,
                no_update,
                dbc.Badge("No data loaded", color="secondary", pill=True),
                "No results directory found. Please load results on the Setup page first.",
                "warning",
                True,
                {"display": "none"},
                None,
            )

        # Reload forces cache rebuild; URL navigation uses cache
        force = bool(triggered_id == "btn-reload-data" and n_reload)

        try:
            bundle: DataBundle = load_or_build_cache(
                cell_fate_dir=cell_fate_dir,
                results_dir=results_dir,
                force=force,
            )
        except Exception as exc:
            logger.exception("Failed to collect data")
            return (
                no_update,
                no_update,
                dbc.Badge("Error", color="danger", pill=True),
                f"Data collection failed: {exc}",
                "danger",
                True,
                {"display": "none"},
                None,
            )

        metadata = bundle_to_metadata(bundle)
        summary  = build_summary_content(metadata)

        status_badge = (
            dbc.Badge(
                [html.I(className="bi bi-check-circle me-1"), "Ready"],
                color="success",
                pill=True,
                className="fs-6",
            )
            if bundle.ready
            else dbc.Badge(
                "Partial – some files missing",
                color="warning",
                pill=True,
                className="fs-6",
            )
        )

        n_tissues = len(bundle.tissues)
        pair_msg  = (
            "Pair-detail DataFrames available – filter-aware ring plots enabled."
            if bundle.has_pair_details
            else "Pair-detail files not found – ring plots will use pre-aggregated comparison files."
        )
        alert_msg   = f"Loaded data for {n_tissues} tissue(s): {', '.join(bundle.tissues)}. " + pair_msg
        alert_color = "success" if bundle.ready else "warning"

        nav_hint = dbc.Alert(
            [
                html.I(className="bi bi-check-circle-fill me-2 text-success"),
                "Data ready.  ",
                dbc.Button(
                    [html.I(className="bi bi-bar-chart-line me-1"), "Open Explorer →"],
                    href="/explorer",
                    color="success",
                    size="sm",
                    className="ms-3",
                ),
            ],
            color="success",
            is_open=True,
            className="mb-0 mt-3",
        )

        return (
            metadata,
            summary,
            status_badge,
            alert_msg,
            alert_color,
            True,
            {"display": "block"},
            nav_hint,
        )

    # ------------------------------------------------------------------
    # 2. Export DataFrames to CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("export-alert", "children"),
        Output("export-alert", "color"),
        Output("export-alert", "is_open"),
        Input("btn-export-data",  "n_clicks"),
        State("input-export-path", "value"),
        State("store-data",        "data"),
        State("store-results-dir",   "data"),
        State("store-cell-fate-dir", "data"),
        prevent_initial_call=True,
    )
    def export_data(n_clicks, export_path, metadata, results_data, cell_fate_data):
        if not n_clicks:
            raise PreventUpdate

        if not export_path or not export_path.strip():
            return "Please enter an export directory path.", "warning", True

        results_dir   = (results_data   or {}).get("results_dir")
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")

        if not results_dir and not cell_fate_dir:
            return "No results loaded. Please collect data first.", "warning", True

        try:
            bundle: DataBundle = load_or_build_cache(
                cell_fate_dir=cell_fate_dir,
                results_dir=results_dir,
                force=False,
            )
        except Exception as exc:
            return f"Could not reload DataBundle: {exc}", "danger", True

        out_dir = Path(export_path.strip())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return f"Cannot create export directory: {exc}", "danger", True

        export_map = {
            "experimental_df.csv":         bundle.experimental_df,
            "cell_line_comparison.csv":    bundle.cell_line_comparison,
            "inhibitor_comparison.csv":    bundle.inhibitor_comparison,
            "cell_line_pair_details.csv":  bundle.cell_line_pair_details,
            "inhibitor_pair_details.csv":  bundle.inhibitor_pair_details,
            "roc_metrics.csv":             bundle.roc_metrics,
        }
        saved   = []
        skipped = []
        errors  = []

        for fname, df in export_map.items():
            if df is None or df.empty:
                skipped.append(fname)
                continue
            try:
                df.to_csv(out_dir / fname, index=False)
                saved.append(fname)
            except Exception as exc:
                errors.append(f"{fname}: {exc}")

        parts = []
        if saved:
            parts.append(f"Saved: {', '.join(saved)}")
        if skipped:
            parts.append(f"Skipped (empty): {', '.join(skipped)}")
        if errors:
            parts.append(f"Errors: {'; '.join(errors)}")

        color = "success" if not errors else "warning"
        msg   = f"Export to {out_dir}. " + "  ".join(parts)
        return msg, color, True
