"""
plot_cb.py – Callbacks for the Explorer page:
  • Updating the coverage store when a sidebar button is clicked.
  • Re-building the evaluation tab bar when coverage changes.
  • Rendering the plot panel when coverage or evaluation changes.
"""

import dash
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from synco.dashboard.adapters import run_plot_adapter
from synco.dashboard.components.coverage_nav import make_coverage_nav
from synco.dashboard.components.eval_tabs import make_eval_tabs
from synco.dashboard.mapping import get_valid_evals, COVERAGE_LEVELS


def register_plot_callbacks(app: dash.Dash) -> None:
    """Attach Explorer-page callbacks to *app*."""

    # ------------------------------------------------------------------
    # 1.  Update coverage store when a sidebar button is clicked
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-coverage", "data"),
        Input({"type": "coverage-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def update_coverage(n_clicks_list):
        if not ctx.triggered or all(n is None for n in n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            raise PreventUpdate

        return {"level": triggered["index"]}

    # ------------------------------------------------------------------
    # 2.  Re-build evaluation tabs when coverage changes
    # ------------------------------------------------------------------
    @app.callback(
        Output("eval-tabs-container", "children"),
        Input("store-coverage",       "data"),
        State("store-eval",           "data"),
    )
    def rebuild_eval_tabs(coverage_data, eval_data):
        coverage = (coverage_data or {}).get("level", "global")
        current_tab = (eval_data or {}).get("tab", "classification")
        valid = get_valid_evals(coverage)
        # Fall back to first valid tab if current tab is not valid
        active = current_tab if current_tab in valid else (valid[0] if valid else None)
        return make_eval_tabs(coverage=coverage, active_tab=active)

    # ------------------------------------------------------------------
    # 3.  Keep eval store in sync with active tab
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-eval", "data"),
        Input("eval-tabs",   "active_tab"),
        prevent_initial_call=False,
    )
    def update_eval_store(active_tab):
        if active_tab is None:
            raise PreventUpdate
        return {"tab": active_tab}

    # ------------------------------------------------------------------
    # 4.  Update sidebar button styles to reflect active coverage
    # ------------------------------------------------------------------
    @app.callback(
        Output("coverage-nav-container", "children"),
        Input("store-coverage", "data"),
    )
    def update_nav_styles(coverage_data):
        active = (coverage_data or {}).get("level", "global")
        buttons = []
        for item in COVERAGE_LEVELS:
            is_active = item["id"] == active
            buttons.append(
                dbc.Button(
                    [
                        html.I(className=f"{item['icon']} me-2"),
                        item["label"],
                    ],
                    id={"type": "coverage-btn", "index": item["id"]},
                    color="primary" if is_active else "light",
                    outline=not is_active,
                    className="text-start mb-1 w-100",
                    size="sm",
                    n_clicks=0,
                )
            )
        return buttons

    # ------------------------------------------------------------------
    # 5.  Render plots
    # ------------------------------------------------------------------
    @app.callback(
        Output("plot-panel", "children"),
        Input("store-coverage",    "data"),
        Input("store-eval",        "data"),
        Input("store-results-dir", "data"),
        State("store-cell-fate-dir", "data"),
    )
    def render_plot(coverage_data, eval_data, results_data, cell_fate_data):
        coverage    = (coverage_data   or {}).get("level", "global")
        eval_tab    = (eval_data       or {}).get("tab",   "classification")
        results_dir = (results_data    or {}).get("results_dir")
        cell_fate   = (cell_fate_data  or {}).get("cell_fate_dir")

        if results_dir is None and coverage != "tissue":
            return dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    "No results loaded. Use the ",
                    html.A("Setup page", href="/"),
                    " to run the pipeline or load an existing output folder.",
                ],
                color="info",
                className="mt-4",
            )

        components = run_plot_adapter(
            coverage=coverage,
            eval_tab=eval_tab,
            results_dir=results_dir or "",
            cell_fate_dir=cell_fate,
        )
        return html.Div(components)

    # ------------------------------------------------------------------
    # 6.  Status badge on Setup page
    # ------------------------------------------------------------------
    @app.callback(
        Output("pipeline-status-badge", "children"),
        Output("pipeline-status-badge", "color"),
        Input("store-pipeline-status",  "data"),
    )
    def update_status_badge(status_data):
        data = status_data or {}
        status = data.get("status", "idle")
        message = data.get("message", "")

        colour_map = {
            "idle":    "secondary",
            "running": "warning",
            "done":    "success",
            "error":   "danger",
        }
        label_map = {
            "idle":    "Idle",
            "running": f"Running… {message}",
            "done":    message or "Done",
            "error":   f"Error: {message}",
        }
        return label_map.get(status, status), colour_map.get(status, "secondary")
