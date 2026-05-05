"""
setup.py – Stage 1: Configure & Run (or Load existing results).

Both the pipeline configuration form and the load-path input are always
visible.  After a successful run or load the status section shows links to
the Data and Explorer pages.
"""

import dash
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from synco.dashboard.components.config_form import make_config_form

dash.register_page(__name__, path="/", title="SYNCO – Setup", order=0)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(**kwargs):  # noqa: ARG001  (Dash passes URL kwargs)
    return dbc.Container(
        [
            # ── Page header ────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    [
                        html.H3("Setup", className="mt-4 mb-1"),
                        html.P(
                            "Configure the SYNCO pipeline parameters and run it, "
                            "or load results from a previous run.",
                            className="text-muted mb-3",
                        ),
                    ]
                )
            ),

            # ── Run pipeline section ───────────────────────────────────────
            html.Div(
                id="section-run",
                children=[
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Pipeline configuration", className="card-title mb-3"),
                                make_config_form(),
                                html.Hr(),
                                dbc.Button(
                                    [html.I(className="bi bi-play-fill me-2"), "Run pipeline"],
                                    id="btn-run-pipeline",
                                    color="primary",
                                    className="me-2",
                                    n_clicks=0,
                                ),
                                dbc.Alert(
                                    id="run-alert",
                                    is_open=False,
                                    dismissable=True,
                                    className="mt-3 mb-0",
                                ),
                            ]
                        ),
                        className="mb-4",
                    ),
                ],
            ),

            # ── Load existing section ──────────────────────────────────────
            html.Div(
                id="section-load",
                children=[
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Load existing results", className="card-title mb-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Results directory"),
                                                dbc.FormText(
                                                    "Path to a single synco_output/ folder "
                                                    "or to a parent directory containing "
                                                    "per-tissue results – auto-detected.",
                                                    color="secondary",
                                                ),
                                            ],
                                            width=3,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id="input-load-path",
                                                type="text",
                                                placeholder=(
                                                    "e.g. /path/to/synco_output   "
                                                    "or /path/to/synco_output_window"
                                                ),
                                                debounce=True,
                                                persistence=True,
                                                persistence_type="session",
                                            ),
                                            width=9,
                                        ),
                                    ],
                                    className="mb-3 align-items-center",
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-folder2-open me-2"), "Load results"],
                                    id="btn-load-results",
                                    color="secondary",
                                    n_clicks=0,
                                ),
                                dbc.Alert(
                                    id="load-alert",
                                    is_open=False,
                                    dismissable=True,
                                    className="mt-3 mb-0",
                                ),
                            ]
                        ),
                        className="mb-4",
                    ),
                ],
            ),

            # ── Open Data link (shown once a directory is loaded) ──────────
            html.Div(id="status-explorer-link", className="mb-4"),

            # Polling interval (starts disabled; enabled once pipeline starts)
            dcc.Interval(
                id="poll-interval",
                interval=1_000,   # 1 second
                disabled=True,
            ),
        ],
        fluid=True,
        className="px-4",
    )


# ---------------------------------------------------------------------------
# Local callbacks (no app reference needed – use @callback)
# ---------------------------------------------------------------------------

@callback(
    Output("status-explorer-link", "children"),
    Input("store-results-dir",     "data"),
    Input("store-cell-fate-dir",   "data"),
)
def update_explorer_link(results_data, cell_fate_data):
    results_dir  = (results_data   or {}).get("results_dir")
    cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")

    # Show the button as soon as either store has data
    if results_dir or cell_fate_dir:
        ready_label = results_dir or cell_fate_dir
        return dbc.Alert(
            [
                html.I(className="bi bi-check-circle-fill me-2 text-success"),
                f"Results ready: {ready_label}  ",
                dbc.Button(
                    [html.I(className="bi bi-table me-1"), "Open Data →"],
                    href="/data",
                    external_link=False,
                    color="success",
                    size="sm",
                    className="ms-3",
                ),
                dbc.Button(
                    [html.I(className="bi bi-bar-chart-line me-1"), "Open Explorer →"],
                    href="/explorer",
                    external_link=False,
                    color="primary",
                    size="sm",
                    className="ms-2",
                ),
            ],
            color="success",
            className="mb-0",
            is_open=True,
        )
    return None


@callback(
    Output("run-alert", "children",  allow_duplicate=True),
    Output("run-alert", "color",     allow_duplicate=True),
    Output("run-alert", "is_open",   allow_duplicate=True),
    Input("store-pipeline-status",   "data"),
    prevent_initial_call=True,
)
def update_run_alert_from_status(status_data):
    """Mirror pipeline-thread status messages into the run-alert banner."""
    if not status_data:
        raise PreventUpdate
    status  = (status_data or {}).get("status", "")
    message = (status_data or {}).get("message", "")
    if not message:
        raise PreventUpdate
    color_map = {"running": "info", "done": "success", "error": "danger"}
    color = color_map.get(status, "secondary")
    icon_map = {
        "running": "bi bi-hourglass-split me-2",
        "done":    "bi bi-check-circle-fill me-2",
        "error":   "bi bi-exclamation-triangle-fill me-2",
    }
    icon = icon_map.get(status, "bi bi-info-circle me-2")
    return [html.I(className=icon), message], color, True
