"""
setup.py – Stage 1: Configure & Run (or Load existing results).

Two modes are available, controlled by a RadioItems toggle:
  • "run"  – Full configuration form + "Run pipeline" button.
  • "load" – Single directory input + "Load existing output" button.

After a successful run or load the status section shows a link to the
Explorer page.
"""

import dash
from dash import Input, Output, State, callback, dcc, html, no_update
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

            # ── Mode toggle ────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    dbc.RadioItems(
                        id="setup-mode-toggle",
                        options=[
                            {"label": "  Run pipeline",          "value": "run"},
                            {"label": "  Load existing results", "value": "load"},
                        ],
                        value="load",
                        inline=True,
                        input_checked_class_name="bg-primary border-primary",
                        className="mb-4",
                    ),
                    width=12,
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
                style={"display": "none"},
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
    Output("section-run",  "style"),
    Output("section-load", "style"),
    Input("setup-mode-toggle", "value"),
)
def toggle_sections(mode):
    if mode == "run":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


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
            ],
            color="success",
            className="mb-0",
            is_open=True,
        )
    return None
