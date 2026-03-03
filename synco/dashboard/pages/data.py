"""
data.py – Stage 2: Collect and inspect pipeline outputs.

Data collection is triggered automatically when the user navigates here
from the Setup page.  The Reload button forces a cache rebuild.

The page shows:
  • A summary table: DataFrame name, row count, column names.
  • Which JSON dictionaries were loaded.
  • Any missing / warning items.
  • A "Save / Export" section to write the full DataFrames to a directory.

Note: ``build_summary_content`` lives in callbacks/data_cb.py, NOT here.
Importing it from a page module would cause Dash to register the page
under two module names and raise a duplicate-path error.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/data", title="SYNCO – Data", order=1)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(**kwargs):  # noqa: ARG001
    return dbc.Container(
        [
            # ── Page header ────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    [
                        html.H3("Data", className="mt-4 mb-1"),
                        html.P(
                            "Collect all pipeline outputs into memory, review the available "
                            "DataFrames, and optionally export them before exploring plots.",
                            className="text-muted mb-3",
                        ),
                    ]
                )
            ),

            # ── Control row ────────────────────────────────────────────────
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-cloud-download me-2"), "Collect Data"],
                            id="btn-collect-data",
                            color="primary",
                            n_clicks=0,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-1"), "Reload"],
                            id="btn-reload-data",
                            color="secondary",
                            outline=True,
                            n_clicks=0,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Div(id="data-status-badge"),
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                ],
                className="mb-4 g-2 align-items-center",
            ),

            # ── Alert / feedback ───────────────────────────────────────────
            dbc.Alert(
                id="data-alert",
                is_open=False,
                dismissable=True,
                className="mb-3",
            ),

            # ── DataFrame summary ──────────────────────────────────────────
            dcc.Loading(
                html.Div(
                    id="data-summary-container",
                    children=_placeholder(),
                ),
                type="dot",
                color="#0d6efd",
            ),

            # ── Export section ─────────────────────────────────────────────
            html.Div(
                id="data-export-section",
                style={"display": "none"},
                children=[
                    html.Hr(),
                    html.H5(
                        [html.I(className="bi bi-download me-2"), "Save DataFrames"],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Export directory"),
                                    dbc.FormText(
                                        "All DataFrames will be written as CSV files to this folder.",
                                        color="secondary",
                                    ),
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="input-export-path",
                                    type="text",
                                    placeholder="e.g. /path/to/output_folder",
                                    debounce=True,
                                ),
                                width=7,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-floppy me-1"), "Save"],
                                    id="btn-export-data",
                                    color="success",
                                    n_clicks=0,
                                ),
                                width=2,
                            ),
                        ],
                        className="mb-3 align-items-end",
                    ),
                    dbc.Alert(
                        id="export-alert",
                        is_open=False,
                        dismissable=True,
                        className="mb-3",
                    ),
                ],
            ),

            # ── Navigation hint ────────────────────────────────────────────
            html.Div(id="data-nav-hint"),
        ],
        fluid=True,
        className="px-4",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _placeholder() -> html.Div:
    return dbc.Alert(
        [
            html.I(className="bi bi-hourglass-split me-2"),
            "Loading data… if this persists, ensure results are loaded on the ",
            html.Strong("Setup"),
            " page and click ",
            html.Strong("Collect Data"),
            ".",
        ],
        color="light",
        className="mt-2",
    )
