"""
explorer.py – Stage 2 + 3: Coverage-level navigation and Evaluation-type tabs.

Layout
------
┌──────────────────────────────────────────────────────┐
│  Coverage sidebar  │  Eval tabs                      │
│  ──────────────    │  ─────────────────────────────  │
│  Global            │  [Classification][Performance]… │
│  Tissue            │                                  │
│  Cell Line         │  ┌─────────────────────────┐    │
│  Combinations      │  │   Plot panel (scrollable)│   │
│  Experimental Drugs│  └─────────────────────────┘    │
│  Drug Profiles     │                                  │
└──────────────────────────────────────────────────────┘
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from synco.dashboard.components.coverage_nav import make_coverage_nav
from synco.dashboard.components.eval_tabs import make_eval_tabs

dash.register_page(__name__, path="/explorer", title="SYNCO – Explorer", order=1)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(**kwargs):  # noqa: ARG001
    return dbc.Container(
        [
            dbc.Row(
                [
                    # ── Left sidebar: coverage navigation ─────────────────
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    make_coverage_nav(active="global"),
                                    className="px-2 py-2",
                                ),
                                className="sticky-top",
                                style={"top": "70px"},
                            ),
                        ],
                        width=2,
                        className="pt-3",
                    ),

                    # ── Main content: eval tabs + plot panel ──────────────
                    dbc.Col(
                        [
                            # Row: back link + title
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                dcc.Link(
                                                    [html.I(className="bi bi-arrow-left me-1"), "Setup"],
                                                    href="/",
                                                    className="text-muted small",
                                                ),
                                                html.Span(
                                                    " / Explorer",
                                                    className="text-muted small",
                                                ),
                                            ],
                                            className="mt-3 mb-1",
                                        ),
                                    ]
                                )
                            ),

                            # Eval tabs (rebuilt dynamically by plot_cb.rebuild_eval_tabs)
                            html.Div(
                                make_eval_tabs(coverage="global", active_tab="classification"),
                                id="eval-tabs-container",
                                className="mb-0",
                            ),

                            # Plot panel
                            dbc.Card(
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Div(
                                            _placeholder(),
                                            id="plot-panel",
                                        ),
                                        type="circle",
                                        color="#0d6efd",
                                    ),
                                    className="p-3",
                                ),
                                className="rounded-0 rounded-bottom border-top-0",
                            ),
                        ],
                        width=10,
                    ),
                ],
                className="g-0",
            )
        ],
        fluid=True,
        className="px-3",
    )


def _placeholder() -> dbc.Alert:
    return dbc.Alert(
        [
            html.I(className="bi bi-info-circle me-2"),
            "Select a coverage level and evaluation tab to generate plots.",
        ],
        color="light",
        className="mt-2",
    )
