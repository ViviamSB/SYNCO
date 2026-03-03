"""
explorer.py – Flat sidebar + 5-tab Explorer page.

Sidebar layout
--------------
┌──────────────────────────────────────┐
│  Tissue                              │
│  [ — All tissues — ▼ ]             │  (populated by callback)
│ ─────────────────────────────────────│
│  Filters (optional)                  │
│  Cell line / Combination / Drug /    │
│  Profile dropdowns                   │
│ ─────────────────────────────────────│
│  [ Explore ▶ ]  [ Reset ]           │
└──────────────────────────────────────┘

Main panel: 5 static dbc.Tabs with fixed-ID content divs.
Each tab's html.Div is always in the DOM — switching tabs
does not require re-clicking Explore.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from synco.dashboard.mapping import TABS

dash.register_page(__name__, path="/explorer", title="SYNCO – Explorer", order=2)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(**kwargs):  # noqa: ARG001
    sidebar = dbc.Card(
        dbc.CardBody(
            [
                # ── Tissue selection ─────────────────────────────────────
                html.Small(
                    [html.I(className="bi bi-geo-alt me-1 text-muted"), "Tissue"],
                    className="fw-semibold d-block mb-1",
                ),
                html.Div(
                    id="tissue-selector-container",
                    children=html.Small(
                        [
                            html.I(className="bi bi-info-circle me-1"),
                            "Load results on the Setup page first.",
                        ],
                        className="text-muted",
                    ),
                ),

                html.Hr(className="my-2"),

                # ── Filters (optional) ───────────────────────────────────
                html.Small(
                    [html.I(className="bi bi-funnel me-1 text-muted"), "Filters (optional)"],
                    className="fw-semibold d-block mb-1",
                ),
                html.Div(
                    id="filter-container",
                    children=html.Small(
                        "Options appear here once results are loaded.",
                        className="text-muted",
                    ),
                ),

                html.Hr(className="my-2"),

                # ── Action buttons ───────────────────────────────────────
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            [html.I(className="bi bi-bar-chart-line me-2"), "Explore"],
                            id="btn-explore",
                            color="primary",
                            size="sm",
                            n_clicks=0,
                            className="flex-grow-1",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-x-circle me-1"), "Reset"],
                            id="btn-reset",
                            color="secondary",
                            outline=True,
                            size="sm",
                            n_clicks=0,
                        ),
                    ],
                    className="w-100",
                ),
            ],
            className="p-2",
        ),
        className="sticky-top",
        style={"top": "70px"},
    )

    # ── 5 static tabs, each with a fixed-ID content div ──────────────────
    tab_items = []
    for t in TABS:
        tab_items.append(
            dbc.Tab(
                html.Div(
                    _placeholder(),
                    id=f"tab-content-{t['id']}",
                    style={"minHeight": "100px"},
                ),
                label=t["label"],
                tab_id=t["id"],
            )
        )

    main_content = dbc.Col(
        [
            # Breadcrumb
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            dcc.Link(
                                [html.I(className="bi bi-arrow-left me-1"), "Setup"],
                                href="/",
                                className="text-muted small",
                            ),
                            html.Span(" / Explorer", className="text-muted small"),
                        ],
                        className="mt-3 mb-2",
                    )
                )
            ),

            # Tabs (all content divs are always in the DOM)
            dbc.Tabs(
                tab_items,
                id="explorer-tabs",
                active_tab="classification",
            ),
        ],
        width=9,
    )

    return dbc.Container(
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className="pt-3"),
                main_content,
            ],
            className="g-0",
        ),
        fluid=True,
        className="px-3",
    )


def _placeholder() -> dbc.Alert:
    return dbc.Alert(
        [
            html.I(className="bi bi-info-circle me-2"),
            "Set options in the left panel and click ",
            html.Strong("Explore"),
            " to generate plots.",
        ],
        color="light",
        className="mt-2",
    )
