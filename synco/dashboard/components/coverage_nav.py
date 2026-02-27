"""
coverage_nav.py – Left-sidebar coverage-level navigation component.

Each button uses a dict-based ID ``{"type": "coverage-btn", "index": id}``
so the plot callback can use Dash pattern-matching (ALL) to detect which
coverage level is active.
"""

import dash_bootstrap_components as dbc
from dash import html

from synco.dashboard.mapping import COVERAGE_LEVELS


def make_coverage_nav(active: str = "global") -> html.Div:
    """
    Return a vertical list of coverage-level buttons.

    Parameters
    ----------
    active : ID of the currently active coverage level.
    """
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

    return html.Div(
        [
            html.P(
                "Coverage level",
                className="text-muted small fw-semibold text-uppercase mb-2",
            ),
            html.Div(buttons, id="coverage-nav-container"),
        ],
        className="pt-3",
    )
