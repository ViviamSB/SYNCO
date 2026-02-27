"""
eval_tabs.py – Top-level evaluation-type tab bar for the Explorer page.
"""

import dash_bootstrap_components as dbc

from synco.dashboard.mapping import EVAL_LEVELS, get_valid_evals


def make_eval_tabs(coverage: str = "global", active_tab: str = "classification") -> dbc.Tabs:
    """
    Return a ``dbc.Tabs`` bar for selecting the evaluation type.

    Tabs that are not valid for the current *coverage* level are disabled.

    Parameters
    ----------
    coverage   : Currently active coverage level ID.
    active_tab : Currently selected tab ID.
    """
    valid = get_valid_evals(coverage)
    tabs = []
    resolved_active = active_tab if active_tab in valid else (valid[0] if valid else None)

    for item in EVAL_LEVELS:
        is_valid = item["id"] in valid
        tabs.append(
            dbc.Tab(
                label=item["label"],
                tab_id=item["id"],
                disabled=not is_valid,
            )
        )

    return dbc.Tabs(
        tabs,
        id="eval-tabs",
        active_tab=resolved_active,
        className="mb-0 border-bottom-0",
    )
