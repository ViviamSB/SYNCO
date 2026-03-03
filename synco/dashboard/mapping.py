"""
mapping.py – UI metadata for Explorer tabs.

The (context, tab) → PlotSpec mapping lives in ``synco.dashboard.plot_registry``.
This module retains only the UI tab metadata (label, icon).
"""

# ---------------------------------------------------------------------------
# Tabs (shown in the Explorer main panel)
# ---------------------------------------------------------------------------
TABS = [
    {"id": "classification", "label": "Classification", "icon": "bi-table"},
    {"id": "performance",    "label": "Performance",    "icon": "bi-pie-chart"},
    {"id": "roc",            "label": "ROC / PR",       "icon": "bi-graph-up"},
    {"id": "distributions",  "label": "Distributions",  "icon": "bi-bar-chart"},
    {"id": "profiles",       "label": "Profiles",       "icon": "bi-bar-chart-steps"},
]
