"""
plot_cb.py – Callbacks for the Explorer page.

Callback map
------------
5.  store-cell-fate-dir changes   → populate tissue-selector-container
6.  tissue-select-dropdown change → store-results-dir
7.  store-results-dir / store-cell-fate-dir → populate filter-container
    (renders up to 4 fixed-ID dropdowns: cell-line, combination, drug, profile)
8.  any of the 4 filter dropdowns → store-filters
9.  btn-explore click             → populate all 5 tab-content divs with gallery cards
12. btn-reset click               → clear store-filters + all 5 tab-content divs
13. card-btn click (MATCH)        → render one plot card and update its output div
"""

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import MATCH, Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from synco.dashboard.adapters import build_gallery, render_one_plot
from synco.dashboard.callbacks.pipeline_cb import _scan_multi_tissue_root
from synco.dashboard.plot_registry import get_spec_by_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter data helpers
# ---------------------------------------------------------------------------

def _read_cell_lines(results_dir: str) -> list:
    """Return sorted unique cell-line names found in *results_dir*."""
    rd = Path(results_dir)
    candidates = [
        rd / "cell_line_comparison_results.csv",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                df = pd.read_csv(cand, nrows=5000)
                col = "cell_line" if "cell_line" in df.columns else df.columns[0]
                values = df[col].dropna().astype(str).unique()
                return sorted(v for v in values if v and v.lower() not in ("nan", "none", ""))
            except Exception:
                pass
    return []


def _read_combinations(results_dir: str) -> list:
    """Return sorted unique combination names found in *results_dir*."""
    rd = Path(results_dir)
    candidates = [
        rd / "inhibitor_combination_comparison_results.csv",
        rd / "combination_comparison_results.csv",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                df = pd.read_csv(cand, nrows=5000)
                col_candidates = ["inhibitor_combination", "combination", "inhibitor_group"]
                col = next((c for c in col_candidates if c in df.columns), df.columns[0])
                values = df[col].dropna().astype(str).unique()
                return sorted(v for v in values if v and v.lower() not in ("nan", "none", ""))
            except Exception:
                pass
    return []


def _read_global_options(filter_type: str, cell_fate_dir: str) -> list:
    """Return sorted unique drug names or inhibitor-profile names.

    Strategy (fastest-first):
    1. Check ``<cell_fate_dir>/synco_shared/`` for a shared CSV.
    2. Fall back to scanning up to 5 per-tissue ``synco_output/`` directories.
    """
    if filter_type == "drug":
        cols_to_try = ["drug_name_A", "drug_name_B", "drug_name"]
    else:  # "profile"
        cols_to_try = ["inhibitor_group_A", "inhibitor_group_B", "inhibitor_group"]

    values: set = set()

    # 1. Preferred: synco_shared inside cell_fate_dir
    shared_dir = Path(cell_fate_dir) / "synco_shared"
    shared_candidates = [
        shared_dir / "experimental_matrix_df.csv",
        shared_dir / "experimental_window_df.csv",
        shared_dir / "experimental_drug_names_synergies_df.csv",
        shared_dir / "experimental_full_df.csv",
    ]
    for cand in shared_candidates:
        if cand.exists():
            try:
                df = pd.read_csv(cand, nrows=10_000)
                cols = [c for c in cols_to_try if c in df.columns]
                for col in cols:
                    values.update(df[col].dropna().astype(str).unique())
                if values:
                    break
            except Exception:
                pass

    if values:
        return sorted(v for v in values if v and v.lower() not in ("nan", "none", ""))

    # 2. Fallback: scan up to 5 per-tissue synco_output directories
    tissue_dirs = _scan_multi_tissue_root(Path(cell_fate_dir))
    per_tissue_candidates = [
        "experimental_drug_names_synergies_df.csv",
        "experimental_full_df.csv",
        "experimental_matrix_df.csv",
        "experimental_window_df.csv",
    ]
    for tissue_output in tissue_dirs[:5]:
        rd = Path(tissue_output)
        for fname in per_tissue_candidates:
            cand = rd / fname
            if cand.exists():
                try:
                    df = pd.read_csv(cand, nrows=5000)
                    cols = [c for c in cols_to_try if c in df.columns]
                    for col in cols:
                        values.update(df[col].dropna().astype(str).unique())
                    break
                except Exception:
                    pass

    return sorted(v for v in values if v and v.lower() not in ("nan", "none", ""))


def _make_filter_section(label: str, dropdown_id: str, options: list) -> html.Div:
    """Build one labelled filter row, hidden via CSS when *options* is empty."""
    return html.Div(
        [
            html.Hr(className="my-2"),
            html.Small(label, className="text-muted fw-semibold d-block mb-1"),
            dbc.Select(
                id=dropdown_id,
                options=[{"label": v, "value": v} for v in options],
                value=None,
                placeholder="— all —",
                size="sm",
            ),
            html.Small(
                "Optional – leave blank to show all.",
                className="text-muted d-block mt-1",
                style={"fontSize": "0.72rem"},
            ),
        ],
        style={"display": "block" if options else "none"},
    )


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_plot_callbacks(app: dash.Dash) -> None:
    """Attach Explorer-page callbacks to *app*."""

    # ------------------------------------------------------------------
    # 5.  Populate tissue-selector dropdown from cell_fate_dir
    # ------------------------------------------------------------------
    @app.callback(
        Output("tissue-selector-container", "children"),
        Input("store-cell-fate-dir",        "data"),
    )
    def populate_tissue_selector(cell_fate_data):
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")
        if not cell_fate_dir:
            return []

        tissue_dirs = _scan_multi_tissue_root(Path(cell_fate_dir))
        if not tissue_dirs:
            return []

        tissue_names = [t.parent.name for t in tissue_dirs]
        options = (
            [{"label": "— All tissues —", "value": "all"}]
            + [{"label": name, "value": name} for name in tissue_names]
        )

        return [
            html.Small(
                "Active tissue",
                className="text-muted fw-semibold d-block mb-1",
            ),
            dbc.Select(
                id="tissue-select-dropdown",
                options=options,
                value="all",
                size="sm",
            ),
            html.Small(
                "Select a tissue for single-tissue views; keep 'All tissues' for cross-tissue plots.",
                className="text-muted d-block mt-1 mb-2",
                style={"fontSize": "0.72rem"},
            ),
            html.Hr(className="my-2"),
        ]

    # ------------------------------------------------------------------
    # 6.  Update results_dir store when a tissue is picked
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-results-dir",      "data", allow_duplicate=True),
        Input("tissue-select-dropdown",  "value"),
        State("store-cell-fate-dir",     "data"),
        prevent_initial_call=True,
    )
    def select_tissue(tissue_name, cell_fate_data):
        if tissue_name is None:
            raise PreventUpdate
        if tissue_name == "all":
            return {"results_dir": None}
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")
        if not cell_fate_dir:
            raise PreventUpdate
        results_dir = Path(cell_fate_dir) / tissue_name / "synco_output"
        return {"results_dir": str(results_dir)}

    # ------------------------------------------------------------------
    # 7.  Populate filter dropdowns (cell line, combination, drug, profile)
    # ------------------------------------------------------------------
    @app.callback(
        Output("filter-container",   "children"),
        Input("store-results-dir",   "data"),
        Input("store-cell-fate-dir", "data"),
    )
    def populate_filters(results_data, cell_fate_data):
        results_dir   = (results_data   or {}).get("results_dir")
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")

        cell_lines   = _read_cell_lines(results_dir)   if results_dir else []
        combinations = _read_combinations(results_dir) if results_dir else []

        scan_root = cell_fate_dir
        if not scan_root and results_dir:
            scan_root = str(Path(results_dir).parent.parent)
        drugs    = _read_global_options("drug",    scan_root) if scan_root else []
        profiles = _read_global_options("profile", scan_root) if scan_root else []

        sections = [
            _make_filter_section("Cell line",        "filter-cell-line",   cell_lines),
            _make_filter_section("Combination",      "filter-combination", combinations),
            _make_filter_section("Drug",             "filter-drug",        drugs),
            _make_filter_section("Profile category", "filter-profile",     profiles),
        ]

        if not any([cell_lines, combinations, drugs, profiles]):
            return [
                html.Small(
                    "Options appear here once results are loaded.",
                    className="text-muted",
                )
            ] + sections

        intro = html.Small(
            "All filters are optional – leave blank to show all.",
            className="text-muted d-block mb-1",
            style={"fontSize": "0.72rem"},
        )
        return [intro] + sections

    # ------------------------------------------------------------------
    # 8.  Persist all four filter values to store-filters
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-filters",      "data"),
        Input("filter-cell-line",    "value"),
        Input("filter-combination",  "value"),
        Input("filter-drug",         "value"),
        Input("filter-profile",      "value"),
        prevent_initial_call=True,
    )
    def update_filters(cell_line, combination, drug, profile):
        return {
            "cell_line":   cell_line   or None,
            "combination": combination or None,
            "drug":        drug        or None,
            "profile":     profile     or None,
        }

    # ------------------------------------------------------------------
    # 9.  Build plot gallery on "Explore" button click.
    #     All 5 tab-content divs are updated at once so switching between
    #     tabs never requires re-clicking Explore.
    # ------------------------------------------------------------------
    @app.callback(
        Output("tab-content-classification", "children"),
        Output("tab-content-performance",    "children"),
        Output("tab-content-roc",            "children"),
        Output("tab-content-distributions",  "children"),
        Output("tab-content-profiles",       "children"),
        Input("btn-explore",          "n_clicks"),
        State("store-results-dir",    "data"),
        State("store-cell-fate-dir",  "data"),
        State("store-filters",        "data"),
        prevent_initial_call=True,
    )
    def explore(n_clicks, results_data, cell_fate_data, filters_data):
        if not n_clicks:
            raise PreventUpdate

        results_dir   = (results_data   or {}).get("results_dir")
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")
        context = "cross_tissue" if not results_dir else "tissue"
        primary_dir = cell_fate_dir if context == "cross_tissue" else results_dir

        if not primary_dir:
            msg = dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    "Load results on the Setup page first.",
                ],
                color="warning",
                className="mt-3",
            )
            return msg, msg, msg, msg, msg

        active_filters = {k: v for k, v in (filters_data or {}).items() if v} or None

        tabs = ["classification", "performance", "roc", "distributions", "profiles"]
        return tuple(
            html.Div(build_gallery(context, tab, primary_dir, active_filters=active_filters))
            for tab in tabs
        )

    # ------------------------------------------------------------------
    # 12. Reset button – clear all filters and all 5 tab-content divs
    # ------------------------------------------------------------------
    @app.callback(
        Output("store-filters",              "data",     allow_duplicate=True),
        Output("tab-content-classification", "children", allow_duplicate=True),
        Output("tab-content-performance",    "children", allow_duplicate=True),
        Output("tab-content-roc",            "children", allow_duplicate=True),
        Output("tab-content-distributions",  "children", allow_duplicate=True),
        Output("tab-content-profiles",       "children", allow_duplicate=True),
        Input("btn-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_filters(_n_clicks):
        empty = {"cell_line": None, "combination": None, "drug": None, "profile": None}
        return empty, [], [], [], [], []

    # ------------------------------------------------------------------
    # 13. Per-card "Render" button (MATCH pattern).
    #     Each card button updates only its own output div.
    # ------------------------------------------------------------------
    @app.callback(
        Output({"type": "card-output", "index": MATCH}, "children"),
        Input({"type": "card-btn",     "index": MATCH}, "n_clicks"),
        State("store-results-dir",    "data"),
        State("store-cell-fate-dir",  "data"),
        State("store-filters",        "data"),
        prevent_initial_call=True,
    )
    def render_plot_card(n_clicks, results_data, cell_fate_data, filters_data):
        if not n_clicks:
            raise PreventUpdate

        plot_id = ctx.triggered_id["index"]
        spec    = get_spec_by_id(plot_id)
        if spec is None:
            return dbc.Alert(f"Unknown plot id: {plot_id!r}", color="danger")

        results_dir   = (results_data   or {}).get("results_dir")
        cell_fate_dir = (cell_fate_data or {}).get("cell_fate_dir")
        primary_dir   = cell_fate_dir if spec.input_type == "cell_fate_dir" else results_dir

        active_filters = {k: v for k, v in (filters_data or {}).items() if v} or None

        return render_one_plot(spec, primary_dir, filters=active_filters)
