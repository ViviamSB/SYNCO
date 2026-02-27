"""
adapters.py – Bridge between SYNCO plotting functions and Dash components.

Calling ``run_plot_adapter(coverage, eval_tab, results_dir, cell_fate_dir)``
will:
  1. Resolve the correct plotting function from ``mapping.PLOT_MAP``.
  2. Call it with the right input directory and a ``plots_dir`` that lives
     next to the results.
  3. Collect any new files written to ``plots_dir`` (.html / .png).
  4. Return a list of Dash components ready to embed in the layout
     (``dcc.Graph`` for returned Plotly figures, ``html.Iframe`` for
     saved HTML, ``html.Img`` for saved PNG).
"""

import base64
import logging
import os
import time
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

import synco.plotting as _plotting_module
from synco.dashboard.mapping import PLOT_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_plot_func(func_name: str):
    """Retrieve a plotting callable from ``synco.plotting`` by name."""
    func = getattr(_plotting_module, func_name, None)
    if func is None:
        raise AttributeError(f"synco.plotting has no function '{func_name}'")
    return func


def _embed_html_file(filepath: str) -> html.Iframe:
    """Serve an HTML plot file via the ``/serve-plot/`` Flask route."""
    # Normalise path separators so the URL works on Windows too
    url_path = Path(filepath).as_posix()
    return html.Iframe(
        src=f"/serve-plot/{url_path}",
        style={"width": "100%", "height": "640px", "border": "none"},
    )


def _embed_png_file(filepath: str) -> html.Img:
    """Inline-encode a PNG as base64 and return an <img> component."""
    with open(filepath, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={
            "width": "100%",
            "maxWidth": "1100px",
            "display": "block",
            "margin": "auto",
        },
    )


def _collect_new_files(plots_dir: str, since: float) -> list:
    """Return ``(ext, filepath)`` tuples for files created/modified after *since*."""
    plots_path = Path(plots_dir)
    if not plots_path.exists():
        return []
    results = []
    for fp in sorted(plots_path.iterdir()):
        if fp.is_file() and fp.stat().st_mtime >= since:
            ext = fp.suffix.lower()
            if ext in (".html", ".png"):
                results.append((ext, str(fp)))
    return results


def _wrap_in_card(child) -> dbc.Card:
    return dbc.Card(dbc.CardBody(child), className="mb-3 shadow-sm")


def _figures_from_result(result) -> list:
    """
    Extract ``dcc.Graph`` components from a dict returned by a plotting
    function (e.g. ``make_multi_tissue_plots``).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return []

    components = []

    def _add_fig(fig, title=""):
        if isinstance(fig, go.Figure):
            components.append(
                _wrap_in_card(
                    [
                        html.H6(title, className="text-muted mb-2") if title else None,
                        dcc.Graph(figure=fig, style={"height": "600px"}),
                    ]
                )
            )

    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_fig in value.items():
                    _add_fig(sub_fig, title=f"{key} – {sub_key}".replace("_", " ").title())
            else:
                _add_fig(value, title=key.replace("_", " ").title())

    return components


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_plot_adapter(
    coverage: str,
    eval_tab: str,
    results_dir: str,
    cell_fate_dir: str | None = None,
) -> list:
    """
    Resolve and call the correct plotting function, then return a list
    of Dash components for display.

    Parameters
    ----------
    coverage      : Coverage level ID (e.g. ``"global"``, ``"cell_line"``).
    eval_tab      : Evaluation tab ID (e.g. ``"classification"``, ``"roc"``).
    results_dir   : Path to ``synco_output/`` from a pipeline run.
    cell_fate_dir : Parent directory containing per-tissue result folders
                    (needed only for tissue-level coverage).

    Returns
    -------
    list of Dash components.
    """
    key = (coverage, eval_tab)

    if key not in PLOT_MAP:
        return [
            dbc.Alert(
                f"No plot is available for coverage '{coverage}' "
                f"with evaluation '{eval_tab}'.",
                color="secondary",
                className="mt-3",
            )
        ]

    entry = PLOT_MAP[key]
    func_name: str = entry["func"]
    extra_kwargs: dict = entry["kwargs"].copy()
    input_type: str = entry["input"]

    # ------------------------------------------------------------------
    # Resolve input directory
    # ------------------------------------------------------------------
    if input_type == "cell_fate_dir":
        if not cell_fate_dir:
            return [
                dbc.Alert(
                    "Tissue-level plots require a Cell Fate Directory. "
                    "Please set it on the Setup page.",
                    color="warning",
                    className="mt-3",
                )
            ]
        primary_dir = cell_fate_dir
    else:
        if not results_dir:
            return [
                dbc.Alert(
                    "No results directory loaded. Please run the pipeline or "
                    "load an existing output folder on the Setup page.",
                    color="warning",
                    className="mt-3",
                )
            ]
        primary_dir = results_dir

    # ------------------------------------------------------------------
    # Prepare output directory for plots
    # ------------------------------------------------------------------
    plots_dir = os.path.join(results_dir or primary_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Call plotting function
    # ------------------------------------------------------------------
    func = _get_plot_func(func_name)
    before = time.time() - 0.5  # small buffer for file-system timestamp precision

    logger.info("Calling %s(primary_dir=%r, plots_dir=%r)", func_name, primary_dir, plots_dir)

    try:
        result = func(primary_dir, plots_dir=plots_dir, show=False, **extra_kwargs)
    except Exception as exc:
        logger.exception("Error in %s", func_name)
        return [
            dbc.Alert(
                [
                    html.Strong("Plot error: "),
                    str(exc),
                ],
                color="danger",
                className="mt-3",
            )
        ]

    # ------------------------------------------------------------------
    # Collect components
    # ------------------------------------------------------------------
    components: list = []

    # 1. Figures returned directly (make_multi_tissue_plots returns a dict)
    if result is not None:
        components.extend(_figures_from_result(result))

    # 2. Files written to disk
    new_files = _collect_new_files(plots_dir, before)
    for ext, filepath in new_files:
        if ext == ".html":
            components.append(_wrap_in_card(_embed_html_file(filepath)))
        elif ext == ".png":
            components.append(_wrap_in_card(_embed_png_file(filepath)))

    if not components:
        components = [
            dbc.Alert(
                "The plotting function completed but produced no visible output. "
                "Ensure the results directory contains the expected files.",
                color="info",
                className="mt-3",
            )
        ]

    return components
