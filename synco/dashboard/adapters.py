"""
adapters.py – Bridge between the plot registry and Dash components.

Public API
----------
``build_gallery(context, tab, primary_dir, active_filters)``
    Returns a list of ``dbc.Card`` components — one per ``PlotSpec`` for the
    given (context, tab) combination.  Each card starts empty with a
    "▶ Render" button.

``render_one_plot(spec, primary_dir, filters)``
    Calls ``spec.func(primary_dir, plots_dir=None, filters=filters,
    return_fig=True)``, converts each returned ``(fig, fig_type)`` tuple
    into a responsive Dash component and returns the list.

Both functions are called from ``plot_cb.py``:
- ``build_gallery`` is invoked by the "Explore" button callback.
- ``render_one_plot`` is invoked by per-card "Render" button callbacks
  using Dash's ``MATCH`` pattern.
"""

import base64
import io
import logging

import dash_bootstrap_components as dbc
from dash import dcc, html

from synco.dashboard.plot_registry import PlotSpec, get_specs, NoFilterMatchError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Figure → Dash component conversion
# ---------------------------------------------------------------------------

def fig_to_component(fig, fig_type: str):
    """Convert a figure returned by a PlotSpec func into a Dash component.

    Parameters
    ----------
    fig      : A Plotly ``go.Figure`` or a Matplotlib ``Figure`` instance.
    fig_type : ``"plotly"`` or ``"matplotlib"``.

    Returns
    -------
    A ``dcc.Graph`` for Plotly figures, or an ``html.Img`` (base64 PNG)
    for Matplotlib figures.
    """
    if fig_type == "plotly":
        return dcc.Graph(
            figure=fig,
            style={"width": "100%"},
            config={"responsive": True, "displayModeBar": True},
        )

    # Matplotlib → base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    import matplotlib.pyplot as plt  # lazy import — only used here
    plt.close(fig)
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={
            "width": "100%",
            "maxWidth": "1100px",
            "display": "block",
            "margin": "auto",
        },
    )


# ---------------------------------------------------------------------------
# Gallery card builder
# ---------------------------------------------------------------------------

def _build_plot_card(spec: PlotSpec, existing_components: list) -> dbc.Card:
    """Build one gallery card for *spec*.

    Cards always start empty (no cache); the user clicks "Render" to
    trigger the MATCH callback which populates ``card-output``.
    """
    btn = dbc.Button(
        [html.I(className="bi bi-play-fill me-1"), "Render"],
        id={"type": "card-btn", "index": spec.plot_id},
        color="primary",
        size="sm",
        n_clicks=0,
    )

    header = dbc.CardHeader(
        [
            html.Div(
                [
                    html.Span(spec.label, className="fw-semibold me-2"),
                    html.Small(spec.description, className="text-muted"),
                ],
                className="flex-grow-1",
            ),
            btn,
        ],
        className="d-flex justify-content-between align-items-center py-2",
    )

    body_content = existing_components if existing_components else [
        html.Div(
            [
                html.I(className="bi bi-bar-chart me-2 text-muted"),
                html.Small("Click Render to generate this plot.", className="text-muted"),
            ],
            className="py-3 text-center",
        )
    ]

    body = dbc.CardBody(
        html.Div(body_content, id={"type": "card-output", "index": spec.plot_id}),
        className="p-2",
    )

    return dbc.Card([header, body], className="mb-3 shadow-sm")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gallery(
    context: str,
    tab: str,
    primary_dir: str,
    active_filters: dict | None = None,
) -> list:
    """Build the full plot gallery for *(context, tab)*.

    Returns a list of Dash components: an optional active-filter header row
    followed by one ``dbc.Card`` per ``PlotSpec``.  Cards always start
    empty — no disk cache is consulted.

    Parameters
    ----------
    context        : ``"cross_tissue"`` or ``"tissue"``.
    tab            : Tab ID (``"classification"``, ``"performance"``, etc.).
    primary_dir    : The input directory (results_dir or cell_fate_dir).
    active_filters : Dict of active filter values (shown as info badges).
    """
    specs = get_specs(context, tab)

    if not specs:
        return [
            dbc.Alert(
                f"No plots defined for context '{context}' / tab '{tab}'.",
                color="secondary",
                className="mt-3",
            )
        ]

    components: list = []

    # ── Active filter badges ──────────────────────────────────────────────
    filt = {k: v for k, v in (active_filters or {}).items() if v}
    if filt:
        label_map = {
            "cell_line":   ("bi-grid",             "Cell line"),
            "combination": ("bi-bezier2",          "Combination"),
            "drug":        ("bi-eyedropper",       "Drug"),
            "profile":     ("bi-bar-chart-steps",  "Profile"),
        }
        badges = []
        for key, value in filt.items():
            icon_cls, lbl = label_map.get(key, ("bi-funnel", key.replace("_", " ").title()))
            badges.append(
                dbc.Badge(
                    [html.I(className=f"bi {icon_cls} me-1"), f"{lbl}: {value}"],
                    color="info",
                    className="me-2",
                    pill=True,
                )
            )
        components.append(
            html.Div(
                [html.I(className="bi bi-funnel me-2 text-muted"), "Active filters: "] + badges,
                className="mb-3 small",
            )
        )

    # ── One card per PlotSpec (always starts empty) ───────────────────────
    for spec in specs:
        components.append(_build_plot_card(spec, []))

    return components


def render_one_plot(
    spec: PlotSpec,
    primary_dir: str,
    filters: dict | None = None,
) -> list:
    """Call ``spec.func`` with ``return_fig=True`` and return Dash components.

    This function is invoked from the MATCH callback in ``plot_cb.py``
    whenever a card's "Render" button is clicked.

    Returns a list of components suitable for setting as the card's
    ``{"type": "card-output", "index": spec.plot_id}`` children.
    No files are written to disk.
    """
    logger.info(
        "Rendering plot '%s': func=%s primary_dir=%r",
        spec.plot_id, spec.func.__name__, primary_dir,
    )

    try:
        fig_list = spec.func(
            primary_dir,
            plots_dir=None,
            filters=filters,
            return_fig=True,
        )
    except NoFilterMatchError:
        return [
            dbc.Alert(
                "No data matches the current filter settings. "
                "Adjust the filters to visualise more plots.",
                color="warning",
                className="mt-2",
            )
        ]
    except Exception as exc:
        logger.exception("Error rendering plot '%s'", spec.plot_id)
        return [
            dbc.Alert(
                [html.Strong("Render error: "), str(exc)],
                color="danger",
                className="mt-2",
            )
        ]

    components = [fig_to_component(fig, fig_type) for fig, fig_type in (fig_list or [])]

    if not components:
        return [
            dbc.Alert(
                "The plotting function completed but produced no output. "
                "Check that the results directory contains the expected data files.",
                color="info",
                className="mt-2",
            )
        ]

    return components
