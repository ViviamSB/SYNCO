"""
app.py – Dash application factory.

``create_app()`` builds and returns the configured ``dash.Dash`` instance.
All globally-accessible ``dcc.Store`` components and the Flask route for
serving plot files are set up here.

Usage
-----
    from synco.dashboard import create_app
    app = create_app()
    app.run(debug=True)
"""

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import flask
from dash import dcc, html

logger = logging.getLogger(__name__)

# Path to this file's directory (used as Dash assets folder)
_ASSETS_DIR = Path(__file__).parent / "assets"
_ASSETS_DIR.mkdir(exist_ok=True)


def create_app(debug: bool = False) -> dash.Dash:
    """
    Build and return the configured ``dash.Dash`` instance.

    Parameters
    ----------
    debug : Enable Dash debug / hot-reload mode.
    """
    server = flask.Flask(__name__)

    # ------------------------------------------------------------------
    # Flask route: serve plot files from arbitrary filesystem paths.
    # The Dash layout uses ``/serve-plot/<filepath>`` as the src for
    # iframes and images.
    # ------------------------------------------------------------------
    @server.route("/serve-plot/<path:filepath>")
    def serve_plot_file(filepath: str):
        """Serve .html / .png plot files from the user's output directory.

        On Windows, ``filepath`` looks like ``C:/path/to/file.html`` (no leading
        slash because Flask strips it).  ``Path("C:/...")`` is already absolute
        on Windows so we try it first, then fall back to prepending a slash for
        Unix paths.
        """
        # Try as-is first (handles Windows drive-letter paths like C:/...)
        fp = Path(filepath)
        if not fp.exists() or not fp.is_absolute():
            # Unix fallback: prepend leading slash
            fp = Path("/" + filepath)

        if not fp.exists() or not fp.is_file():
            flask.abort(404)

        return flask.send_from_directory(str(fp.parent), fp.name)


    # ------------------------------------------------------------------
    # Dash app
    # ------------------------------------------------------------------
    app = dash.Dash(
        __name__,
        server=server,
        use_pages=True,
        pages_folder=str(Path(__file__).parent / "pages"),
        assets_folder=str(_ASSETS_DIR),
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP,
        ],
        suppress_callback_exceptions=True,
        title="SYNCO Dashboard",
    )

    # ------------------------------------------------------------------
    # Global stores (accessible from any page)
    # ------------------------------------------------------------------
    stores = html.Div(
        [
            dcc.Store(id="store-results-dir",      storage_type="session"),
            dcc.Store(id="store-cell-fate-dir",    storage_type="session"),
            dcc.Store(id="store-config",           storage_type="session"),
            dcc.Store(
                id="store-pipeline-status",
                storage_type="memory",
                data={"status": "idle", "message": ""},
            ),
            dcc.Store(
                id="store-coverage",
                storage_type="memory",
                data={"level": "global"},
            ),
            dcc.Store(
                id="store-eval",
                storage_type="memory",
                data={"tab": "classification"},
            ),
        ],
        id="global-stores",
    )

    # ------------------------------------------------------------------
    # Navbar
    # ------------------------------------------------------------------
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [html.I(className="bi bi-diagram-3 me-2"), "SYNCO Dashboard"],
                    href="/",
                    className="fw-bold",
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink(
                            [html.I(className="bi bi-gear me-1"), "Setup"],
                            href="/",
                        )),
                        dbc.NavItem(dbc.NavLink(
                            [html.I(className="bi bi-bar-chart-line me-1"), "Explorer"],
                            href="/explorer",
                        )),
                    ],
                    navbar=True,
                    className="ms-auto",
                ),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-0",
        sticky="top",
    )

    # ------------------------------------------------------------------
    # Root layout
    # ------------------------------------------------------------------
    app.layout = html.Div(
        [
            stores,
            navbar,
            dash.page_container,
        ]
    )

    # ------------------------------------------------------------------
    # Register callbacks
    # ------------------------------------------------------------------
    from synco.dashboard.callbacks.pipeline_cb import register_pipeline_callbacks
    from synco.dashboard.callbacks.plot_cb import register_plot_callbacks

    register_pipeline_callbacks(app)
    register_plot_callbacks(app)

    logger.info("SYNCO Dashboard app created (debug=%s)", debug)
    return app
