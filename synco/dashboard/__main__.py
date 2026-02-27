"""
synco/dashboard/__main__.py – Entry point for ``python -m synco.dashboard``.

Usage
-----
    python -m synco.dashboard               # default port 8050
    python -m synco.dashboard --port 8080
    python -m synco.dashboard --debug
    python -m synco.dashboard --host 0.0.0.0 --port 8080
"""

import argparse
import logging


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m synco.dashboard",
        description="Launch the SYNCO Dashboard Dash application.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Hostname/IP to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port to listen on (default: 8050)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug / hot-reload mode"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    from synco.dashboard.app import create_app

    app = create_app(debug=args.debug)
    print(
        f"\n  SYNCO Dashboard running on http://{args.host}:{args.port}/\n"
        "  Press Ctrl+C to stop.\n"
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
