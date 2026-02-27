"""
synco.dashboard
---------------
Dash-based interactive application for running the SYNCO pipeline
and exploring its visualisation outputs.

Usage
-----
    python -m synco.dashboard          # starts the development server on port 8050
    python -m synco.dashboard --port 8080 --debug
"""

from .app import create_app

__all__ = ["create_app"]
