# synco/__init__.py

"""
SYNCO package
Root package file — makes 'synco' a package and exposes the main entry points.
"""

from .main import run_pipeline, build_pipeline_config

__all__ = ["run_pipeline", "build_pipeline_config"]
