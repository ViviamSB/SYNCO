# synco/plotting/core.py

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# CORE SYNCO PLOTTING FUNCTIONS (extracted from notebook)
# - Load functions
# - Helpers to prepare data for plotting
# - Plotting functions (plotly)
#///////////////////////////////////////////////////////////////////////////////////////////////////////

import os
import re
import ast
import pathlib
import json
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import math
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# helper for saving figures
from ..utils import save_fig


#///////////////////////////////////////////////////////////////////////////////////////////////////////
# ---------------------------
# LOAD mappings and main results
# ---------------------------

def _load_main_results(results_dir: str) -> dict:
	"""Load available result CSVs and JSON dictionaries from a results directory.

	Returns a dictionary with keys for dataframes and dictionaries found in the
	folder. Missing files are skipped. Patterns (like "*_comparison_results.csv")
	are supported by picking the first match.
	"""
	results_dir = pathlib.Path(results_dir)
	if not results_dir.exists():
		raise FileNotFoundError(f"Results directory not found: {results_dir}")

	# Helper to try reading a CSV, return None if not present
	def _read_csv_if_exists(p: pathlib.Path):
		try:
			if p.exists():
				return pd.read_csv(p)
		except Exception:
			pass
		return None

	# Helper to read first file matching a glob pattern
	def _read_first_glob(pattern: str):
		matches = list(results_dir.glob(pattern))
		if not matches:
			return None
		return _read_csv_if_exists(matches[0])

	# Expected CSV files (common names used in the notebook)
	predictions_full_df = _read_csv_if_exists(results_dir / "predictions_full_df.csv")
	predictions_drugnames_df = _read_csv_if_exists(results_dir / "predictions_drug_names_synergies_df.csv")
	predictions_inhibitor_groups_df = _read_csv_if_exists(results_dir / "predictions_inhibitor_group_synergies_df.csv")

	experimental_full_df = _read_csv_if_exists(results_dir / "experimental_full_df.csv")
	experimental_drugnames_df = _read_csv_if_exists(results_dir / "experimental_drug_names_synergies_df.csv")
	experimental_inhibitor_groups_df = _read_csv_if_exists(results_dir / "experimental_inhibitor_group_synergies_df.csv")

	# Summary / metrics
	# Read all comparison result files (e.g. 'cell_line_comparison_results.csv',
	# 'inhibitor_combination_comparison_results.csv') and return a mapping
	# from the filename prefix to the DataFrame. This allows callers to select
	# the appropriate comparison table based on analysis type.
	comp_matches = list(results_dir.glob("*_comparison_results.csv"))
	comparison_results = {}
	for p in comp_matches:
		df = _read_csv_if_exists(p)
		if df is None:
			continue
		name = p.name
		if name.endswith('_comparison_results.csv'):
			key = name[:-len('_comparison_results.csv')]
		else:
			key = p.stem
		comparison_results[key] = df

	# If only one comparison file was found, expose it directly for
	# backward-compatibility; otherwise return the mapping.
	comparison_results_df = None
	if len(comparison_results) == 1:
		comparison_results_df = next(iter(comparison_results.values()))
	elif len(comparison_results) > 1:
		comparison_results_df = comparison_results

	roc_metrics_df = _read_csv_if_exists(results_dir / "roc_metrics_df.csv")

	summ_results_files = {
		"comparison": comparison_results_df,
		"roc_metrics": roc_metrics_df,
		"experimental": experimental_full_df,
		"experimental_drugnames": experimental_drugnames_df,
		"experimental_inhibitor_groups": experimental_inhibitor_groups_df,
		"predictions": predictions_full_df,
		"predictions_drugnames": predictions_drugnames_df,
		"predictions_inhibitor_groups": predictions_inhibitor_groups_df,
	}

	# JSON dictionary files to load if present
	dict_files = {
		"PD_inhibitors_dict": "PD_inhibitors_dict.json",
		"Drugnames_PD_dict": "Drugnames_PD_dict.json",
		"PD_drugnames_dict": "PD_drugnames_dict.json",
		"inhibitorgroups_dict": "inhibitorgroups_dict.json",
		"Drugnames_inhibitor_dict": "Drugnames_inhibitor_dict.json",
		"PD_targets_dict": "PD_targets_dict.json",
	}

	mechanism_dicts = {
		"PD_mechanism_dict": "PD_mechanism_dict.json",
		"Drugnames_mechanism_dict": "Drugnames_mechanism_dict.json",
		"mechanism_drugnames_dict": "mechanism_drugnames_dict.json",
		"mechanism_PD_dict": "mechanism_PD_dict.json",
	}

	def _read_json_if_exists(fname: str):
		p = results_dir / fname
		if p.exists():
			try:
				with open(p, 'r', encoding='utf8') as fh:
					return json.load(fh)
			except Exception:
				return None
		return None

	loaded_dicts = {}
	for key, fname in {**dict_files, **mechanism_dicts}.items():
		loaded = _read_json_if_exists(fname)
		loaded_dicts[key] = loaded

	results = {
		'files': summ_results_files,
		'dicts': loaded_dicts,
		'results_dir': str(results_dir)
	}

	return results



# ---------------------------
# Example usage (callable from scripts)
# ---------------------------
__all__ = [
	'_load_main_results',
]
