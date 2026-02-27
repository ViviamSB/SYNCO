# synco/plotting/performance.py

"""Performance-oriented plotting helpers and lightweight loader.

This module provides a small loader that only fetches comparison/experimental
files needed for funnel and ring plots and exposes the plotting helpers.
"""

import os
import math
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from ..utils import save_fig
from .load_results import (_load_main_results,)
from .pred_distributions import _prepare_exp_pairs


def _load_results_for_performance(results_dir: str) -> dict:
	"""Load only the files needed for performance plots from a results dir.

	Returns a reduced `results` dict compatible with the helpers in this module.
	"""
	results = _load_main_results(results_dir)

	reduced = {
		'files': {
			'comparison': results.get('files', {}).get('comparison'),
			'experimental': results.get('files', {}).get('experimental'),
			'predictions': results.get('files', {}).get('predictions'),
		},
		'dicts': results.get('dicts', {}),
		'results_dir': results.get('results_dir')
	}
	return reduced


def _prepare_funnel_data(
	results: Optional[dict] = None,
	total_combinations: Optional[int] = None,
	selected_combinations: Optional[int] = None,
	synergiestic_responses: Optional[int] = None,
	experimental_results: Optional[pd.DataFrame] = None,
	threshold: float = 0.0,
) -> Dict[str, Optional[int]]:
	"""Prepare counts for the funnel plot.

	The function will attempt to compute sensible defaults from the provided
	`results` dict returned by `_load_main_results`. Values explicitly passed
	as parameters take precedence. Returned dictionary contains:
	  - total_combinations: number of unique unordered PD pairs
	  - selected_combinations: number of experimental PD pairs (from experimental file)
	  - n_cell_lines: number of cell lines present in the predictions file
	  - possible_tests: total_combinations * n_cell_lines
	  - selected_tests: selected_combinations * n_cell_lines
	  - synergiestic_responses: observed synergistic responses (from experimental_full_df or provided)
	  - total_TP: sum of True Positives in hsa_results when available

	Any missing numeric value is returned as None.
	"""
	out: Dict[str, Optional[int]] = {
		'total_combinations': None,
		'selected_combinations': None,
		'n_cell_lines': None,
		'possible_tests': None,
		'selected_tests': None,
		'synergiestic_responses': None,
		'total_TP': None,
	}

	# 1) n_cell_lines from predictions file (count of non-meta columns)
	if results and isinstance(results, dict):
		preds = results.get('files', {}).get('predictions')
		if isinstance(preds, pd.DataFrame):
			# Identify info/meta columns similarly to _prepare_predictions
			info_columns = ['Perturbation', 'PD_A', 'PD_B', 'drug_name_A', 'drug_name_B',
							'node_targets_A', 'node_targets_B', 'drug_combination',
							'inhibitor_group_A', 'inhibitor_group_B', 'inhibitor_combination',
							'targets_A', 'targets_B', 'target_combination']
			cell_line_columns = [c for c in preds.columns if c not in info_columns]
			out['n_cell_lines'] = int(len(cell_line_columns)) if cell_line_columns else None

			# total_combinations: unique unordered PD pairs present in predictions
			try:
				pairs = preds[['PD_A', 'PD_B']].astype(str).apply(lambda r: tuple(sorted([r['PD_A'], r['PD_B']])), axis=1)
				out['total_combinations'] = int(pairs.drop_duplicates().shape[0])
			except Exception:
				out['total_combinations'] = None

	# 2) selected_combinations from experimental file if not provided
	if selected_combinations is not None:
		out['selected_combinations'] = int(selected_combinations)
	else:
		if results and isinstance(results, dict):
			exp_pairs = _prepare_exp_pairs(results)
			out['selected_combinations'] = int(len(exp_pairs)) if exp_pairs else None

	# 3) allow explicit override for total_combinations
	if total_combinations is not None:
		out['total_combinations'] = int(total_combinations)

	# 4) possible / selected tests
	if out['total_combinations'] is not None and out['n_cell_lines'] is not None:
		out['possible_tests'] = int(out['total_combinations'] * out['n_cell_lines'])
	if out['selected_combinations'] is not None and out['n_cell_lines'] is not None:
		out['selected_tests'] = int(out['selected_combinations'] * out['n_cell_lines'])

	# 5) synergiestic_responses: explicit override, else count rows in the
	#    experimental (hsa) dataframe where the synergy value > threshold.
	if synergiestic_responses is not None:
		out['synergiestic_responses'] = int(synergiestic_responses)
	else:
		# Use comparison table: observed synergistic responses are the sum of
		# True Positive and False Negative counts (per user's specification).
		comp = results.get('files', {}).get('comparison') if results and isinstance(results, dict) else None
		if isinstance(comp, pd.DataFrame):
			tp_col = 'True Positive' if 'True Positive' in comp.columns else ('True Positives' if 'True Positives' in comp.columns else None)
			fn_col = 'False Negative' if 'False Negative' in comp.columns else ('False Negatives' if 'False Negatives' in comp.columns else None)
			if tp_col is not None or fn_col is not None:
				try:
					tp_vals = pd.to_numeric(comp[tp_col], errors='coerce') if tp_col is not None else pd.Series(0, index=comp.index)
					fn_vals = pd.to_numeric(comp[fn_col], errors='coerce') if fn_col is not None else pd.Series(0, index=comp.index)
					out['synergiestic_responses'] = int((tp_vals.fillna(0) + fn_vals.fillna(0)).sum())
				except Exception:
					out['synergiestic_responses'] = None
			else:
				out['synergiestic_responses'] = None
		else:
			out['synergiestic_responses'] = None

		# total_TP should be taken from the comparison results file when present
		comp = results.get('files', {}).get('comparison') if results and isinstance(results, dict) else None
		if isinstance(comp, pd.DataFrame):
			# Use the uniquely-named column when available. Prefer exact
			# 'True Positive' (singular) then 'True Positives' (plural).
			if 'True Positive' in comp.columns:
				try:
					out['total_TP'] = int(pd.to_numeric(comp['True Positive'], errors='coerce').sum())
				except Exception:
					out['total_TP'] = None
			elif 'True Positives' in comp.columns:
				try:
					out['total_TP'] = int(pd.to_numeric(comp['True Positives'], errors='coerce').sum())
				except Exception:
					out['total_TP'] = None

	return out


def _prepare_ring_data(results: dict) -> pd.DataFrame:
	"""Extract the comparison DataFrame from `results` and return a simple
	normalized DataFrame for ring plotting.

	This helper expects the `results` dict returned by `_load_main_results` and
	a comparison DataFrame available under `results['files']['comparison']`.
	The comparison DataFrame is expected to have the standard headers used by
	the pipeline (e.g. 'True Positive', 'True Negative', 'False Positive',
	'False Negative', 'Match', 'Mismatch', 'Recall'). If plural variants are
	present the function will accept them but prefers the singular names.

	Returns a DataFrame with columns: ['label','Match','Mismatch','TP','TN','FP','FN','Recall']
	"""
	if not isinstance(results, dict):
		raise ValueError('`results` must be the dict returned by _load_main_results')

	comp = results.get('files', {}).get('comparison')
	if comp is None:
		raise ValueError('comparison DataFrame not found in results["files"]["comparison"]')
	if isinstance(comp, dict):
		found = None
		for v in comp.values():
			if isinstance(v, pd.DataFrame):
				found = v
				break
		if found is None:
			raise ValueError('comparison DataFrame not found in results["files"]["comparison"]')
		comp = found
	elif not isinstance(comp, pd.DataFrame):
		raise ValueError('comparison DataFrame not found in results["files"]["comparison"]')

	comp = comp.copy()

	required = ['Match', 'Mismatch', 'True Positive', 'True Negative', 'False Positive', 'False Negative']
	missing = [c for c in required if c not in comp.columns]
	if missing:
		raise ValueError(f'comparison DataFrame missing required columns: {missing}')

	# build normalized frame
	df_norm = pd.DataFrame()
	# Simplified label selection: use the first column as labels when present.
	# This handles CSVs where the first column contains row names (no index_col).
	# If no columns are present, fall back to the DataFrame index.
	first_col = comp.columns[0] if len(comp.columns) > 0 else None
	if first_col is not None:
		labels = comp.iloc[:, 0].astype(str).tolist()
	else:
		labels = [str(i) for i in comp.index]

	df_norm['label'] = labels
	df_norm['Match'] = pd.to_numeric(comp['Match'], errors='coerce').fillna(0).astype(int)
	df_norm['Mismatch'] = pd.to_numeric(comp['Mismatch'], errors='coerce').fillna(0).astype(int)
	df_norm['TP'] = pd.to_numeric(comp['True Positive'], errors='coerce').fillna(0).astype(int)
	df_norm['TN'] = pd.to_numeric(comp['True Negative'], errors='coerce').fillna(0).astype(int)
	df_norm['FP'] = pd.to_numeric(comp['False Positive'], errors='coerce').fillna(0).astype(int)
	df_norm['FN'] = pd.to_numeric(comp['False Negative'], errors='coerce').fillna(0).astype(int)

	if 'Recall' in comp.columns:
		df_norm['Recall'] = pd.to_numeric(comp['Recall'], errors='coerce')
	else:
		df_norm['Recall'] = np.nan

	# Preserve original metric columns when present so callers can access
	# the raw values (and so sorting/annotation that expects these names
	# continues to work). Coerce numeric-like columns to floats when
	# appropriate, otherwise leave as-is.
	_preserve_cols = ['True Positive', 'True Negative', 'False Positive', 'False Negative',
	                  'Total', 'Match %', 'Mismatch %', 'Accuracy', 'Recall', 'Precision']
	for c in _preserve_cols:
		if c in comp.columns:
			# try numeric coercion for readability; fall back to original dtype
			try:
				df_norm[c] = pd.to_numeric(comp[c], errors='coerce')
			except Exception:
				df_norm[c] = comp[c]
		else:
			# keep column absent if not present in source
			pass

	return df_norm


def _legend_elements() -> List[Line2D]:
	"""Return a consistent list of legend `Line2D` elements used across
	ring plots. Centralising this ensures identical legend appearance.
	"""
	return [
		Line2D([0], [0], marker='o', color='w', label='Correct prediction', markerfacecolor='royalblue', markersize=12),
		Line2D([0], [0], marker='o', color='w', label='Missed prediction', markerfacecolor='#D94602', markersize=12),
		Line2D([0], [0], marker='o', color='w', label='True Positives', markerfacecolor='#458cff', markersize=12),
		Line2D([0], [0], marker='o', color='w', label='True Negatives', markerfacecolor='#6db0ff', markersize=12),
		Line2D([0], [0], marker='o', color='w', label='False Positives', markerfacecolor='#FA7F2E', markersize=12),
		Line2D([0], [0], marker='o', color='w', label='False Negatives', markerfacecolor='#FDAA65', markersize=12),
		Line2D([0], [0], marker='o', color='w', label='% Recall', markerfacecolor='w', markersize=15),
	]


def _draw_ring(ax, row, title: str = '',
			   outer_radius: float = 1.0, inner_radius: float = 0.7,
			   outer_width: float = 0.3, inner_width: float = 0.3,
			   outer_colors: Optional[List[str]] = None,
			   inner_colors: Optional[List[str]] = None,
			   center_metric: str = 'recall',
			   center_fontsize: int = 18,
			   show_legend: bool = False,
			   title_fontsize: int = 20,
			   title_pad: int = 5):
	"""Draw a two-ring (outer=Match/Mismatch, inner=TP/TN/FP/FN) on `ax`.

	`row` may be a pandas Series from either the normalized DataFrame
	(columns: 'Match','Mismatch','TP','TN','FP','FN','Recall') or the
	raw comparison DataFrame (with e.g. 'True Positive' columns). This
	helper will coerce both forms.
	"""
	if outer_colors is None:
		outer_colors = ['royalblue', '#D94602']
	if inner_colors is None:
		inner_colors = ['#458cff', '#6db0ff', '#FA7F2E', '#FDAA65']

	def _get_value(keys):
		for k in keys:
			if k in row.index:
				try:
					return int(pd.to_numeric(row[k], errors='coerce'))
				except Exception:
					try:
						return int(row[k])
					except Exception:
						return 0
		return 0

	Match = _get_value(['Match', 'Matches'])
	Mismatch = _get_value(['Mismatch', 'Mismatches'])
	TP = _get_value(['TP', 'True Positive', 'True Positives'])
	TN = _get_value(['TN', 'True Negative', 'True Negatives'])
	FP = _get_value(['FP', 'False Positive', 'False Positives'])
	FN = _get_value(['FN', 'False Negative', 'False Negatives'])

	outer_sizes = [Match, Mismatch]
	inner_values = [TP, TN, FP, FN]

	ax.set_aspect('equal')
	# outer ring
	ax.pie(outer_sizes, radius=outer_radius, colors=outer_colors, startangle=90,
		   wedgeprops=dict(width=outer_width, edgecolor='white'))

	# inner ring with actual values displayed for non-zero slices
	disp_labels = [str(val) if val > 0 else '' for val in inner_values]
	ax.pie(inner_values, radius=inner_radius, colors=inner_colors, startangle=90,
		   labels=disp_labels, labeldistance=0.7, textprops={'color': 'w', 'fontsize': 12},
		   wedgeprops=dict(width=inner_width, edgecolor='white'))

	# Compute metrics we may display in the center
	try:
		recall_val = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0
	except Exception:
		recall_val = 0.0
	try:
		accuracy_val = (Match / (Match + Mismatch) * 100) if (Match + Mismatch) > 0 else 0.0
	except Exception:
		accuracy_val = 0.0
	try:
		precision_val = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0.0
	except Exception:
		precision_val = 0.0

	# Allow explicit values from row (prefer provided numeric columns)
	if 'Recall' in row.index and not pd.isna(row.get('Recall')):
		try:
			recall_val = float(row.get('Recall'))
		except Exception:
			pass
	if 'Accuracy' in row.index and not pd.isna(row.get('Accuracy')):
		try:
			accuracy_val = float(row.get('Accuracy'))
		except Exception:
			pass
	if 'Precision' in row.index and not pd.isna(row.get('Precision')):
		try:
			precision_val = float(row.get('Precision'))
		except Exception:
			pass

	center_metric = (center_metric or 'recall').lower()
	if center_metric == 'accuracy':
		metric_val = accuracy_val
		metric_label = 'Accuracy'
	elif center_metric == 'precision':
		metric_val = precision_val
		metric_label = 'Precision'
	else:
		metric_val = recall_val
		metric_label = 'Recall'

	try:
		ax.text(0, 0, f"{metric_val:.1f}%", ha='center', va='center', fontsize=center_fontsize)
	except Exception:
		pass

	# Title handling: prefer explicit title argument; if missing, try to use
	# the Series name (row.name) so cell-line / combination names appear.
	final_title = title
	if (not final_title or str(final_title).strip() == '') and hasattr(row, 'name') and row.name is not None:
		try:
			if not pd.isna(row.name) and str(row.name).strip() != '':
				final_title = str(row.name)
		except Exception:
			pass
	if final_title:
		try:
			ax.set_title(final_title, fontsize=title_fontsize, pad=title_pad)
		except Exception:
			pass

	# Optional legend: when requested draw a single figure-level legend using
	# consistent legend elements. For grid plots, callers should request a
	# single legend once (not per-subplot).
	if show_legend:
		try:
			ax.figure.legend(handles=_legend_elements(), fontsize=10, loc='upper right', bbox_to_anchor=(1.28, 0.9))
		except Exception:
			pass


def plot_funnel(
	total_combinations: Optional[int] = None,
	selected_combinations: Optional[int] = None,
	synergiestic_responses: Optional[int] = None,
	experimental_results: Optional[pd.DataFrame] = None,
	results: Optional[dict] = None,
	threshold: float = 0.0,
	plot_dir: Optional[str] = None,
	plot_name: Optional[str] = None,
	width: Optional[int] = None,
	height: Optional[int] = None,
	show: bool = True,
) -> go.Figure:
	"""Create a funnel plot summarising possible/selected/observed counts.

	All counts may be provided explicitly or inferred from `results` (the
	dict returned by `_load_main_results`) and/or `experimental_results`.
	The helper `_prepare_funnel_data` centralises the logic for computing
	sensible defaults.

	Args:
		threshold: float cutoff applied to the experimental `synergy` column
			to count synergistic responses (default 0.0).
	"""
	prep = _prepare_funnel_data(results=results,
								total_combinations=total_combinations,
								selected_combinations=selected_combinations,
								synergiestic_responses=synergiestic_responses,
								experimental_results=experimental_results,
								threshold=threshold)

	tc = prep.get('total_combinations') or 0
	sc = prep.get('selected_combinations') or 0
	n_cell_lines = prep.get('n_cell_lines') or 0
	possible_tests = prep.get('possible_tests') or (tc * n_cell_lines)
	selected_tests = prep.get('selected_tests') or (sc * n_cell_lines)
	syner = prep.get('synergiestic_responses') or 0

	x = [possible_tests, selected_tests, syner]
	y = ['Total Possible Experiments', 'Predicted Priority Experiments', 'Observed Synergistic Responses']

	fig = px.funnel(x=x, y=y, color_discrete_sequence=['#11a2a5', '#EEA9FC', '#FF6F61'])
	fig.update_traces(textinfo='value+percent initial', textfont_size=14, textposition='inside')
	fig.update_layout(
		title='Modelling Framework Efficiency',
		yaxis_title='',
		xaxis_title='Number of combinations',
		showlegend=False,
		font=dict(size=18),
		width=width or 1200,
		height=height or 500)

	if show:
		try:
			fig.show()
		except ValueError as e:
			msg = str(e)
			if 'nbformat' in msg or 'Mime type rendering' in msg:
				try:
					save_fig(fig, plot_dir or '.', plot_name or 'funnel_plot', formats=['png', 'html'], scale=2, fig_type='plotly')
					print(f"Plotly renderer fallback: saved funnel plot to {plot_dir or '.'}")
				except Exception:
					logging.getLogger(__name__).exception('Fallback save for funnel plot failed')
			else:
				raise

	if plot_dir:
		save_fig(fig, plot_dir, plot_name or 'funnel_plot', formats=['png', 'html'], scale=2, fig_type='plotly')

	return fig


def plot_ring_summary(results_or_df,
					  plots_dir: Optional[str] = None,
					  plot_name: Optional[str] = 'rings_summary',
					  show: bool = False,
					  center_metric: str = 'recall',
					  center_fontsize: int = 18) -> plt.Figure:
	"""Create a concentric ring (donut) summary plot from comparison results.

	`results_or_df` can be either the `results` dict returned by
	`_load_main_results` or a pandas DataFrame with comparison summary rows
	(columns expected: Match, Mismatch, True Positives, True Negatives,
	False Negatives, False Positives).
	"""
	# Normalize input to standard ring DataFrame
	comp_norm = _prepare_ring_data(results_or_df)

	# Sum across rows (e.g., across cell lines / combinations)
	Match = int(comp_norm['Match'].sum())
	Mismatch = int(comp_norm['Mismatch'].sum())
	TP = int(comp_norm['TP'].sum())
	TN = int(comp_norm['TN'].sum())
	FN = int(comp_norm['FN'].sum())
	FP = int(comp_norm['FP'].sum())

	Total = Match + Mismatch
	Accuracy = (Match / Total * 100) if Total > 0 else 0.0
	Recall = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0
	Precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0.0
	TPR = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0
	TNR = (TN / (TN + FP) * 100) if (TN + FP) > 0 else 0.0
	Balanced_Accuracy = (TPR + TNR) / 2

	# Build figure with two concentric donuts using centralized helper
	fig, ax = plt.subplots(figsize=(7, 6))
	row = pd.Series({
		'Match': Match,
		'Mismatch': Mismatch,
		'TP': TP,
		'TN': TN,
		'FP': FP,
		'FN': FN,
		'Recall': Recall,
		'Accuracy': Accuracy,
		'Precision': Precision,
	})

	outer_colors = ['royalblue', '#D94602']
	inner_colors = ['#458cff', '#6db0ff', '#FA7F2E', '#FDAA65']

	# draw using centralized helper (request a figure legend here)
	_draw_ring(ax, row, title='', outer_colors=outer_colors, inner_colors=inner_colors,
			   center_metric=center_metric, center_fontsize=center_fontsize,
			   show_legend=True)

	# Accuracy / Precision box text
	fig.text(0.5, 0.06, f"Accuracy: {Accuracy:.1f}%", fontsize=12, ha='center', va='center')
	fig.text(0.5, 0.02, f"Precision: {Precision:.1f}%", fontsize=12, ha='center', va='center')
	fig.text(0.5, -0.02, f"Balanced accuracy: {Balanced_Accuracy:.1f}%", fontsize=12, ha='center', va='center')
	fig.text(0.5, -0.06, f"Total cell lines: {len(comp_norm)}", 
			 fontsize=10, ha='center', va='center', style='italic', color='#555555')
	fig.text(0.5, -0.10, f"Total comparisons: {Total:,}", 
			 fontsize=10, ha='center', va='center', style='italic', color='#555555')

	plt.title(f'Modelling predicted performance', fontsize=14)
	plt.tight_layout()

	# Save
	if plots_dir:
		# Use centralized exporter which understands matplotlib figures
		os.makedirs(plots_dir, exist_ok=True)
		try:
			save_fig(fig, plots_dir, plot_name or 'rings_summary', formats=['png', 'html'], scale=2, fig_type='matplotlib')
		except Exception:
			logging.getLogger(__name__).exception('Failed to save ring summary via save_fig')

	# When running in notebook/headless environments the matplotlib backend
	if show:
		try:
			from IPython.display import display, Image as IPythonImage
		except Exception:
			IPythonImage = None

		png_path = None
		if plots_dir:
			candidate = os.path.join(plots_dir, (plot_name or 'rings_summary') + '.png')
			if os.path.exists(candidate):
				png_path = candidate

		if IPythonImage and png_path:
			try:
				display(IPythonImage(filename=png_path))
			except Exception:
				# fallback to standard show only when interactive backend
				try:
					if not mpl.get_backend().lower().startswith('agg'):
						plt.show()
				except Exception:
					pass
		else:
			try:
				if not mpl.get_backend().lower().startswith('agg'):
					plt.show()
			except Exception:
				pass

	return fig


def plot_combination_rings(combi_df,
						   plots_dir: Optional[str] = None,
						   plot_name: Optional[str] = 'rings_combination',
						   ncols: int = 4,
						   figsize: Tuple[int, int] = (16, 20),
						   show: bool = False,
						   center_metric: str = 'recall',
						   center_fontsize: int = 14,
						   show_legend: bool = False) -> plt.Figure:
	"""Plot grid of ring plots for drug combinations.

	Parameters
	- `combi_df`: DataFrame with per-combination comparison metrics. Expected
	  columns: 'Match', 'Mismatch', 'True Positive'|'True Positives',
	  'True Negative'|'True Negatives', 'False Positive'|'False Positives',
	  'False Negative'|'False Negatives', and optionally 'Recall'. The index
	  or a column will be used as the title for each subplot.
	- `plots_dir`: where to save outputs (if provided).
	- `plot_name`: base name for saved files.
	- `ncols`: number of columns in the grid.
	- `figsize`: matplotlib figure size as (width, height).
	- `show`: whether to call `plt.show()`.

	Returns the matplotlib `Figure`.
	"""
	# Expect a pre-normalized DataFrame with required columns. This function
	# no longer attempts to coerce raw comparison results; callers should
	# call `_prepare_ring_data` prior to plotting (e.g., `make_ring_plots`).
	if not isinstance(combi_df, pd.DataFrame):
		raise ValueError('`combi_df` must be a pandas DataFrame (pre-normalized)')

	needed = {'label', 'Match', 'Mismatch', 'TP', 'TN', 'FP', 'FN'}
	if not needed.issubset(set(combi_df.columns)):
		raise ValueError(f'combi_df must contain columns: {sorted(list(needed))}')

	df_norm = combi_df.copy()

	if df_norm is None or df_norm.empty:
		raise ValueError('No valid comparison rows provided for ring plotting')

	n_items = len(df_norm)
	nrows = int(math.ceil(n_items / float(ncols))) if n_items else 0
	fig, axes = plt.subplots(nrows or 1, ncols, figsize=figsize)
	axes = np.array(axes).reshape(-1)

	outer_colors = ["royalblue", "#D94602"]
	inner_colors = ["#458cff", "#6db0ff", "#FA7F2E", "#FDAA65"]

	def _draw(ax, row, title):
		ax.set_aspect('equal')
		# delegate drawing to centralized helper
		_draw_ring(ax, row, title=title, outer_colors=outer_colors, inner_colors=inner_colors,
				   center_metric=center_metric, center_fontsize=center_fontsize, title_fontsize=14,)

	# iterate rows and draw
	for ax, (_, row) in zip(axes, df_norm.iterrows()):
		# Prefer explicit 'label' column; fall back to the row name and ensure
		# it's a non-empty string so cell-line titles are shown.
		title_val = row.get('label', row.name) if hasattr(row, 'get') else row.name
		if pd.isna(title_val) or str(title_val).strip() == '':
			title_val = row.name
		title = str(title_val)
		_draw(ax, row, title)

	# hide any unused axes
	for ax in axes[len(df_norm):]:
		ax.axis('off')

	# legend (optional for grid plots)
	if show_legend:
		legend_elements = _legend_elements()
		fig.legend(handles=legend_elements, fontsize=10, loc='upper right', bbox_to_anchor=(1.03, 0.95))

	plt.tight_layout()

	# save
	if plots_dir:
		save_fig(fig, plots_dir, plot_name, fig_type='matplotlib')

	if show:
		try:
			if not mpl.get_backend().lower().startswith('agg'):
				plt.show()
		except Exception:
			pass

	return fig


def plot_cell_line_rings(cell_df: pd.DataFrame,
				plots_dir: Optional[str] = None,
				plot_name: Optional[str] = 'rings_cell_lines',
				ncols: int = 7,
				figsize: Tuple[int, int] = (16, 16),
				show: bool = False,
				center_metric: str = 'recall',
				center_fontsize: int = 14,
				show_legend: bool = False) -> plt.Figure:
	"""Plot grid of ring plots for cell lines using a comparison DataFrame.

	This mirrors the notebook's cell-line ring plotting layout. See
	`plot_combination_rings` for column requirements and behavior.
	"""
	# Reuse the combination plotting function but with different defaults
	return plot_combination_rings(cell_df, plots_dir=plots_dir, plot_name=plot_name,
			ncols=ncols, figsize=figsize, show=show,
			center_metric=center_metric, center_fontsize=center_fontsize,
			show_legend=show_legend)


def make_performance_plots(
	results_dir: str,
	plots_dir: Optional[str] = None,
	performance: str = 'ring', # or 'funnel' or 'both'
	funnel_plot_name: str = 'global_efficiency',
	rings_plot_name: str = 'global_performance',
	show: bool = False,
	funnel_size: Optional[tuple] = None,
	center_metric: str = 'recall',
	center_fontsize: int = 18,
) -> None:
	"""Load minimal performance results and render funnel + ring summary plots.

	This wrapper uses the reduced loader to prepare data and calls the
	plotting functions in this module. Files are saved under `plots_dir`.
	"""
	results = _load_results_for_performance(results_dir)

	if plots_dir is None:
		plots_dir = os.path.join(results.get('results_dir', results_dir), 'plots')
	os.makedirs(plots_dir, exist_ok=True)

	if performance.lower() not in {'funnel', 'ring', 'both'}:
		raise ValueError(f'Unknown performance plot type: {performance}')
	if performance.lower() in {'funnel'}:
	# Funnel
		fw, fh = (funnel_size or (1200, 500))
		try:
			plot_funnel(results=results, plot_dir=plots_dir, plot_name=funnel_plot_name, width=fw, height=fh, show=show)
		except Exception:
			logging.getLogger(__name__).exception('Failed to render funnel plot')

	if performance.lower() in {'ring'}:
	# Ring summary (aggregate)
		try:
			plot_ring_summary(results, plots_dir=plots_dir, plot_name=rings_plot_name, show=show,
						center_metric=center_metric, center_fontsize=center_fontsize)
		except Exception:
			logging.getLogger(__name__).exception('Failed to render ring summary')

def make_ring_plots(
	results_dir: str,
	analysis_type: str = 'cell_line', # or 'inhibitor_combination'
	plots_dir: Optional[str] = None,
	sort_by: Optional[str] = 'Accuracy', # or 'recall', 'precision', or None
	show: bool = False,
	size: Tuple[int, int] = (16, 16),
	ncols: int = 7,
	center_metric: str = 'recall',
	center_fontsize: int = 14,
) -> None:
	"""Load minimal performance results and render ring plots for cell lines or combinations.

	This wrapper uses the reduced loader to prepare data and calls the
	plotting functions in this module. Files are saved under `plots_dir`.
	"""
	results = _load_results_for_performance(results_dir)

	if plots_dir is None:
		plots_dir = os.path.join(results.get('results_dir', results_dir), 'plots')
	os.makedirs(plots_dir, exist_ok=True)

	# Prepare normalized comparison DataFrame
	try:
		# The loader may return multiple comparison files as a dict keyed by
		# filename prefix (e.g. {'cell_line': df, 'inhibitor_combination': df}).
		# Select the appropriate table based on `analysis_type`.
		comp_src = results.get('files', {}).get('comparison')
		if isinstance(comp_src, dict):
			# map common analysis_type names to expected file prefixes
			key_map = {
				'cell_line': 'cell_line',
				'inhibitor_combination': 'inhibitor_combination',
				'combination': 'inhibitor_combination'
			}
			desired = key_map.get(analysis_type, analysis_type)
			selected = None
			# direct hit
			if desired in comp_src:
				selected = comp_src[desired]
			else:
				# try looser matching (substring or exact match)
				for k, v in comp_src.items():
					if k == analysis_type or analysis_type in k or desired in k:
						selected = v
						break
			# fallback to the first available file
			if selected is None:
				selected = next(iter(comp_src.values()))

			temp_results = {'files': {'comparison': selected}, 'results_dir': results.get('results_dir')}
		else:
			temp_results = results

		comp_norm = _prepare_ring_data(temp_results)
		# Leave computation of Accuracy/Precision to the preparation step so
		# sorting by those columns works when they are present in the source.
		comp_norm = comp_norm.sort_values(by=sort_by, ascending=False) if sort_by in comp_norm.columns else comp_norm
	except Exception:
		logging.getLogger(__name__).exception('Failed to prepare normalized comparison data for ring plots')
		return

	# Ring plots
	try:
		if analysis_type == 'cell_line':
			rings_plot_name = 'cell_line_rings'
			plot_cell_line_rings(comp_norm, plots_dir=plots_dir, plot_name=rings_plot_name,
								 ncols=ncols, figsize=size, show=show,
								 center_metric=center_metric, center_fontsize=center_fontsize)
		elif analysis_type == 'inhibitor_combination':
			rings_plot_name = 'combination_rings'
			plot_combination_rings(comp_norm, plots_dir=plots_dir, plot_name=rings_plot_name,
								   ncols=ncols, figsize=size, show=show,
								   center_metric=center_metric, center_fontsize=center_fontsize)
		else:
			raise ValueError(f'Unknown analysis_type: {analysis_type}')
	except Exception:
		logging.getLogger(__name__).exception('Failed to render ring plots')