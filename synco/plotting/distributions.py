# Wrapper module for distribution-related plotting functions

import os
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import save_fig
from .load_results import (_load_main_results,)


#///////////////////////////////////////////////////////////////////////////////////////////////////////
# ---------------------------
# STYLE settings
# ---------------------------

def _style_moa_colors(results: Optional[dict] = None) -> Dict[str, str]:
	"""Return a mapping mechanism -> color.

	If `results` dict (from `_load_main_results`) is provided, extract mechanism
	names from `mechanism_drugnames_dict` keys or `PD_mechanism_dict` values.
	Otherwise create numeric mechanism names ('Mechanism 1', ...) up to 9 items.

	Returns up to 10 colors; the last color is reserved for 'Unlabeled'.
	"""

	palette = [
		"#636EFA",  # blue
		"#FC7299",  # pink
		"#71C715",  # green
		"#FF97FF",  # light magenta
		"#16B7D3",  # cyan
		"#BD7EF7",  # purple
		"#F09138",  # orange
		"#FF6F61",  # coral
		"#0C40A0",  # dark blue
		"#FFDE4D",  # amber
		"#757575",  # gray (Unlabeled)
	]

	mechs: List[str] = []
	if results and isinstance(results, dict):
		md = results.get('dicts', {}).get('mechanism_drugnames_dict')
		# support both 'mechanism_PD_dict' and legacy/alternate 'PD_mechanism_dict'
		mp = results.get('dicts', {}).get('mechanism_PD_dict') or results.get('dicts', {}).get('PD_mechanism_dict')
		if md and isinstance(md, dict):
			mechs = list(md.keys())
		elif mp and isinstance(mp, dict):
			# unique mechanism names from mapping PD -> mechanism
			mechs = sorted({v for v in mp.values() if v})

	# If no mechanisms found in results, define numeric mechanism names
	if not mechs:
		n = len(palette) - 1  # reserve one for Unlabeled
		mechs = [f"Mechanism {i+1}" for i in range(n)]

	mapping: Dict[str, str] = {}
	for i, mech in enumerate(mechs):
		if i < len(palette) - 1:
			mapping[mech] = palette[i]
		else:
			mapping[mech] = palette[-1]

	mapping['Unlabeled'] = palette[-1]
	return mapping

	
		
#///////////////////////////////////////////////////////////////////////////////////////////////////////
# ---------------------------
# PROCESS main data
# ---------------------------

def _prepare_predictions(results: dict) -> pd.DataFrame:
	"""Process predictions from the `results` dict returned by `_load_main_results`.

	This function requires the `results` dict (it will not accept a path or DataFrame).
	It expects `results['files']['predictions']` to be a pandas DataFrame and will use
	`results['dicts']['mechanism_PD_dict']` (if present) to create `moa_group_A/B` from
	PD IDs. If `mechanism_PD_dict` is absent, `moa_group_A/B` will not be created.
	"""
	if not isinstance(results, dict) or 'files' not in results:
		raise ValueError('`results` must be the dict returned by _load_main_results')

	predictions_full = results['files'].get('predictions')

	# Prefer a PD -> mechanism mapping when available. The loader places
	# mappings in `results['dicts']` using either key; use whichever is present.
	dicts_bundle = results.get('dicts') or {}
	# Prefer the canonical PD -> mechanism mapping key `PD_mechanism_dict`.
	mechanism_PD = dicts_bundle.get('PD_mechanism_dict') or dicts_bundle.get('mechanism_PD_dict')

	if predictions_full is None:
		raise ValueError('Predictions DataFrame not found in results["files"]["predictions"]')

	if not isinstance(predictions_full, pd.DataFrame):
		raise ValueError('predictions entry in results["files"] must be a pandas DataFrame')

	info_columns = ['Perturbation', 'PD_A', 'PD_B', 'drug_name_A', 'drug_name_B',
					'node_targets_A', 'node_targets_B', 'drug_combination',
					'inhibitor_group_A', 'inhibitor_group_B', 'inhibitor_combination',
					'targets_A', 'targets_B', 'target_combination'
					]

						# Placeholder for pairs color mapping. Currently unused.
	# Heuristic to detect cell-line columns: exclude info columns and any
	# obviously index-like names, then accept columns where at least half
	# of the values are numeric.
	candidates = [c for c in predictions_full.columns if c not in info_columns]
	cell_line_columns: List[str] = []
	for c in candidates:
		cn = str(c).lower()
		if cn == 'index' or cn.startswith('unnamed'):
			continue
		num_non_na = pd.to_numeric(predictions_full[c], errors='coerce').notna().sum()
		frac = float(num_non_na) / max(1, len(predictions_full))
		if frac >= 0.5:
			cell_line_columns.append(c)
	if not cell_line_columns:
		# Last-resort: treat any non-index/non-unnamed columns as cell-lines
		cell_line_columns = [c for c in candidates if not str(c).lower().startswith('unnamed') and str(c).lower() != 'index']
	if not cell_line_columns:
		raise ValueError('No cell line columns detected in predictions dataframe')

	predictions_melted = predictions_full.melt(
		id_vars=info_columns,
		value_vars=cell_line_columns,
		var_name='cell_line',
		value_name='synergy'
	)

	# Only create moa_group columns when mechanism_PD_dict is provided
	if mechanism_PD:
		mechanism_PD_str = {str(k): v for k, v in mechanism_PD.items()}
		predictions_melted['moa_group_A'] = predictions_melted['PD_A'].astype(str).map(mechanism_PD_str)
		predictions_melted['moa_group_B'] = predictions_melted['PD_B'].astype(str).map(mechanism_PD_str)

	# Ensure synergy values are numeric, then flip the sign
	predictions_melted['synergy'] = pd.to_numeric(predictions_melted['synergy'], errors='coerce')
	predictions_melted['synergy'] = -predictions_melted['synergy']

	return predictions_melted


def _prepare_exp_pairs(results: dict) -> List[str]:
	"""Extract experimental PD combinations from `results['files']['experimental']`.

	Returns a sorted list of unique PD pairs formatted as "A | B" where A <= B
	(string-sorted) to ensure a consistent representation.
	If the experimental dataframe is missing or invalid, returns an empty list.
	"""
	if not isinstance(results, dict) or 'files' not in results:
		raise ValueError('`results` must be the dict returned by _load_main_results')

	experimental_df = results['files'].get('experimental')
	if experimental_df is None or not isinstance(experimental_df, pd.DataFrame):
		return []

	pairs = set()
	for _, row in experimental_df.iterrows():
		pd_a = row.get('PD_A') if hasattr(row, 'get') else row['PD_A']
		pd_b = row.get('PD_B') if hasattr(row, 'get') else row['PD_B']
		if pd.isna(pd_a) or pd.isna(pd_b):
			continue
		a = str(pd_a).strip()
		b = str(pd_b).strip()
		pair = sorted([a, b])
		pairs.add(f"{pair[0]} | {pair[1]}")

	# Return deterministic sorted list
	return sorted(pairs)


def _prepare_pairs_table(predictions_melted: pd.DataFrame, PD_combi: List[str], supplementary_files_path: Optional[str] = None) -> pd.DataFrame:
	"""Aggregate pair-level stats from `predictions_melted` and return `pair_table`.

	`PD_combi` is expected to be provided by the caller (list of "A | B" strings).
	"""
	pair_table = (predictions_melted
				  .groupby(['PD_A', 'PD_B'], as_index=False)
				  .agg(
					  median=('synergy', 'median'),
					  mean=('synergy', 'mean'),
					  std=('synergy', 'std'),
					  iqr=('synergy', lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
					  n_cell_lines=('synergy', 'size'),
					  drug_name_A=('drug_name_A', 'first'),
					  drug_name_B=('drug_name_B', 'first'),
					  moa_A=('moa_group_A', 'first'),
					  moa_B=('moa_group_B', 'first')
				  )
				  )

	pair_table = pair_table.sort_values(by=['median', 'iqr'], ascending=[False, True]).reset_index(drop=True)
	pair_table['PD_pair'] = pair_table['PD_A'].astype(str) + " | " + pair_table['PD_B'].astype(str)
	pair_table['selected'] = pair_table['PD_pair'].isin(PD_combi)

	if supplementary_files_path:
		os.makedirs(supplementary_files_path, exist_ok=True)
		pair_table.to_csv(os.path.join(supplementary_files_path, 'predictions_stats_all_pairs.csv'), index=False)

	return pair_table


def _prepare_mechanism_summary(pair_table: pd.DataFrame, results: Optional[dict] = None, supplementary_files_path: Optional[str] = None) -> Optional[pd.DataFrame]:
	"""Create a mechanism-level summary from `pair_table` when mechanism data exists.

	This function checks the provided `results` dict for mechanism mappings and only
	computes the summary if mechanism information is available. Returns the
	mechanism_summary DataFrame or `None` when mechanism mappings are not present.
	"""
	if not isinstance(results, dict):
		return None

	mech_map = results.get('dicts', {}) if results else {}
	# Check for PD -> mechanism mapping under the canonical key first
	has_mechanism = bool(mech_map.get('PD_mechanism_dict') or mech_map.get('mechanism_drugnames_dict') or mech_map.get('mechanism_PD_dict'))
	if not has_mechanism:
		return None

	# Ensure moa columns exist in the pair_table
	if 'moa_A' not in pair_table.columns or 'moa_B' not in pair_table.columns:
		return None

	mechanism_summary = (pair_table
						 .groupby(['moa_A', 'moa_B'], as_index=False)
						 .agg(
							 n_predicted_pairs=('PD_pair', 'size'),
							 median=('median', 'median'),
							 iqr=('iqr', 'mean'),
							 mean=('mean', 'mean'),
							 std=('std', 'mean'),
							 n_selected_pairs=('selected', 'sum'),
						 )
						 )
	mechanism_summary[['median', 'iqr', 'mean', 'std']] = mechanism_summary[['median', 'iqr', 'mean', 'std']].round(4)

	if supplementary_files_path:
		os.makedirs(supplementary_files_path, exist_ok=True)
		mechanism_summary.to_csv(os.path.join(supplementary_files_path, 'predictions_stats_by_mechanism.csv'), index=False)

	return mechanism_summary


def _prepare_family_pairs(pair_table: pd.DataFrame, PD_combi: Optional[List[str]] = None, family_priority: Optional[List[str]] = None) -> pd.DataFrame:
	"""Assign family labels to pairs and compute consistency for plotting.

	- family: categorical label used for plotting colors. Prioritises
			entries from `family_priority` in order, defaulting to
			['PI3K/AKT/MTOR', 'DNA Damage'] then 'Other'.
	- consistency: negative IQR (so higher is more consistent).

	`PD_combi` can be provided to mark a `selected` column when not present.
	Returns a DataFrame with the added columns `family` and `consistency`.
	"""
	if not isinstance(pair_table, pd.DataFrame):
		raise ValueError('pair_table must be a pandas DataFrame')

	df = pair_table.copy()

	# Ensure moa columns exist
	if 'moa_A' not in df.columns:
		df['moa_A'] = None
	if 'moa_B' not in df.columns:
		df['moa_B'] = None

	# Ensure selected column exists if PD_combi provided
	if PD_combi is not None and 'selected' not in df.columns:
		df['selected'] = df.apply(lambda r: f"{r['PD_A']} | {r['PD_B']}" in PD_combi, axis=1)

	# Use provided priority list or sensible default
	priority = family_priority or ['PI3K/AKT/MTOR', 'DNA_Damage']

	def fam_label(r):
		for fam in priority:
			try:
				if (r.get('moa_A') == fam) or (r.get('moa_B') == fam):
					return fam
			except Exception:
				continue
		return 'Other'

	df['family'] = df.apply(fam_label, axis=1)
	# consistency: higher is better (negative IQR)
	if 'iqr' in df.columns:
		df['consistency'] = -df['iqr']
	else:
		df['consistency'] = np.nan

	return df



def _load_results_for_distributions(results_dir: str) -> dict:
    """Load only the files needed for distribution plots from a results dir.

    Returns a reduced `results` dict compatible with the helpers in this module.
    """
    results = _load_main_results(results_dir)
    reduced = {
        'files': {
            'predictions': results.get('files', {}).get('predictions'),
            'predictions_drugnames': results.get('files', {}).get('predictions_drugnames'),
            'predictions_inhibitor_groups': results.get('files', {}).get('predictions_inhibitor_groups'),
            'experimental': results.get('files', {}).get('experimental'),
        },
        'dicts': results.get('dicts', {}),
        'results_dir': results.get('results_dir')
    }
    return reduced


#///////////////////////////////////////////////////////////////////////////////////////////////////////
# ---------------------------
# Plotting functions
# ---------------------------
def plot_violin_scatter(predictions_melted: pd.DataFrame,
						PD_combi: Optional[List[str]] = None,
						results: Optional[dict] = None,
						moa_colors: Optional[Dict[str, str]] = None,
						family_priority: Optional[List[str]] = None,
						plot_name: Optional[str] = None,
						width: Optional[int] = None,
						height: Optional[int] = None,
						plot_dir: Optional[str] = None,
						show: bool = True) -> go.Figure:
	"""Violin + scatter figure.

	Accepts either a `predictions_melted` DataFrame and optional `PD_combi`,
	or a `results` dict (from `_load_main_results`) — when `PD_combi` is not
	provided it will be inferred from `results` using `_prepare_exp_pairs`.
	If `moa_colors` is not provided, uses `_style_moa_colors(results)` when
	`results` is available, otherwise falls back to a small default palette.
	"""
	# determine colors
	if moa_colors is None:
		moa_colors = _style_moa_colors(results)

	df = predictions_melted.copy()
	# prepare stacked mechanism distributions
	A_part = df[["synergy", "moa_group_A"]].rename(columns={"moa_group_A": "mechanism"}).copy()
	B_part = df[["synergy", "moa_group_B"]].rename(columns={"moa_group_B": "mechanism"}).copy()
	stacked = pd.concat([A_part, B_part], ignore_index=True).dropna(subset=["synergy"])

	order_mech = (stacked.groupby("mechanism", observed=False)["synergy"].median().sort_values(ascending=True).index.tolist())
	stacked["mechanism"] = pd.Categorical(stacked["mechanism"], order_mech, ordered=True)

	summary_A = (stacked.groupby("mechanism", observed=False)["synergy"]
				 .agg(median="median", q1=lambda x: np.nanpercentile(x, 25), q3=lambda x: np.nanpercentile(x, 75)).reset_index())

	# Infer PD_combi from results if not provided
	if PD_combi is None and results is not None:
		PD_combi = _prepare_exp_pairs(results)

	# compute pair stats (use lower-case drug columns from predictions)
	pair_stats = (df.groupby(["PD_A", "PD_B"], as_index=False)
				  .agg(median=("synergy", "median"),
					   iqr=("synergy", lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
					   n=("synergy", "size"),
					   drug_name_A=("drug_name_A", "first"),
					   drug_name_B=("drug_name_B", "first"),
					   moa_A=("moa_group_A", "first"),
					   moa_B=("moa_group_B", "first")))

	# Use helper to compute family and consistency
	pair_stats = _prepare_family_pairs(pair_stats, PD_combi or [], family_priority=family_priority)

	family_order = [f for f in (family_priority or ['PI3K/AKT/MTOR', 'DNA_Damage'])] + ['Other']
	family_colors = {}
	for fam in family_order:
		family_colors[fam] = moa_colors.get(fam, moa_colors.get('Unlabeled', '#757575'))

	fig = make_subplots(rows=1, cols=2, subplot_titles=("Synergy distributions by mechanism", "Pair-level magnitude vs consistency"),
						specs=[[{"type": "violin"}, {"type": "xy"}]],
						column_widths=[0.6, 0.4], horizontal_spacing=0.1)

	for mech in order_mech:
		yvals = stacked.loc[stacked["mechanism"] == mech, "synergy"].dropna().values
		if len(yvals) == 0:
			continue
		fig.add_trace(go.Violin(y=yvals, name=mech, box_visible=True, meanline_visible=False,
								 points="all", marker=dict(color=moa_colors.get(mech, '#cccccc')), showlegend=False), row=1, col=1)

	if 'selected' not in pair_stats.columns and PD_combi is not None:
		pair_stats['selected'] = pair_stats.apply(lambda r: f"{r['PD_A']} | {r['PD_B']}" in PD_combi, axis=1)

	for fam in family_order:
		sub = pair_stats[pair_stats["family"] == fam]
		if sub.empty:
			continue
		not_sel = sub[~sub['selected']]
		sel = sub[sub['selected']]
		if not not_sel.empty:
			fig.add_trace(go.Scatter(x=not_sel['median'], y=not_sel['consistency'], mode='markers',
									 marker=dict(size=10, color=family_colors[fam], line=dict(width=0.5, color='white')),
									 name=fam, showlegend=True), row=1, col=2)
		if not sel.empty:
			fig.add_trace(go.Scatter(x=sel['median'], y=sel['consistency'], mode='markers',
									 marker=dict(size=13, color=family_colors[fam], line=dict(width=1.2, color='white'), symbol='diamond'),
									 name=fam + ' (selected)', showlegend=True), row=1, col=2)

	fig.add_vline(x=0, line_dash="dash", line_color="#b6b5b5", row=1, col=2)
	fig.add_hline(y=0, line_dash="dash", line_color="#b6b5b5", row=1, col=2)

	fig.update_layout(plot_bgcolor="#eeeeee", height=height or 600, width=width or 1400)
	fig.update_xaxes(title_text="", row=1, col=1)
	fig.update_yaxes(title_text="Synergy score", row=1, col=1)
	fig.update_xaxes(title_text="Median synergy (across cell lines)", row=1, col=2)
	fig.update_yaxes(title_text="Consistency (-IQR)", row=1, col=2)

	fig.add_annotation(dict(x=-0.05, y=1.1, xref='paper', yref='paper', text='A', showarrow=False,
							font=dict(size=28, family='Arial', color='black'), xanchor='left', yanchor='top'))
	fig.add_annotation(dict(x=0.58, y=1.1, xref='paper', yref='paper', text='B', showarrow=False,
							font=dict(size=28, family='Arial', color='black'), xanchor='left', yanchor='top'))

	if show:
		try:
			fig.show()
		except ValueError as e:
			msg = str(e)
			# Common notebook rendering error when nbformat is missing
			if 'nbformat' in msg or 'Mime type rendering' in msg:
				try:
					save_fig(fig, plot_dir or '.', plot_name or 'violin_scatter_plot', formats=['png', 'html'], fig_type='plotly')
					print(f"Plotly renderer fallback: saved violin/scatter to {plot_dir or '.'}")
				except Exception:
					logging.getLogger(__name__).exception('Fallback save for violin+scatter failed')
			else:
				raise

	# Save outputs if requested
	if plot_dir:
		save_fig(fig, plot_dir, plot_name or 'violin_scatter_plot', formats=['svg', 'html'], scale=2, fig_type='plotly')

	return fig


def plot_mechanism_summary_table(
	mechanism_summary: pd.DataFrame,
	plot_name: Optional[str] = None,
	width: Optional[int] = None,
	height: Optional[int] = None,
	plots_dir: Optional[str] = None,
	show: bool = True,
) -> go.Figure:
	"""Render a mechanism-level summary as an HTML table (plotly).

	Accepts optional `plot_name`, `width`, and `height` for layout and
	file output. When `plots_dir` is provided the function will attempt to
	save an Excel file and an HTML view using `save_fig`.
	"""
	df = mechanism_summary.copy()
	df = df.rename(columns={
		'moa_A': 'Mechanism A',
		'moa_B': 'Mechanism B',
		'n_predicted_pairs': 'N. Predicted Pairs',
		'median': 'Median Synergy',
		'iqr': 'IQR',
		'mean': 'Mean Synergy',
		'std': 'SD Synergy',
		'n_selected_pairs': 'N. Selected Pairs'
	})

    # Filter df by combinations with at least one predicted pair
	df = df[df['N. Predicted Pairs'] > 0].reset_index(drop=True)
	
    # Sort by number of selected pairs descending
	df = df.sort_values(by=['N. Selected Pairs', 'N. Predicted Pairs'], ascending=[False, False]).reset_index(drop=True)

	fig = go.Figure(data=[go.Table(
		header=dict(values=list(df.columns), fill_color='paleturquoise', align='center'),
		cells=dict(values=[df[col] for col in df.columns], fill_color="#ecf5f7", align='left'))])

	fig.update_traces(cells=dict(height=30))
	fig.update_layout(title="Summary of Predicted Synergistic Pairs by Mechanism Combination",
					  height=height or 500, width=width or 1000)

	if plots_dir:
		try:
			os.makedirs(plots_dir, exist_ok=True)
		except Exception:
			pass
		# Save table as Excel when possible
		try:
			df.to_csv(os.path.join(plots_dir, 'Table1.csv'), index=False)
		except Exception:
			pass
		# Save HTML view via helper
		try:
			save_fig(fig, plots_dir, plot_name or 'mechanism_summary_table', formats=['html'], fig_type='plotly')
		except Exception:
			pass

	if show:
		try:
			fig.show()
		except ValueError as e:
			msg = str(e)
			if 'nbformat' in msg or 'Mime type rendering' in msg:
				try:
					save_fig(fig, plots_dir or '.', plot_name or 'mechanism_summary_table', formats=['html', 'png'], fig_type='plotly')
					print(f"Plotly renderer fallback: saved mechanism summary to {plots_dir or '.'}")
				except Exception:
					logging.getLogger(__name__).exception('Fallback save for mechanism summary failed')
			else:
				raise
	return fig


def make_pred_distribution_plots(
	results_dir: str,
	plots_dir: Optional[str] = None,
	family_priority: Optional[List[str]] = None,
	violin_scatter_plot_name: str = 'Figure1',
	table_plot_name: str = 'Figure1_mechanism_summary',
	show: bool = False,
	violin_size: Optional[tuple] = None,
	table_size: Optional[tuple] = None,
) -> None:
	"""Load results, prepare data and create distribution-related figures.

	This wrapper loads the minimal files for distributions, prepares the
	predictions and summaries, and renders/saves the violin + scatter and
	mechanism summary table.
	"""
	results = _load_results_for_distributions(results_dir)

	if plots_dir is None:
		plots_dir = os.path.join(results.get('results_dir', results_dir), 'plots')
	os.makedirs(plots_dir, exist_ok=True)

	# Prepare predictions and derived summaries
	preds = _prepare_predictions(results)
	PD_combi = _prepare_exp_pairs(results)
	pair_table = _prepare_pairs_table(preds, PD_combi, supplementary_files_path=os.path.join(results.get('results_dir', results_dir), 'supplementary_files'))
	family_pairs = _prepare_family_pairs(pair_table, PD_combi, family_priority=family_priority)
	mechanism_summary = _prepare_mechanism_summary(pair_table, results, supplementary_files_path=os.path.join(results.get('results_dir', results_dir), 'supplementary_files'))

	moa_colors = _style_moa_colors(results)

	# Violin + scatter
	vw, vh = (violin_size or (1400, 600))
	try:
		plot_violin_scatter(preds,
							PD_combi=PD_combi,
							results=results,
							moa_colors=moa_colors,
							family_priority=family_priority,
							plot_dir=plots_dir,
							plot_name=f"{violin_scatter_plot_name}_violin",
							width=vw,
							height=vh,
							show=show)
	except Exception:
		logging.getLogger(__name__).exception('Failed to render violin + scatter')

	# Mechanism summary table
	if mechanism_summary is not None:
		tw, th = (table_size or (1000, 500))
		try:
			plot_mechanism_summary_table(mechanism_summary,
										plot_name=table_plot_name,
										width=tw,
										height=th,
										plots_dir=plots_dir,
										show=show)
		except Exception:
			logging.getLogger(__name__).exception('Failed to render mechanism summary table')

