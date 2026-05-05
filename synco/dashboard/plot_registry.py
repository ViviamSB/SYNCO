"""
plot_registry.py – Per-plot callable registry for the Explorer gallery.

Architecture
------------
Each entry in ``PLOT_REGISTRY`` maps a ``(context, tab)`` tuple to a list of
``PlotSpec`` objects.  The two contexts are:

- ``"cross_tissue"``  — input is *cell_fate_dir* (multi-tissue root)
- ``"tissue"``        — input is *results_dir* (single-tissue synco_output)

Every ``PlotSpec`` holds:

- ``plot_id``    – unique slug used as the HTML component ID.
- ``label``      – short display title shown in the gallery card header.
- ``description``– one-line caption shown under the title.
- ``func``       – callable with signature::

      func(primary_dir, plots_dir=None, filters=None, return_fig=False)
          -> list[tuple[fig, str]] | None

  When ``return_fig=True`` the function returns a list of ``(fig, fig_type)``
  tuples where ``fig_type`` is ``"plotly"`` or ``"matplotlib"``.
  When ``return_fig=False`` (default) it saves files under ``plots_dir`` and
  returns ``None`` — the legacy notebook / CLI behaviour is preserved.

- ``input_type`` – ``"cell_fate_dir"`` or ``"results_dir"``.

All wrapper functions use *lazy imports* so heavy plotting libraries are only
loaded when the user clicks "Render".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# PlotSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlotSpec:
    plot_id:           str
    label:             str
    description:       str
    func:              Callable
    input_type:        str                # "cell_fate_dir" | "results_dir"
    supported_filters: frozenset = frozenset()  # filter keys this plot can use


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NoFilterMatchError(ValueError):
    """Raised when active dashboard filters produce no matching data.

    Caught by ``adapters.render_one_plot`` to display a user-friendly
    "no data" message instead of a blank or errored plot card.
    """


# ---------------------------------------------------------------------------
# Wrapper functions – Cross-tissue (input: cell_fate_dir)
# ---------------------------------------------------------------------------

def _tissue_cls_boxes(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_cls_boxes
    figs = plot_tissue_cls_boxes(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_cls_violins(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_cls_violins
    figs = plot_tissue_cls_violins(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_cls_bars(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_cls_bars
    figs = plot_tissue_cls_bars(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_cls_boxes_cl(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_cls_boxes_cl
    figs = plot_tissue_cls_boxes_cl(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_cls_violins_cl(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_cls_violins_cl
    figs = plot_tissue_cls_violins_cl(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_rings(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_rings
    figs = plot_tissue_rings(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _aggregate_ring(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_aggregate_ring
    figs = plot_aggregate_ring(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_boxes_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_boxes_ct
    figs = plot_roc_pr_boxes_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_metric_violins_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_metric_violins_ct
    figs = plot_roc_pr_metric_violins_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_heatmap_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_heatmap_ct
    figs = plot_roc_pr_heatmap_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_bars_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_bars_ct
    figs = plot_roc_pr_bars_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_violin_roc_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_violin_roc_ct
    figs = plot_roc_violin_roc_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_violin_pr_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_violin_pr_ct
    figs = plot_roc_violin_pr_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_metric_violins_cl_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_metric_violins_cl_ct
    figs = plot_roc_pr_metric_violins_cl_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_violin_table_roc_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_violin_table_roc_ct
    figs = plot_roc_pr_violin_table_roc_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_violin_table_pr_ct(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_violin_table_pr_ct
    figs = plot_roc_pr_violin_table_pr_ct(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _exp_dist_by_tissue(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_exp_dist_by_tissue
    figs = plot_exp_dist_by_tissue(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _exp_dist_by_combo(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_exp_dist_by_combo
    figs = plot_exp_dist_by_combo(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _exp_synergy_counts(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_exp_synergy_counts
    figs = plot_exp_synergy_counts(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _exp_distributions_cross_tissue(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_exp_distributions_all
    figs = plot_exp_distributions_all(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _pred_distributions_cross_tissue(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_pred_distributions_all
    figs = plot_pred_distributions_all(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _profile_categories_cross_tissue(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_profiles_all
    figs = plot_profiles_all(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


# ---------------------------------------------------------------------------
# Wrapper functions – Single tissue (input: results_dir)
# ---------------------------------------------------------------------------

def _cls_cell_line(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.classification import plot_by_cell_line
    figs = plot_by_cell_line(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _cls_combination(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.classification import plot_by_combination
    figs = plot_by_combination(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _ring_summary(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.performance import plot_ring_summary
    figs = plot_ring_summary(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _cell_line_rings(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.performance import plot_cell_line_rings
    figs = plot_cell_line_rings(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _combination_rings(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.performance import plot_combination_rings
    figs = plot_combination_rings(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_curves(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.roc import plot_roc_pr_curves
    figs = plot_roc_pr_curves(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _threshold_sweeps(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.roc import plot_threshold_sweeps
    figs = plot_threshold_sweeps(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _exp_distributions(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.distributions import plot_experimental
    figs = plot_experimental(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _pred_distributions(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.distributions import plot_predicted
    figs = plot_predicted(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _pred_distributions_by_inhibitor_group(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.distributions import plot_predicted_by_inhibitor_group
    figs = plot_predicted_by_inhibitor_group(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _profile_categories(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.profiles import plot_profile_categories
    figs = plot_profile_categories(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _cls_cell_heatmap(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.classification import plot_cell_line_heatmap
    figs = plot_cell_line_heatmap(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _cls_cell_boxes(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.classification import plot_cell_line_boxes
    figs = plot_cell_line_boxes(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _cls_combi_heatmap(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.classification import plot_combination_heatmap
    figs = plot_combination_heatmap(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _cls_combi_boxes(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.classification import plot_combination_boxes
    figs = plot_combination_boxes(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _perf_funnel(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.performance import plot_funnel_efficiency
    figs = plot_funnel_efficiency(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_auc_bars(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.roc import plot_auc_bars
    figs = plot_auc_bars(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_auc_summary(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.roc import plot_auc_summary
    figs = plot_auc_summary(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


# ---------------------------------------------------------------------------
# Registry  (2 contexts × 5 tabs)
# ---------------------------------------------------------------------------

PLOT_REGISTRY: dict[tuple[str, str], list[PlotSpec]] = {

    # ── Cross-tissue ──────────────────────────────────────────────────────
    ("cross_tissue", "classification"): [
        PlotSpec(
            "ct_cls_boxes",
            "Metric box plots",
            "Accuracy, Recall and Precision box plots across tissues.",
            _tissue_cls_boxes,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_cls_violins",
            "Metric violin plots",
            "Accuracy, Recall and Precision violin plots across tissues.",
            _tissue_cls_violins,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_cls_bars",
            "Metric bar plots",
            "Accuracy, Recall and Precision bar plots across tissues.",
            _tissue_cls_bars,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_cls_boxes_cl",
            "Metric box plots (cell-line)",
            "Accuracy, Recall and Precision boxes — individual cell lines as data points.",
            _tissue_cls_boxes_cl,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_cls_violins_cl",
            "Metric violin plots (cell-line)",
            "Accuracy, Recall and Precision violins — individual cell lines as data points.",
            _tissue_cls_violins_cl,
            "cell_fate_dir",
            frozenset(),
        ),
    ],
    ("cross_tissue", "performance"): [
        PlotSpec(
            "ct_perf_agg_ring",
            "Aggregate ring",
            "Single aggregate ring summarising all tissues.",
            _aggregate_ring,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_perf_rings",
            "Tissue rings",
            "Per-tissue TP / TN / FP / FN donut rings.",
            _tissue_rings,
            "cell_fate_dir",
            frozenset(),
        ),
    ],
    ("cross_tissue", "roc"): [
        PlotSpec(
            "ct_roc_boxes",
            "Metric box plots",
            "F1 / AUC-ROC / AUC-PR box plots across tissues.",
            _roc_pr_boxes_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_metric_violins",
            "Metric violin plots",
            "F1 / AUC-ROC / AUC-PR violin plots across tissues.",
            _roc_pr_metric_violins_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_metric_violins_cl",
            "Metric violin plots (cell-line)",
            "F1 / AUC-ROC / AUC-PR violins — individual cell lines as data points.",
            _roc_pr_metric_violins_cl_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_heatmap",
            "AUC heatmap",
            "AUC score heatmap per tissue.",
            _roc_pr_heatmap_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_bars",
            "AUC bar plots",
            "F1 / AUC bar plots across tissues.",
            _roc_pr_bars_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_auc_violin",
            "AUC-ROC violin",
            "AUC-ROC score distribution across tissues.",
            _roc_violin_roc_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_pr_auc_violin",
            "AUC-PR violin",
            "AUC-PR score distribution across tissues.",
            _roc_violin_pr_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_violin_table_roc",
            "AUC-ROC violin + table",
            "AUC-ROC violin plots with per-tissue summary statistics table.",
            _roc_pr_violin_table_roc_ct,
            "cell_fate_dir",
            frozenset(),
        ),
        PlotSpec(
            "ct_roc_violin_table_pr",
            "AUC-PR violin + table",
            "AUC-PR violin plots with per-tissue summary statistics table.",
            _roc_pr_violin_table_pr_ct,
            "cell_fate_dir",
            frozenset(),
        ),
    ],
    ("cross_tissue", "distributions"): [
        PlotSpec(
            "ct_exp_dist_tissue",
            "Score dist. by tissue",
            "Synergy scores across tissues: tissues on y-axis, per cell line.",
            _exp_dist_by_tissue,
            "cell_fate_dir",
            frozenset({"drug", "profile", "combination"}),
        ),
        PlotSpec(
            "ct_exp_dist_combo",
            "Score dist. by combination",
            "Synergy scores across tissues: drug combinations on y-axis.",
            _exp_dist_by_combo,
            "cell_fate_dir",
            frozenset({"drug", "profile"}),
        ),
        PlotSpec(
            "ct_exp_synergy_counts",
            "Synergy counts",
            "Synergistic combination and cell-line counts per tissue.",
            _exp_synergy_counts,
            "cell_fate_dir",
            frozenset({"drug", "profile", "combination"}),
        ),
        PlotSpec(
            "ct_pred_dist",
            "Pred. distributions",
            "Predicted synergy violin + scatter plots per tissue.",
            _pred_distributions_cross_tissue,
            "cell_fate_dir",
            frozenset({"drug", "profile"}),
        ),
    ],
    ("cross_tissue", "profiles"): [
        PlotSpec(
            "ct_profiles",
            "Profile categories",
            "Drug profile parallel-categories charts per tissue.",
            _profile_categories_cross_tissue,
            "cell_fate_dir",
            frozenset({"drug", "combination", "profile"}),
        ),
    ],

    # ── Single tissue ─────────────────────────────────────────────────────
    ("tissue", "classification"): [
        PlotSpec(
            "tis_cls_cell",
            "By cell line",
            "Classification metrics (accuracy, recall, precision, AUC) per cell line.",
            _cls_cell_line,
            "results_dir",
            frozenset({"cell_line"}),
        ),
        PlotSpec(
            "tis_cls_combi",
            "By combination",
            "Classification metrics per drug combination.",
            _cls_combination,
            "results_dir",
            frozenset({"combination"}),
        ),
        PlotSpec(
            "tis_cls_cell_heatmap",
            "Metric heatmaps (cell line)",
            "Accuracy / Recall / Precision and F1 / AUC heatmaps per cell line.",
            _cls_cell_heatmap,
            "results_dir",
            frozenset(),
        ),
        PlotSpec(
            "tis_cls_cell_boxes",
            "Metric box plots (cell line)",
            "Accuracy / Recall / Precision and F1 / AUC box plots across cell lines.",
            _cls_cell_boxes,
            "results_dir",
            frozenset(),
        ),
        PlotSpec(
            "tis_cls_combi_heatmap",
            "Metric heatmap (combination)",
            "Accuracy / Recall / Precision heatmap per drug combination.",
            _cls_combi_heatmap,
            "results_dir",
            frozenset(),
        ),
        PlotSpec(
            "tis_cls_combi_boxes",
            "Metric box plots (combination)",
            "Accuracy / Recall / Precision box plots across drug combinations.",
            _cls_combi_boxes,
            "results_dir",
            frozenset(),
        ),
    ],
    ("tissue", "performance"): [
        PlotSpec(
            "tis_perf_funnel",
            "Modelling efficiency funnel",
            "Funnel chart: total combinations → priority experiments → synergies.",
            _perf_funnel,
            "results_dir",
            frozenset(),
        ),
        PlotSpec(
            "tis_perf_ring",
            "Aggregate ring",
            "Overall TP / TN / FP / FN donut ring for this tissue.",
            _ring_summary,
            "results_dir",
            frozenset({"combination", "drug", "profile"}),
        ),
        PlotSpec(
            "tis_perf_cl_rings",
            "Cell-line rings",
            "Per-cell-line TP / TN / FP / FN donut rings.",
            _cell_line_rings,
            "results_dir",
            frozenset({"cell_line"}),
        ),
        PlotSpec(
            "tis_perf_cb_rings",
            "Combination rings",
            "Per-combination TP / TN / FP / FN donut rings.",
            _combination_rings,
            "results_dir",
            frozenset({"combination", "drug", "profile"}),
        ),
    ],
    ("tissue", "roc"): [
        PlotSpec(
            "tis_roc_curves",
            "ROC / PR curves",
            "Per-cell-line ROC and Precision-Recall curves with AUC.",
            _roc_pr_curves,
            "results_dir",
            frozenset({"cell_line"}),
        ),
        PlotSpec(
            "tis_roc_sweeps",
            "Threshold sweeps",
            "AUC score sweep across classification thresholds.",
            _threshold_sweeps,
            "results_dir",
            frozenset({"cell_line"}),
        ),
        PlotSpec(
            "tis_roc_auc_bars",
            "AUC score bar chart",
            "AUC-ROC / AUC-PR / F1 Score horizontal bar chart per cell line.",
            _roc_auc_bars,
            "results_dir",
            frozenset({"cell_line"}),
        ),
        PlotSpec(
            "tis_roc_auc_summary",
            "AUC violin + table",
            "F1 / AUC-ROC / AUC-PR violin plots with summary statistics table.",
            _roc_auc_summary,
            "results_dir",
            frozenset({"cell_line"}),
        ),
    ],
    ("tissue", "distributions"): [
        PlotSpec(
            "tis_exp_dist",
            "Exp. distributions",
            "Synergy counts and distribution histograms.",
            _exp_distributions,
            "results_dir",
            frozenset({"cell_line", "combination", "drug", "profile"}),
        ),
        PlotSpec(
            "tis_pred_dist",
            "Pred. distributions (mechanism)",
            "Violin + scatter of predicted synergy grouped by mechanism class (when available) or inhibitor group.",
            _pred_distributions,
            "results_dir",
            frozenset({"cell_line", "drug", "profile"}),
        ),
        PlotSpec(
            "tis_pred_dist_inh_group",
            "Pred. distributions (inhibitor groups)",
            "Violin plots of predicted synergy grouped explicitly by inhibitor group pair.",
            _pred_distributions_by_inhibitor_group,
            "results_dir",
            frozenset({"cell_line", "drug", "profile"}),
        ),
    ],
    ("tissue", "profiles"): [
        PlotSpec(
            "tis_profiles",
            "Profile categories",
            "Drug profile and combination parallel-categories chart.",
            _profile_categories,
            "results_dir",
            frozenset({"drug", "combination", "profile"}),
        ),
    ],
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_specs(context: str, tab: str) -> list[PlotSpec]:
    """Return the list of PlotSpecs for *(context, tab)*, or ``[]``."""
    return PLOT_REGISTRY.get((context, tab), [])


def get_spec_by_id(plot_id: str) -> Optional[PlotSpec]:
    """Look up a PlotSpec by its ``plot_id`` across the entire registry."""
    for specs in PLOT_REGISTRY.values():
        for spec in specs:
            if spec.plot_id == plot_id:
                return spec
    return None


def get_tab_filters(context: str, tab: str) -> frozenset:
    """Return the union of ``supported_filters`` for all specs in *(context, tab)*.

    Used by the filter panel to show only dropdowns that are relevant to the
    currently active tab.  Returns an empty frozenset if no specs are defined
    or none of them declare any supported filters.
    """
    result: set = set()
    for spec in PLOT_REGISTRY.get((context, tab), []):
        result |= spec.supported_filters
    return frozenset(result)
