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
    plot_id:     str
    label:       str
    description: str
    func:        Callable
    input_type:  str       # "cell_fate_dir" | "results_dir"


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

def _tissue_metric_boxplots(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_metric_boxplots
    figs = plot_tissue_metric_boxplots(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_rings(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_rings
    figs = plot_tissue_rings(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _aggregate_ring(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_aggregate_ring
    figs = plot_aggregate_ring(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _roc_pr_violin(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_roc_pr_violin
    figs = plot_roc_pr_violin(cell_fate_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


def _tissue_roc_pr_f1(cell_fate_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.cross_tissue import plot_tissue_roc_pr_detail
    figs = plot_tissue_roc_pr_detail(cell_fate_dir, filters=filters)
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


def _profile_categories(results_dir, plots_dir=None, filters=None, return_fig=False):
    from synco.dashboard.plots.profiles import plot_profile_categories
    figs = plot_profile_categories(results_dir, filters=filters)
    return [(fig, "plotly") for fig in figs]


# ---------------------------------------------------------------------------
# Registry  (2 contexts × 5 tabs)
# ---------------------------------------------------------------------------

PLOT_REGISTRY: dict[tuple[str, str], list[PlotSpec]] = {

    # ── Cross-tissue ──────────────────────────────────────────────────────
    ("cross_tissue", "classification"): [
        PlotSpec(
            "ct_cls_boxplots",
            "Metric box plots",
            "Accuracy, Recall and Precision distributions across tissues.",
            _tissue_metric_boxplots,
            "cell_fate_dir",
        ),
    ],
    ("cross_tissue", "performance"): [
        PlotSpec(
            "ct_perf_rings",
            "Tissue rings",
            "Per-tissue TP / TN / FP / FN donut rings.",
            _tissue_rings,
            "cell_fate_dir",
        ),
        PlotSpec(
            "ct_perf_agg_ring",
            "Aggregate ring",
            "Single aggregate ring summarising all tissues.",
            _aggregate_ring,
            "cell_fate_dir",
        ),
    ],
    ("cross_tissue", "roc"): [
        PlotSpec(
            "ct_roc_violin",
            "ROC / PR violin",
            "AUC score distributions across tissues.",
            _roc_pr_violin,
            "cell_fate_dir",
        ),
        PlotSpec(
            "ct_roc_detail",
            "ROC / PR / F1 detail",
            "Box, bar and heatmap views per tissue.",
            _tissue_roc_pr_f1,
            "cell_fate_dir",
        ),
    ],
    ("cross_tissue", "distributions"): [
        PlotSpec(
            "ct_exp_dist",
            "Exp. distributions",
            "Experimental synergy count distributions per tissue.",
            _exp_distributions_cross_tissue,
            "cell_fate_dir",
        ),
        PlotSpec(
            "ct_pred_dist",
            "Pred. distributions",
            "Predicted synergy violin + scatter plots per tissue.",
            _pred_distributions_cross_tissue,
            "cell_fate_dir",
        ),
    ],
    ("cross_tissue", "profiles"): [
        PlotSpec(
            "ct_profiles",
            "Profile categories",
            "Drug profile parallel-categories charts per tissue.",
            _profile_categories_cross_tissue,
            "cell_fate_dir",
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
        ),
        PlotSpec(
            "tis_cls_combi",
            "By combination",
            "Classification metrics per drug combination.",
            _cls_combination,
            "results_dir",
        ),
    ],
    ("tissue", "performance"): [
        PlotSpec(
            "tis_perf_ring",
            "Aggregate ring",
            "Overall TP / TN / FP / FN donut ring for this tissue.",
            _ring_summary,
            "results_dir",
        ),
        PlotSpec(
            "tis_perf_cl_rings",
            "Cell-line rings",
            "Per-cell-line TP / TN / FP / FN donut rings.",
            _cell_line_rings,
            "results_dir",
        ),
        PlotSpec(
            "tis_perf_cb_rings",
            "Combination rings",
            "Per-combination TP / TN / FP / FN donut rings.",
            _combination_rings,
            "results_dir",
        ),
    ],
    ("tissue", "roc"): [
        PlotSpec(
            "tis_roc_curves",
            "ROC / PR curves",
            "Per-cell-line ROC and Precision-Recall curves with AUC.",
            _roc_pr_curves,
            "results_dir",
        ),
        PlotSpec(
            "tis_roc_sweeps",
            "Threshold sweeps",
            "AUC score sweep across classification thresholds.",
            _threshold_sweeps,
            "results_dir",
        ),
    ],
    ("tissue", "distributions"): [
        PlotSpec(
            "tis_exp_dist",
            "Exp. distributions",
            "Synergy counts and distribution histograms.",
            _exp_distributions,
            "results_dir",
        ),
        PlotSpec(
            "tis_pred_dist",
            "Pred. distributions",
            "Violin + scatter of predicted synergy scores and mechanism summary.",
            _pred_distributions,
            "results_dir",
        ),
    ],
    ("tissue", "profiles"): [
        PlotSpec(
            "tis_profiles",
            "Profile categories",
            "Drug profile and combination parallel-categories chart.",
            _profile_categories,
            "results_dir",
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
