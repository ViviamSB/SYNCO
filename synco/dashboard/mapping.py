"""
mapping.py – Coverage × Evaluation → plotting function lookup table.

Every key is a (coverage_id, eval_id) tuple.
Every value is a dict with:
  func  – function name as string (resolved via synco.plotting)
  kwargs – extra keyword arguments to pass to the function
  input – "results_dir" | "cell_fate_dir"
"""

# ---------------------------------------------------------------------------
# Coverage levels
# ---------------------------------------------------------------------------
COVERAGE_LEVELS = [
    {"id": "global",       "label": "Global",              "icon": "bi-globe"},
    {"id": "tissue",       "label": "Tissue",              "icon": "bi-diagram-3"},
    {"id": "cell_line",    "label": "Cell Line",           "icon": "bi-grid"},
    {"id": "combis",       "label": "Combinations",        "icon": "bi-bezier2"},
    {"id": "exp_drugs",    "label": "Experimental Drugs",  "icon": "bi-eyedropper"},
    {"id": "drug_profiles","label": "Drug Profiles",       "icon": "bi-bar-chart-steps"},
]

# ---------------------------------------------------------------------------
# Evaluation levels
# ---------------------------------------------------------------------------
EVAL_LEVELS = [
    {"id": "classification", "label": "Classification"},
    {"id": "performance",    "label": "Performance"},
    {"id": "roc",            "label": "ROC / PR"},
    {"id": "exp_dist",       "label": "Exp. Distributions"},
    {"id": "pred_dist",      "label": "Pred. Distributions"},
    {"id": "profile_cat",    "label": "Profile Categories"},
]

# ---------------------------------------------------------------------------
# (coverage, evaluation) → plot specification
# ---------------------------------------------------------------------------
PLOT_MAP = {
    # --- Global ---
    ("global", "classification"): {
        "func":   "make_classification_plots",
        "kwargs": {},
        "input":  "results_dir",
    },
    ("global", "performance"): {
        "func":   "make_performance_plots",
        "kwargs": {"performance": "both"},
        "input":  "results_dir",
    },
    ("global", "roc"): {
        "func":   "make_roc_plots",
        "kwargs": {},
        "input":  "results_dir",
    },

    # --- Tissue ---
    ("tissue", "classification"): {
        "func":   "make_multi_tissue_plots",
        "kwargs": {},
        "input":  "cell_fate_dir",
    },
    ("tissue", "performance"): {
        "func":   "make_multi_tissue_plots",
        "kwargs": {},
        "input":  "cell_fate_dir",
    },
    ("tissue", "roc"): {
        "func":   "make_multi_tissue_plots",
        "kwargs": {},
        "input":  "cell_fate_dir",
    },

    # --- Cell Line ---
    ("cell_line", "classification"): {
        "func":   "make_classification_plots",
        "kwargs": {"analysis_type": "cell_line"},
        "input":  "results_dir",
    },
    ("cell_line", "performance"): {
        "func":   "make_ring_plots",
        "kwargs": {"analysis_type": "cell_line"},
        "input":  "results_dir",
    },
    ("cell_line", "roc"): {
        "func":   "make_roc_plots",
        "kwargs": {},
        "input":  "results_dir",
    },

    # --- Combinations ---
    ("combis", "classification"): {
        "func":   "make_classification_plots",
        "kwargs": {"analysis_type": "combination"},
        "input":  "results_dir",
    },
    ("combis", "performance"): {
        "func":   "make_ring_plots",
        "kwargs": {"analysis_type": "inhibitor_combination"},
        "input":  "results_dir",
    },

    # --- Experimental Drugs ---
    ("exp_drugs", "exp_dist"): {
        "func":   "make_experimental_distribution_plots",
        "kwargs": {},
        "input":  "results_dir",
    },

    # --- Drug Profiles ---
    ("drug_profiles", "pred_dist"): {
        "func":   "make_pred_distribution_plots",
        "kwargs": {},
        "input":  "results_dir",
    },
    ("drug_profiles", "profile_cat"): {
        "func":   "make_profilecat_plots",
        "kwargs": {},
        "input":  "results_dir",
    },
}


def get_valid_evals(coverage: str) -> list:
    """Return the list of evaluation IDs that are valid for *coverage*."""
    return [eval_id for (cov, eval_id) in PLOT_MAP if cov == coverage]
