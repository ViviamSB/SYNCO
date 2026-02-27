"""
config_form.py – Collapsible configuration form for the SYNCO pipeline.

The form is organised as a ``dbc.Accordion`` with sections for each
config group.  ``FORM_FIELD_IDS`` maps logical names to Dash component IDs
so callbacks can reference fields without hard-coding strings in multiple
places.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

# ---------------------------------------------------------------------------
# Component-ID registry  (import from here, never hard-code in callbacks)
# ---------------------------------------------------------------------------
FORM_FIELD_IDS = {
    # -- Paths (required) --
    "base":             "cfg-path-base",
    "pipeline_runs":    "cfg-path-pipeline-runs",
    "input":            "cfg-path-input",
    "output":           "cfg-path-output",
    "cell_fate_dir":    "cfg-path-cell-fate-dir",
    # -- General --
    "cell_lines":        "cfg-cell-lines",
    "prediction_method": "cfg-prediction-method",
    "verbose":           "cfg-verbose",
    # -- Comparison --
    "threshold":          "cfg-threshold",
    "analysis_mode":      "cfg-analysis-mode",
    "duplicate_strategy": "cfg-duplicate-strategy",
    # -- ROC / Bootstrap --
    "threshold_offsets": "cfg-threshold-offsets",
    "roc_bootstrap_n":   "cfg-roc-bootstrap-n",
    "roc_bootstrap_ci":  "cfg-roc-bootstrap-ci",
    # -- Output control --
    "output_control_enabled":           "cfg-oc-enabled",
    "write_profiles":                   "cfg-oc-write-profiles",
    "write_experimental_full_df":       "cfg-oc-write-exp-full",
    "write_predictions_full_df":        "cfg-oc-write-pred-full",
    "write_synergy_predictions":        "cfg-oc-write-syn-pred",
    "write_compare_outputs":            "cfg-oc-write-compare",
    "write_roc_outputs":                "cfg-oc-write-roc",
    # -- Advanced --
    "advance_json": "cfg-advance-json",
}


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _label(text: str, required: bool = False) -> dbc.Label:
    children = [text]
    if required:
        children.append(html.Span(" *", className="text-danger"))
    return dbc.Label(children)


def _path_row(label: str, field_key: str, placeholder: str, required: bool = False) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(_label(label, required=required), width=4),
            dbc.Col(
                dbc.Input(
                    id=FORM_FIELD_IDS[field_key],
                    type="text",
                    placeholder=placeholder,
                    debounce=True,
                    persistence=True,
                    persistence_type="session",
                ),
                width=8,
            ),
        ],
        className="mb-2 align-items-center",
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_paths() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            _path_row("Base directory", "base",          "Absolute path to project root", required=True),
            _path_row("Pipeline runs",  "pipeline_runs", "Directory containing prediction outputs", required=True),
            _path_row("Input data",     "input",         "Directory with synergies & drug profiles", required=True),
            _path_row("Output",         "output",        "Where synco_output/ will be created", required=True),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        _label("Cell fate directory", required=False),
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Input(
                                id=FORM_FIELD_IDS["cell_fate_dir"],
                                type="text",
                                placeholder="Parent dir for multi-tissue analysis (optional)",
                                debounce=True,
                                persistence=True,
                                persistence_type="session",
                            ),
                            dbc.FormText(
                                "Only needed for Tissue-level visualisations in the Explorer.",
                                color="secondary",
                            ),
                        ],
                        width=8,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
        ],
        title="Paths  (required)",
        item_id="acc-paths",
    )


def _section_general() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            dbc.Row(
                [
                    dbc.Col(_label("Cell lines"), width=4),
                    dbc.Col(
                        [
                            dbc.Input(
                                id=FORM_FIELD_IDS["cell_lines"],
                                type="text",
                                placeholder="Comma-separated list, CSV filename, or leave blank for auto-discover",
                                debounce=True,
                                persistence=True,
                                persistence_type="session",
                            ),
                            dbc.FormText(
                                'e.g. "AsPC-1, BxPC-3"  or  "cell_lines.csv"  or blank (auto)',
                                color="secondary",
                            ),
                        ],
                        width=8,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(_label("Prediction method"), width=4),
                    dbc.Col(
                        dbc.Select(
                            id=FORM_FIELD_IDS["prediction_method"],
                            options=[
                                {"label": "DrugLogics", "value": "DrugLogics"},
                                {"label": "BooLEVARD",  "value": "BooLEVARD"},
                            ],
                            value="DrugLogics",
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Switch(
                            id=FORM_FIELD_IDS["verbose"],
                            label="Verbose logging",
                            value=False,
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=4,
                        className="d-flex align-items-center",
                    ),
                ],
                className="mb-2",
            ),
        ],
        title="General",
        item_id="acc-general",
    )


def _section_comparison() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            dbc.Row(
                [
                    dbc.Col(_label("Synergy threshold"), width=4),
                    dbc.Col(
                        dbc.Input(
                            id=FORM_FIELD_IDS["threshold"],
                            type="number",
                            value=0.5,
                            step=0.01,
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=3,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(_label("Analysis mode"), width=4),
                    dbc.Col(
                        dbc.Select(
                            id=FORM_FIELD_IDS["analysis_mode"],
                            options=[
                                {"label": "Cell line",            "value": "cell_line"},
                                {"label": "Inhibitor combination","value": "inhibitor_combination"},
                            ],
                            value="cell_line",
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=4,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(_label("Duplicate strategy"), width=4),
                    dbc.Col(
                        dbc.Select(
                            id=FORM_FIELD_IDS["duplicate_strategy"],
                            options=[
                                {"label": "Mean",  "value": "mean"},
                                {"label": "First", "value": "first"},
                            ],
                            value="mean",
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=4,
                    ),
                ],
                className="mb-2",
            ),
        ],
        title="Comparison",
        item_id="acc-comparison",
    )


def _section_roc() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            dbc.Row(
                [
                    dbc.Col(_label("Threshold offsets"), width=4),
                    dbc.Col(
                        [
                            dbc.Input(
                                id=FORM_FIELD_IDS["threshold_offsets"],
                                type="text",
                                placeholder="e.g.  -2, -1, 0, 1, 2",
                                debounce=True,
                                persistence=True,
                                persistence_type="session",
                            ),
                            dbc.FormText(
                                "Comma-separated numeric offsets for threshold sensitivity sweep.",
                                color="secondary",
                            ),
                        ],
                        width=8,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(_label("Bootstrap samples"), width=4),
                    dbc.Col(
                        dbc.Input(
                            id=FORM_FIELD_IDS["roc_bootstrap_n"],
                            type="number",
                            placeholder="e.g. 500  (blank = disabled)",
                            min=10, step=10,
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=3,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(_label("Bootstrap CI level"), width=4),
                    dbc.Col(
                        dbc.Input(
                            id=FORM_FIELD_IDS["roc_bootstrap_ci"],
                            type="number",
                            value=0.95,
                            min=0.5, max=0.999, step=0.01,
                            persistence=True,
                            persistence_type="session",
                        ),
                        width=3,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
        ],
        title="ROC / Bootstrap",
        item_id="acc-roc",
    )


def _section_output_control() -> dbc.AccordionItem:
    write_switches = [
        ("cfg-oc-write-profiles",   "Write drug profiles"),
        ("cfg-oc-write-exp-full",   "Write experimental full DF"),
        ("cfg-oc-write-pred-full",  "Write predictions full DF"),
        ("cfg-oc-write-syn-pred",   "Write synergy predictions"),
        ("cfg-oc-write-compare",    "Write comparison outputs"),
        ("cfg-oc-write-roc",        "Write ROC outputs"),
    ]
    return dbc.AccordionItem(
        [
            dbc.Switch(
                id=FORM_FIELD_IDS["output_control_enabled"],
                label="Enable selective output writing",
                value=False,
                className="mb-2",
                persistence=True,
                persistence_type="session",
            ),
            html.Div(
                [dbc.Switch(id=sid, label=lbl, value=False,
                            className="mb-1",
                            persistence=True, persistence_type="session")
                 for sid, lbl in write_switches],
                className="ms-3",
            ),
        ],
        title="Output control",
        item_id="acc-output-control",
    )


def _section_advanced() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            dbc.Label("Advance overrides (JSON)"),
            dbc.Textarea(
                id=FORM_FIELD_IDS["advance_json"],
                placeholder='{}',
                style={"fontFamily": "monospace", "height": "120px"},
                persistence=True,
                persistence_type="session",
            ),
            dbc.FormText(
                "Optional JSON dict to override step-level defaults "
                "(merged into the \"advance\" config section).",
                color="secondary",
            ),
        ],
        title="Advanced overrides",
        item_id="acc-advanced",
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_config_form() -> dbc.Accordion:
    """Return the full pipeline configuration form as a ``dbc.Accordion``."""
    return dbc.Accordion(
        [
            _section_paths(),
            _section_general(),
            _section_comparison(),
            _section_roc(),
            _section_output_control(),
            _section_advanced(),
        ],
        active_item="acc-paths",
        always_open=False,
    )
