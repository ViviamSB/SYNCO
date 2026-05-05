import streamlit as st
from typing import Dict, Any
from pathlib import Path


def build_config_form() -> Dict[str, Any]:
    """
    Build and display the configuration form, returning the complete config dict.
    Uses Streamlit expanders to organize sections like the Dash accordion.
    """
    config = {"paths": {}, "general": {}, "compare": {}, "roc": {}, "output_control": {}}

    # Paths Section
    with st.expander("📁 Paths (required)", expanded=True):
        config["paths"]["base"] = st.text_input(
            "Base directory *",
            value="" if "config_base" not in st.session_state else None,
            placeholder="Absolute path to project root",
            key="config_base",
        )
        config["paths"]["pipeline_runs"] = st.text_input(
            "Pipeline runs directory *",
            value="" if "config_pipeline_runs" not in st.session_state else None,
            placeholder="Directory containing prediction outputs",
            key="config_pipeline_runs",
        )
        config["paths"]["input"] = st.text_input(
            "Input data directory *",
            value="" if "config_input" not in st.session_state else None,
            placeholder="Directory with synergies & drug profiles",
            key="config_input",
        )
        config["paths"]["output"] = st.text_input(
            "Output directory *",
            value="" if "config_output" not in st.session_state else None,
            placeholder="Where synco_output/ will be created",
            key="config_output",
        )
        st.divider()
        config["paths"]["cell_fate_dir"] = st.text_input(
            "Cell fate directory (optional)",
            value="" if "config_cell_fate_dir" not in st.session_state else None,
            placeholder="Parent dir for multi-tissue analysis",
            key="config_cell_fate_dir",
        )
        st.caption("Only needed for Tissue-level visualisations in the Explorer.")

    # General Section
    with st.expander("⚙️ General", expanded=False):
        cell_lines_input = st.text_input(
            "Cell lines",
            value="" if "config_cell_lines" not in st.session_state else None,
            placeholder='Comma-separated list, CSV filename, or blank for auto-discover',
            key="config_cell_lines",
        )
        config["general"]["cell_lines"] = [cl.strip() for cl in cell_lines_input.split(",") if cl.strip()] if cell_lines_input else []

        config["general"]["prediction_method"] = st.selectbox(
            "Prediction method",
            options=["DrugLogics", "BooLEVARD"],
            index=0 if "config_prediction_method" not in st.session_state else (0 if st.session_state.get("config_prediction_method") == "DrugLogics" else 1),
            key="config_prediction_method",
        )

        config["general"]["verbose"] = st.checkbox(
            "Verbose logging",
            value=st.session_state.get("config_verbose", False),
            key="config_verbose",
        )

    # Comparison Section
    with st.expander("📊 Comparison", expanded=False):
        config["compare"]["threshold"] = st.number_input(
            "Synergy threshold",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state.get("config_threshold", 0.5),
            step=0.0001,
            format="%.6f",
            key="config_threshold",
        )

        analysis_mode_current = st.session_state.get("config_analysis_mode", "cell_line")
        analysis_mode_index = 0 if analysis_mode_current == "cell_line" else 1
        config["compare"]["analysis_mode"] = st.selectbox(
            "Analysis mode",
            options=["cell_line", "inhibitor_combination"],
            index=analysis_mode_index,
            format_func=lambda x: {
                "cell_line": "Cell line",
                "inhibitor_combination": "Inhibitor combination",
            }[x],
            key="config_analysis_mode",
        )

        dup_strat_current = st.session_state.get("config_duplicate_strategy", "mean")
        dup_strat_index = 0 if dup_strat_current == "mean" else 1
        config["compare"]["duplicate_strategy"] = st.selectbox(
            "Duplicate strategy",
            options=["mean", "first"],
            index=dup_strat_index,
            format_func=lambda x: {"mean": "Mean", "first": "First"}[x],
            key="config_duplicate_strategy",
        )

    # ROC / Bootstrap Section
    with st.expander("📈 ROC / Bootstrap", expanded=False):
        offsets_input = st.text_input(
            "Threshold offsets",
            value="" if "config_threshold_offsets" not in st.session_state else None,
            placeholder="e.g. -2, -1, 0, 1, 2",
            key="config_threshold_offsets",
        )
        config["roc"]["threshold_offsets"] = [float(x.strip()) for x in offsets_input.split(",") if x.strip()] if offsets_input else []

        col1, col2 = st.columns(2)
        with col1:
            bootstrap_n = st.number_input(
                "Bootstrap samples",
                min_value=0,
                step=10,
                value=st.session_state.get("config_roc_bootstrap_n", 0),
                key="config_roc_bootstrap_n",
            )
            config["roc"]["bootstrap_n"] = int(bootstrap_n) if bootstrap_n > 0 else None

        with col2:
            config["roc"]["bootstrap_ci"] = st.slider(
                "Bootstrap CI level",
                min_value=0.5,
                max_value=0.999,
                value=st.session_state.get("config_roc_bootstrap_ci", 0.95),
                step=0.01,
                key="config_roc_bootstrap_ci",
            )

    # Output Control Section
    with st.expander("💾 Output Control", expanded=False):
        enable_output_control = st.checkbox(
            "Enable selective output writing",
            value=st.session_state.get("config_output_control_enabled", False),
            key="config_output_control_enabled",
        )

        if enable_output_control:
            col1, col2, col3 = st.columns(3)
            with col1:
                config["output_control"]["write_profiles"] = st.checkbox(
                    "Write drug profiles",
                    value=st.session_state.get("config_write_profiles", False),
                    key="config_write_profiles",
                )
                config["output_control"]["write_experimental_full_df"] = st.checkbox(
                    "Write experimental full DF",
                    value=st.session_state.get("config_write_exp_full", False),
                    key="config_write_exp_full",
                )
                config["output_control"]["write_predictions_full_df"] = st.checkbox(
                    "Write predictions full DF",
                    value=st.session_state.get("config_write_pred_full", False),
                    key="config_write_pred_full",
                )

            with col2:
                config["output_control"]["write_synergy_predictions"] = st.checkbox(
                    "Write synergy predictions",
                    value=st.session_state.get("config_write_syn_pred", False),
                    key="config_write_syn_pred",
                )
                config["output_control"]["write_compare_outputs"] = st.checkbox(
                    "Write comparison outputs",
                    value=st.session_state.get("config_write_compare", False),
                    key="config_write_compare",
                )
                config["output_control"]["write_roc_outputs"] = st.checkbox(
                    "Write ROC outputs",
                    value=st.session_state.get("config_write_roc", False),
                    key="config_write_roc",
                )

    # Advanced Section
    with st.expander("🔧 Advanced Overrides", expanded=False):
        # Easy synergy filename override
        synergy_filename = st.text_input(
            "Override synergy filename (optional)",
            value="" if "config_synergy_filename" not in st.session_state else None,
            placeholder="e.g., synergy_data_bliss.csv or path/to/file.csv",
            key="config_synergy_filename",
            help="Specify a different synergy file to use instead of the default pattern",
        )

        st.divider()

        # Raw JSON overrides for advanced users
        advanced_json = st.text_area(
            "Additional JSON overrides (advanced)",
            value="" if "config_advanced_json" not in st.session_state else None,
            placeholder='{\n  "compare": {\n    "duplicate_strategy": "first"\n  }\n}',
            height=100,
            key="config_advanced_json",
        )
        st.caption("Override step-level defaults using nested JSON. Top-level keys: data_loading, compare, roc_analysis, etc.")

        # Build overrides dict
        config["advanced_overrides"] = {}

        # Add synergy filename if specified
        if synergy_filename and synergy_filename.strip():
            config["advanced_overrides"]["data_loading"] = {
                "synergy_filename": synergy_filename.strip()
            }

        # Parse JSON overrides
        if advanced_json and advanced_json.strip():
            try:
                import json
                json_overrides = json.loads(advanced_json)
                # Merge with synergy filename if both are specified
                if "data_loading" in json_overrides:
                    if "data_loading" not in config["advanced_overrides"]:
                        config["advanced_overrides"]["data_loading"] = {}
                    config["advanced_overrides"]["data_loading"].update(json_overrides["data_loading"])
                    del json_overrides["data_loading"]
                # Add remaining overrides
                config["advanced_overrides"].update(json_overrides)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in advanced overrides: {e}")

    return config


def display_config_summary(config: Dict[str, Any]):
    """Display a read-only summary of the current configuration."""
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Paths")
            for key, value in config.get("paths", {}).items():
                if value:
                    st.text(f"**{key}**: {value}")

            st.subheader("General")
            general = config.get("general", {})
            st.text(f"**Method**: {general.get('prediction_method', 'N/A')}")
            st.text(f"**Cell lines**: {len(general.get('cell_lines', []))} selected")
            st.text(f"**Verbose**: {general.get('verbose', False)}")

        with col2:
            st.subheader("Comparison")
            compare = config.get("compare", {})
            st.text(f"**Threshold**: {compare.get('threshold', 'N/A')}")
            st.text(f"**Analysis mode**: {compare.get('analysis_mode', 'N/A')}")
            st.text(f"**Duplicate strategy**: {compare.get('duplicate_strategy', 'N/A')}")

            st.subheader("ROC / Bootstrap")
            roc = config.get("roc", {})
            st.text(f"**Bootstrap samples**: {roc.get('bootstrap_n', 'Disabled')}")
            st.text(f"**CI level**: {roc.get('bootstrap_ci', 'N/A')}")
