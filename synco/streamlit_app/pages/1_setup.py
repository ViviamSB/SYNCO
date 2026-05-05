"""Streamlit app page: Setup

This page allows users to configure the SYNCO pipeline parameters.
"""

import streamlit as st
from synco.streamlit_app.utils.session_manager import (
    initialize_session_state,
    set_config,
    get_config,
)
from synco.streamlit_app.utils.config_builder import build_config_form, display_config_summary
from synco.streamlit_app.utils.validators import validate_config


def main():
    st.set_page_config(page_title="SYNCO Setup", layout="wide")

    # Initialize session state
    initialize_session_state()

    st.title("🔧 SYNCO Pipeline Setup")
    st.markdown("Configure your pipeline parameters and input data paths.")

    # Restore previous config if it exists
    if get_config() and "config_base" not in st.session_state:
        prev_config = get_config()
        # Restore paths
        st.session_state.config_base = prev_config.get("paths", {}).get("base", "")
        st.session_state.config_pipeline_runs = prev_config.get("paths", {}).get("pipeline_runs", "")
        st.session_state.config_input = prev_config.get("paths", {}).get("input", "")
        st.session_state.config_output = prev_config.get("paths", {}).get("output", "")
        st.session_state.config_cell_fate_dir = prev_config.get("paths", {}).get("cell_fate_dir", "")
        # Restore general
        st.session_state.config_cell_lines = ",".join(prev_config.get("general", {}).get("cell_lines", []))
        st.session_state.config_prediction_method = prev_config.get("general", {}).get("prediction_method", "DrugLogics")
        st.session_state.config_verbose = prev_config.get("general", {}).get("verbose", False)
        # Restore compare
        st.session_state.config_threshold = prev_config.get("compare", {}).get("threshold", 0.5)
        st.session_state.config_analysis_mode = prev_config.get("compare", {}).get("analysis_mode", "cell_line")
        st.session_state.config_duplicate_strategy = prev_config.get("compare", {}).get("duplicate_strategy", "mean")
        # Restore ROC
        offsets = prev_config.get("roc", {}).get("threshold_offsets", [])
        st.session_state.config_threshold_offsets = ",".join(map(str, offsets)) if offsets else ""
        st.session_state.config_roc_bootstrap_n = prev_config.get("roc", {}).get("bootstrap_n") or 0
        st.session_state.config_roc_bootstrap_ci = prev_config.get("roc", {}).get("bootstrap_ci", 0.95)
        # Restore advanced
        st.session_state.config_synergy_filename = prev_config.get("advanced_overrides", {}).get("data_loading", {}).get("synergy_filename", "")

    # Build configuration form
    config = build_config_form()

    # Create two columns: form and validation
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("✓ Validation")

        # Validate configuration
        is_valid, errors = validate_config(config)

        if is_valid:
            st.success("✅ Configuration is valid!")
        else:
            st.error("❌ Configuration has errors:")
            for error in errors:
                st.write(f"• {error}")

        st.divider()

        # Display config summary
        display_config_summary(config)

    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("💾 Save Configuration", width='stretch'):
            if is_valid:
                set_config(config)
                st.session_state.validated_config = config
                st.success("✅ Configuration saved successfully!")
            else:
                st.error("Please fix the errors above before saving.")

    with col2:
        if st.button("🔄 Reset", width='stretch'):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key.startswith("config_"):
                    del st.session_state[key]
            st.rerun()

    with col3:
        if get_config() and is_valid:
            st.info("✅ Configuration ready! Go to the Pipeline Runner to execute.")
        else:
            st.warning("⚠️ Please configure the pipeline before proceeding.")


if __name__ == "__main__":
    main()
