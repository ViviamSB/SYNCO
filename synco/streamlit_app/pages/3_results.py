"""Streamlit app page: Results

This page displays results and visualizations from the completed pipeline.
"""

import streamlit as st
from synco.streamlit_app.utils.session_manager import (
    initialize_session_state,
    get_artifacts,
)
from synco.streamlit_app.utils.results_displayer import (
    display_results_summary,
    display_pair_details,
    display_skipped_info,
    display_synergy_predictions,
    display_convergence_results,
    display_roc_results,
    display_drug_profiles,
    display_synergy_data,
    display_execution_summary,
    display_artifacts_overview,
    display_artifacts_debug,
)


def main():
    st.set_page_config(page_title="SYNCO Results", layout="wide")

    # Initialize session state
    initialize_session_state()

    st.title("📊 SYNCO Pipeline Results")
    st.markdown("View results from your pipeline execution.")

    # Get artifacts
    artifacts = get_artifacts()

    if not artifacts:
        st.warning(
            "⚠️ No results available yet. Please run the pipeline from the **Pipeline Runner** page.",
            icon="🚀",
        )
        return

    st.success("✅ Results loaded successfully")

    # Create tabs for different result sections
    tabs = st.tabs([
        "📋 Summary",
        "📈 ROC Analysis",
        "🔄 Convergence",
        "📊 Pair Details",
        "💊 Data Overview",
        "⚠️ Skipped",
        "📦 All Data",
        "🔧 Debug",
    ])

    with tabs[0]:  # Summary tab
        st.header("Results Summary")
        display_results_summary(artifacts)
        st.divider()
        display_execution_summary(artifacts)

    with tabs[1]:  # ROC Analysis tab
        st.header("ROC Analysis")
        display_roc_results(artifacts)

    with tabs[2]:  # Convergence tab
        st.header("Convergence Analysis")
        display_convergence_results(artifacts)

    with tabs[3]:  # Pair Details tab
        st.header("Pair Comparison Details")
        display_pair_details(artifacts)

    with tabs[4]:  # Data Overview tab
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            display_drug_profiles(artifacts)
        with col2:
            display_synergy_data(artifacts)

    with tabs[5]:  # Skipped tab
        st.header("Skipped Entries")
        display_skipped_info(artifacts)

    with tabs[6]:  # All Data tab
        st.header("Available Data")
        display_artifacts_overview(artifacts)

    with tabs[7]:  # Debug tab
        st.header("Debug Information")
        display_artifacts_debug(artifacts)

    # Footer with info
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Run Another Analysis", width='stretch'):
            st.switch_page("pages/2_pipeline_runner.py")

    with col2:
        if st.button("⚙️ Modify Configuration", width='stretch'):
            st.switch_page("pages/1_setup.py")

    with col3:
        st.info("💡 **Tip**: Review the convergence and ROC analysis for comprehensive insights.")


if __name__ == "__main__":
    main()
