"""Streamlit app page: Pipeline Runner

This page executes the SYNCO pipeline with a configured configuration.
"""

import streamlit as st
import time
from synco.streamlit_app.utils.session_manager import (
    initialize_session_state,
    get_config,
    set_artifacts,
    get_artifacts,
    set_running,
    is_running as check_is_running,
)
from synco.streamlit_app.utils.pipeline_executor import StreamlitPipelineExecutor
from synco.streamlit_app.utils.progress_tracker import (
    display_progress_bar,
    display_execution_metrics,
    display_execution_log,
    display_step_indicator,
    display_status_message,
)


def main():
    st.set_page_config(page_title="SYNCO Pipeline Runner", layout="wide")

    # Initialize session state
    initialize_session_state()

    st.title("▶️ SYNCO Pipeline Runner")
    st.markdown("Monitor and control pipeline execution in real-time.")

    # Check if configuration is available
    config = get_config()
    if not config:
        st.warning(
            "⚠️ No configuration found. Please go to the **Setup** page to configure the pipeline first.",
            icon="🔧",
        )
        return

    st.success("✅ Configuration loaded from setup page")

    # Create layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("⚙️ Controls")
        run_button = st.button(
            "▶️ Run Pipeline",
            key="run_pipeline_btn",
            width='stretch',
            disabled=check_is_running(),
        )

    st.divider()

    # Initialize executor if needed
    if "executor" not in st.session_state or st.session_state.executor is None:
        st.session_state.executor = StreamlitPipelineExecutor(config)

    executor = st.session_state.executor

    # Handle run button click
    if run_button:
        st.session_state.executor = StreamlitPipelineExecutor(config)
        executor = st.session_state.executor
        executor.start()
        set_running(True)
        st.session_state.start_time = time.time()

    # Display status
    if executor.status != "idle":
        st.subheader("📊 Execution Status")
        display_status_message(executor.status, executor.error_message)

    # Display progress section
    if executor.is_running() or executor.is_completed() or executor.is_error():
        st.subheader("📈 Progress")

        # Progress bar
        display_progress_bar(executor.current_step, total_steps=6)

        # Step indicator
        display_step_indicator(executor.current_step, total_steps=6)

        # Metrics
        display_execution_metrics(st.session_state.start_time, executor.current_step)

        st.divider()

        # Execution log
        st.subheader("📋 Execution Log")
        log_messages = executor.get_log_messages()
        if log_messages:
            display_execution_log(log_messages)

        # Auto-refresh if running
        if executor.is_running():
            time.sleep(0.5)
            st.rerun()

        # Handle completion
        if executor.is_completed():
            result = executor.get_result()
            if result:
                set_artifacts(result.get("artifacts"))
                st.session_state.execution_completed = True
                set_running(False)
                st.success("✅ Pipeline completed successfully!")

        # Handle error
        if executor.is_error():
            set_running(False)
            st.error(f"❌ Pipeline failed: {executor.error_message}")

    else:
        st.info("👉 Click **Run Pipeline** to start execution")

    # Display results if completed
    if executor.is_completed() and get_artifacts():
        st.divider()
        st.subheader("✅ Results Available")
        st.info("Go to the **Results** page to view detailed results and visualizations.")


if __name__ == "__main__":
    main()
