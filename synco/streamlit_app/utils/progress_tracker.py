import streamlit as st
from typing import Optional
import time


def display_progress_bar(current_step: int, total_steps: int = 6):
    """Display a progress bar with step indicator."""
    progress = current_step / total_steps
    st.progress(progress, text=f"Step {current_step}/{total_steps}")


def display_execution_metrics(start_time: Optional[float] = None, current_step: int = 0):
    """Display execution time and estimated remaining time."""
    if start_time is None:
        return

    elapsed = time.time() - start_time

    col1, col2, col3 = st.columns(3)

    with col1:
        minutes, seconds = divmod(int(elapsed), 60)
        st.metric("Elapsed Time", f"{minutes}m {seconds}s")

    with col2:
        if current_step > 0:
            avg_step_time = elapsed / current_step
            remaining_steps = 6 - current_step
            estimated_remaining = avg_step_time * remaining_steps
            est_minutes, est_seconds = divmod(int(estimated_remaining), 60)
            st.metric("Est. Remaining", f"{est_minutes}m {est_seconds}s")

    with col3:
        st.metric("Current Step", f"{current_step}/6")


def display_execution_log(log_messages: list, max_height: int = 300):
    """Display execution log messages with color coding."""
    if not log_messages:
        st.info("No messages yet...")
        return

    log_container = st.container(border=True)

    with log_container:
        for msg in log_messages:
            msg_type = msg.get("type", "message")
            text = msg.get("text", "")

            if msg_type == "error":
                st.error(text, icon="❌")
            elif msg_type == "success":
                st.success(text, icon="✅")
            elif msg_type == "warning":
                st.warning(text, icon="⚠️")
            else:
                st.text(f"ℹ️ {text}")

    # Auto-scroll to bottom (Streamlit limitation - would need custom JS for true auto-scroll)
    st.caption(f"{len(log_messages)} log messages")


def display_step_indicator(current_step: int, total_steps: int = 6):
    """Display a visual step indicator showing which steps are complete."""
    steps = [
        "Configuration",
        "Data Loading",
        "Harmonization",
        "Comparison",
        "ROC Analysis",
        "Finalization"
    ]

    cols = st.columns(total_steps)
    for i, col in enumerate(cols):
        with col:
            step_num = i + 1
            if step_num < current_step:
                st.success(f"✅ {steps[i]}")
            elif step_num == current_step:
                st.info(f"🔄 {steps[i]}")
            else:
                st.info(f"⏳ {steps[i]}")


def display_status_message(status: str, error_message: Optional[str] = None):
    """Display status with appropriate styling."""
    if status == "idle":
        st.info("🔵 Ready to run pipeline")
    elif status == "running":
        st.warning("🟡 Pipeline is running...")
    elif status == "completed":
        st.success("🟢 Pipeline completed successfully!")
    elif status == "error":
        st.error(f"🔴 Error: {error_message or 'Unknown error occurred'}")
