import streamlit as st
from typing import Optional, Any


def initialize_session_state():
    """Initialize all required session state variables."""
    defaults = {
        'config': None,
        'executor': None,
        'artifacts': None,
        'execution_log': [],
        'current_step': 0,
        'total_steps': 6,
        'is_running': False,
        'error_message': None,
        'start_time': None,
        'uploaded_files': {},
        'validated_config': None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def set_config(config: dict):
    """Store configuration in session state."""
    st.session_state.config = config
    st.session_state.validated_config = None


def get_config() -> Optional[dict]:
    """Retrieve configuration from session state."""
    return st.session_state.config


def set_artifacts(artifacts: dict):
    """Store pipeline artifacts in session state."""
    st.session_state.artifacts = artifacts


def get_artifacts() -> Optional[dict]:
    """Retrieve pipeline artifacts from session state."""
    return st.session_state.artifacts


def add_log_message(message: str):
    """Add a message to the execution log."""
    st.session_state.execution_log.append(message)


def clear_log():
    """Clear the execution log."""
    st.session_state.execution_log = []


def get_log() -> list:
    """Get all log messages."""
    return st.session_state.execution_log


def set_running(is_running: bool):
    """Set the running state."""
    st.session_state.is_running = is_running


def is_running() -> bool:
    """Check if pipeline is running."""
    return st.session_state.is_running


def set_current_step(step: int):
    """Set the current pipeline step (1-6)."""
    st.session_state.current_step = step


def get_current_step() -> int:
    """Get the current pipeline step."""
    return st.session_state.current_step


def get_progress_percentage() -> float:
    """Get progress as percentage (0-100)."""
    return (st.session_state.current_step / st.session_state.total_steps) * 100


def set_error(error_message: Optional[str]):
    """Set error message."""
    st.session_state.error_message = error_message


def get_error() -> Optional[str]:
    """Get error message."""
    return st.session_state.error_message


def set_start_time(timestamp: float):
    """Set pipeline start time."""
    st.session_state.start_time = timestamp


def get_start_time() -> Optional[float]:
    """Get pipeline start time."""
    return st.session_state.start_time


def clear_execution_state():
    """Clear all execution-related state (for new run)."""
    st.session_state.execution_log = []
    st.session_state.current_step = 0
    st.session_state.error_message = None
    st.session_state.start_time = None
    st.session_state.artifacts = None
