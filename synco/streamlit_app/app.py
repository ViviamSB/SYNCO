"""SYNCO Streamlit App - Main Entry Point

This is the main Streamlit application for the SYNCO pipeline.
Run with: streamlit run synco/streamlit_app/app.py
or: python -m synco.streamlit_app
"""

import streamlit as st
from synco.streamlit_app.utils.session_manager import initialize_session_state


def main():
    st.set_page_config(
        page_title="SYNCO Pipeline",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🧬 SYNCO")
        st.markdown("SYNergy COnvergency Analysis")
        st.divider()

        st.markdown("""
### Navigation

Use the sidebar to navigate between pages:
- **Setup** — Configure pipeline parameters
- **Pipeline Runner** — Execute the pipeline
- **Results** — View results and visualizations

### About SYNCO

SYNCO is a Python package for the analysis of synergistic drug responses,
predicted using the DrugLogics software.
        """)

        st.divider()

        st.markdown("""
### Features
- Extract and unify predictions and experimental observations
- Harmonise drug profiles
- Calculate performance metrics
- Generate ROC and PR curves
- Interactive visualization
        """)

    # Main content
    st.title("🧬 Welcome to SYNCO Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
## 🚀 Getting Started

The SYNCO pipeline helps you analyze synergistic drug responses by comparing
predictions with experimental observations.

### Quick Steps:

1. **Setup** (📋) - Configure your data paths and analysis parameters
2. **Run** (▶️) - Execute the pipeline with real-time progress monitoring
3. **Results** (📊) - Explore comprehensive visualizations and metrics
        """)

    with col2:
        st.markdown("""
## 📊 Pipeline Highlights

- **Multi-tissue analysis** across cell lines
- **Real-time classification** tables
- **Performance metrics**: sensitivity, specificity, accuracy
- **ROC/PR curve** visualization
- **Distribution analysis** of predictions
- **Drug profile comparisons**
- **Cross-tissue synchronization** analysis
        """)

    st.divider()

    # Current status
    st.subheader("📈 Current Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.config:
            st.success("✅ Configuration Loaded")
        else:
            st.warning("⚠️ No Configuration")

    with col2:
        if st.session_state.artifacts:
            st.success("✅ Results Available")
        else:
            st.info("ℹ️ No Results Yet")

    with col3:
        if st.session_state.is_running:
            st.warning("🟡 Pipeline Running...")
        else:
            st.info("🔵 Ready")

    st.divider()

    # Quick start buttons
    st.subheader("🎯 Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("1️⃣ Go to Setup", use_container_width=True, key="btn_setup"):
            st.switch_page("pages/1_setup.py")

    with col2:
        if st.button("2️⃣ Go to Pipeline Runner", use_container_width=True, key="btn_runner"):
            if st.session_state.config:
                st.switch_page("pages/2_pipeline_runner.py")
            else:
                st.warning("Please configure the pipeline first!")

    with col3:
        if st.button("3️⃣ Go to Results", use_container_width=True, key="btn_results"):
            if st.session_state.artifacts:
                st.switch_page("pages/3_results.py")
            else:
                st.warning("Please run the pipeline first!")

    st.divider()

    # Information
    st.markdown("""
### 📚 Documentation

For detailed usage instructions and examples, please refer to the
[QUICKSTART.md](https://github.com/ViviamSB/SYNCO/blob/main/QUICKSTART.md) guide.

### 📧 Support

For questions or feedback, contact:
- Viviam Bermudez at viviam.bermudez@ntnu.no
- Or viviambermudez@gmail.com
    """)


if __name__ == "__main__":
    main()
