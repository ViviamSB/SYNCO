# SYNCO Streamlit Dashboard

A modern, interactive web-based dashboard for the SYNCO pipeline using Streamlit.

## 🚀 Quick Start

### Installation

Make sure you have the SYNCO package installed:

```bash
# From the SYNCO repository root
pip install -e .
```

This installs all dependencies including Streamlit.

### Running the App

Choose one of these methods:

**Method 1: Using Python module execution**
```bash
python -m synco.streamlit_app
```

**Method 2: Using streamlit directly**
```bash
streamlit run synco/streamlit_app/app.py
```

**Method 3: From anywhere with the installed package**
```bash
# After installation, you can run from any directory
streamlit run /path/to/synco/streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Pages Overview

### 1. Home Page (Main Dashboard)
- Welcome and project overview
- Quick navigation buttons
- Current pipeline status
- Links to all features

### 2. Setup Page (📋)
Configure your pipeline parameters:
- **Paths**: Input directories, output location
- **General**: Cell lines, prediction method, logging
- **Comparison**: Threshold, analysis mode, duplicate strategy
- **ROC/Bootstrap**: Bootstrap configuration for statistical analysis
- **Output Control**: Selective output file generation
- **Advanced**: JSON overrides for fine-tuning

Features:
- Real-time validation
- Configuration summary display
- Save/Reset functionality

### 3. Pipeline Runner Page (▶️)
Monitor pipeline execution:
- Start/stop controls
- Real-time progress tracking
- Step-by-step execution logs
- Performance metrics (elapsed time, estimated remaining)
- Status indicators for each pipeline stage

Features:
- Non-blocking execution (responsive UI during long operations)
- Live log streaming with color-coded messages
- Automatic refresh during execution
- Completion notifications

### 4. Results Page (📊)
Comprehensive results visualization:

**Tabs:**
- **Summary**: Key metrics and overview
- **ROC Curve**: Receiver Operating Characteristic curve with AUC
- **PR Curve**: Precision-Recall curve
- **Distributions**: Prediction and ground truth distributions
- **Confusion Matrix**: Heatmap visualization
- **Details**: Full results table with sorting
- **Execution**: Pipeline runtime statistics
- **Downloads**: Export result files

## 🏗️ Architecture

### Directory Structure
```
synco/streamlit_app/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m synco.streamlit_app`
├── app.py                   # Main app (home page)
├── pages/
│   ├── __init__.py
│   ├── 1_setup.py          # Configuration page
│   ├── 2_pipeline_runner.py # Execution page
│   └── 3_results.py         # Results page
├── components/
│   └── __init__.py          # (For future custom components)
└── utils/
    ├── __init__.py
    ├── session_manager.py   # Streamlit session state management
    ├── validators.py        # Configuration validation
    ├── pipeline_executor.py # Non-blocking pipeline execution with threading
    ├── config_builder.py    # Configuration form UI builder
    ├── progress_tracker.py  # Progress display components
    └── results_displayer.py # Results visualization components
```

### Key Modules

#### `session_manager.py`
Manages Streamlit session state across pages:
- Configuration storage
- Artifact/results storage
- Execution state tracking
- Logging management

#### `validators.py`
Validates pipeline configuration:
- Path existence checks
- Cell line validation
- Parameter range validation
- Comprehensive error reporting

#### `pipeline_executor.py`
Non-blocking pipeline execution:
- Background thread execution
- Queue-based logging
- Status tracking (idle/running/completed/error)
- Elapsed time tracking

#### `config_builder.py`
Streamlit UI for configuration:
- Expander-based form sections
- Type-specific input widgets
- Session state persistence
- Configuration summary display

#### `progress_tracker.py`
Real-time progress visualization:
- Progress bars
- Step indicators
- Elapsed time metrics
- Status messages

#### `results_displayer.py`
Results visualization components:
- Summary metrics
- ROC/PR curves (Plotly)
- Distribution plots
- Confusion matrix heatmaps
- Data tables
- File downloads

## 💾 Session State Management

The app uses Streamlit's session state to maintain data across page navigation:

```python
st.session_state.config          # User-provided configuration
st.session_state.artifacts       # Pipeline results
st.session_state.executor        # Current pipeline executor
st.session_state.execution_log   # Log messages
st.session_state.is_running      # Execution status
st.session_state.error_message   # Error information
st.session_state.start_time      # Pipeline start timestamp
```

## 🔄 Data Flow

1. **User configures pipeline** (Setup page)
   - Fills configuration form
   - Validation occurs in real-time
   - Configuration saved to session state

2. **User starts execution** (Pipeline Runner page)
   - Pipeline executor created in background thread
   - Progress tracked in real-time
   - Logs streamed to UI
   - Results stored to session state on completion

3. **User views results** (Results page)
   - Retrieves artifacts from session state
   - Displays multiple visualizations
   - Enables file downloads

## 🎨 UI Components

The app uses:
- **Streamlit Core**: Layout, forms, state management
- **Streamlit Expanders**: Collapsible configuration sections
- **Plotly**: Interactive charts and visualizations
- **Pandas**: Data display and manipulation
- **Custom Display Functions**: Domain-specific visualizations

## 🚦 Error Handling

- Path validation before execution
- Configuration validation with detailed error messages
- Pipeline execution error capture and display
- Graceful handling of missing data/artifacts

## 📊 Example Configuration

```json
{
  "paths": {
    "base": "/path/to/project",
    "pipeline_runs": "/path/to/predictions",
    "input": "/path/to/data",
    "output": "/path/to/results"
  },
  "general": {
    "cell_lines": ["A549", "MCF7"],
    "prediction_method": "DrugLogics",
    "verbose": true
  },
  "compare": {
    "threshold": 0.5,
    "analysis_mode": "cell_line",
    "duplicate_strategy": "mean"
  },
  "roc": {
    "bootstrap_n": 500,
    "bootstrap_ci": 0.95
  }
}
```

## 🔧 Development

### Adding New Pages

1. Create a new file in `pages/` directory with prefix number (e.g., `4_analysis.py`)
2. Streamlit automatically discovers and adds it to navigation
3. Use `st.switch_page()` to navigate between pages

### Adding New Components

1. Create display functions in `utils/` or `components/`
2. Import in pages as needed
3. Follow existing naming patterns

### Testing

```bash
# Validate app structure
python test_streamlit_app.py

# Run the app
python -m synco.streamlit_app

# Or with custom port
streamlit run synco/streamlit_app/app.py --server.port 8888
```

## 📈 Performance Considerations

- Pipeline execution runs in background thread to keep UI responsive
- Large result tables paginated for performance
- Plotly charts automatically optimize for data size
- Session state persists across page navigation

## 🐛 Troubleshooting

**App won't start**
- Ensure streamlit is installed: `pip install streamlit`
- Check Python version: 3.8+

**Imports fail**
- Install package in editable mode: `pip install -e .`
- Verify synco package is accessible

**Configuration validation fails**
- Check all paths exist and are readable
- Ensure cell lines are properly formatted
- Verify numeric parameters are in valid ranges

**Pipeline execution hangs**
- Check if background process is stuck
- Verify no other instances are running
- Restart the app if needed

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Multi-page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- [Plotly Python](https://plotly.com/python/)
- [SYNCO Documentation](../README.md)

## 📧 Support

For issues or suggestions, please contact:
- Viviam Bermudez at viviam.bermudez@ntnu.no
- Or viviambermudez@gmail.com
