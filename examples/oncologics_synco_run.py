from synco.main import run_pipeline
import json

# Load or create config
config = {
    "paths": {
        "base": "examples/oncologics",
        "pipeline_runs": "examples/oncologics/20251013/drabme_out",
        "input": "examples/oncologics/20251013/synco_input",
        "output": "examples/oncologics/20251013/synco_output"
    },
    "general": {
        "cell_lines": None,  # Auto-discover from pipeline_runs directory
        "run_date": None,
        "verbose": True
    },
    "compare": {
        "prediction_method": "DrugLogics",
        "threshold": 0.00001,
        "threshold_offsets": [-2.0, -1.0, 0.0, 1.0, 2.0],  # NEW: threshold sweep
        "roc_bootstrap_n": 500,  # NEW: enable bootstrap CIs (or None to disable)
        "roc_bootstrap_ci": 0.95,  # NEW: CI level
        "synergy_column": "synergy",
        "analysis_mode": "cell_line",
        "duplicate_strategy": "mean"
    },
    "advance": {
        "data_loading": {
            "synergy_filename": "synergies_observed_hsa.csv"
        }
    }
}

# Run pipeline
run_pipeline(config, verbose=True)