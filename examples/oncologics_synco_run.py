from synco.main import run_pipeline
from pathlib import Path

def main() -> None:
    base_dir = Path("examples/oncologics")
    pipeline_runs = base_dir / "11panel_bliss/drabme_out"
    input_dir = base_dir / "11panel_bliss/synco_input"
    output_root = base_dir / "11panel_bliss/synco_output"

    analysis_modes = ["cell_line", "inhibitor_combination"]

    for analysis_mode in analysis_modes:
        output_dir = output_root
        shared_output = base_dir / "11panel_bliss/synco_shared"
        shared_output.mkdir(parents=True, exist_ok=True)

        config = {
            "paths": {
                "base": str(base_dir),
                "pipeline_runs": str(pipeline_runs),
                "input": str(input_dir),
                "output": str(output_dir),
            },
            "general": {
                "cell_lines": None,  # Auto-discover from pipeline_runs directory
                "run_date": None,
                "verbose": True,
            },
            "compare": {
                "prediction_method": "DrugLogics",
                "threshold": 0.001,
                "threshold_offsets": [-2.0, -1.0, 0.0, 1.0, 2.0],  # NEW: threshold sweep
                "roc_bootstrap_n": 500,  # NEW: enable bootstrap CIs (or None to disable)
                "roc_bootstrap_ci": 0.95,  # NEW: CI level
                "synergy_column": "synergy",
                "analysis_mode": analysis_mode,
                "duplicate_strategy": "mean",
            },
            "advance": {
                "data_loading": {
                    "synergy_filename": "synergies_observed_bliss.csv"
                }
            },
            "output_control": {
                "enabled": True,
                "shared_output": str(shared_output),
                "write_profiles": True,
                "write_experimental_full_df": True,
                "write_predictions_full_df": True,
                "write_synergy_predictions": True,
                "write_compare_outputs": True,
                "write_roc_outputs": True,
            },
        }

        print(f"\nRunning Oncologics | analysis_mode: {analysis_mode}")
        run_pipeline(config, verbose=True)

if __name__ == "__main__":
    main()