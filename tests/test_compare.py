import pandas as pd

from synco.features.compare import compare_synergies


def test_missing_predictions_are_ignored():
    """Ensure entries missing from the predictions are not counted as negatives.

    Scenario:
    - Experimental has 2 combinations (combA, combB) for two cell lines (CL1, CL2) -> 4 entries
    - Predictions have both entries for CL1 but only one entry for CL2 (combB missing)
    Expectation:
    - For CL1: Total comparisons == 2
    - For CL2: Total comparisons == 1 (missing pair ignored)
    - For CL2: True Positive == 1, False Negative == 0, False Positive == 0
    """

    df_exp = pd.DataFrame([
        {"inhibitor_combination": "combA", "cell_line": "CL1", "synergy": 1},
        {"inhibitor_combination": "combB", "cell_line": "CL1", "synergy": 0},
        {"inhibitor_combination": "combA", "cell_line": "CL2", "synergy": 1},
        {"inhibitor_combination": "combB", "cell_line": "CL2", "synergy": 1},
    ])

    df_pred = pd.DataFrame([
        {"inhibitor_combination": "combA", "cell_line": "CL1", "synergy": 1},
        {"inhibitor_combination": "combB", "cell_line": "CL1", "synergy": 0},
        {"inhibitor_combination": "combA", "cell_line": "CL2", "synergy": 1},
        # combB for CL2 is intentionally missing
    ])

    results_df, skipped_info, summary = compare_synergies(
        df_exp, df_pred, cell_line_list=["CL1", "CL2"], output_path=None
    )

    # Results are per cell line (analysis_mode default 'cell_line')
    # CL1 has two valid comparisons
    assert results_df.loc["CL1", "Total"] == 2
    # CL2 has only one valid comparison because combB prediction is missing
    assert results_df.loc["CL2", "Total"] == 1

    # CL2: combA is a true positive
    assert int(results_df.loc["CL2", "True Positive"]) == 1
    assert int(results_df.loc["CL2", "False Negative"]) == 0
    assert int(results_df.loc["CL2", "False Positive"]) == 0
