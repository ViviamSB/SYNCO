import unittest
import numpy as np
import pandas as pd

from synco.features.roc_metrics import calculate_roc_metrics


def _make_dummy_frames():
    # Two perturbations, two cell lines (experimental already long format as pipeline provides)
    df_exp = pd.DataFrame({
        'Perturbation': ['P1', 'P1', 'P2', 'P2'],
        'cell_line': ['CL1', 'CL2', 'CL1', 'CL2'],
        'synergy': [0.5, 0.5, -0.5, 0.5],  # >0 is synergy
    })
    df_pred = pd.DataFrame({
        'Perturbation': ['P1', 'P2'],
        'CL1': [-0.5, 0.5],  # <0 is synergy in predictions
        'CL2': [-0.5, -0.5],
    })
    return df_exp, df_pred


class TestRocMetrics(unittest.TestCase):
    def test_calculate_roc_metrics_basic(self):
        df_exp, df_pred = _make_dummy_frames()
        cell_lines = ['CL1', 'CL2']
        (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df), skipped = calculate_roc_metrics(
            df_experiment=df_exp,
            df_predictions=df_pred,
            cell_line_list=cell_lines,
            threshold=0.0,
            threshold_offsets=[-1.0, 0.0, 1.0],
            n_bootstrap=0,
            ci_level=0.95,
            verbose=False,
            output_path=None,
        )

        # No cell lines should be skipped
        self.assertEqual(skipped, [])

        # CL2 is single-class → only one set of traces/scores (CL1)
        self.assertEqual(len(traces_roc), 1)
        self.assertEqual(len(traces_pr), 1)
        self.assertEqual(len(rocauc_scores), 1)
        self.assertEqual(len(prauc_scores), 1)

        # Metrics DataFrame should have expected columns and two rows
        expected_cols = {
            'cell_line', 'threshold', 'roc_auc', 'pr_auc', 'f1_score',
            'mcc', 'balanced_accuracy', 'roc_auc_ci_low', 'roc_auc_ci_high',
            'pr_auc_ci_low', 'pr_auc_ci_high', 'n_positive', 'n_negative', 'pred_min'
        }
        self.assertTrue(expected_cols.issubset(set(metrics_df.columns)))
        self.assertEqual(len(metrics_df), 2)

        # For CL1: perfect separation -> AUCs and F1 = 1.0, MCC/Balanced acc = 1.0
        cl1 = metrics_df[metrics_df['cell_line'] == 'CL1'].iloc[0]
        self.assertEqual(cl1['roc_auc'], 1.0)
        self.assertEqual(cl1['pr_auc'], 1.0)
        self.assertEqual(cl1['f1_score'], 1.0)
        self.assertEqual(cl1['mcc'], 1.0)
        self.assertEqual(cl1['balanced_accuracy'], 1.0)
        self.assertEqual(cl1['n_positive'], 1)
        self.assertEqual(cl1['n_negative'], 1)
        self.assertEqual(cl1['pred_min'], -0.5)

        # CIs are disabled when n_bootstrap=0
        self.assertTrue(np.isnan(cl1['roc_auc_ci_low']))
        self.assertTrue(np.isnan(cl1['pr_auc_ci_low']))

        # For CL2: all positives in truth, all predictions synergistic
        cl2 = metrics_df[metrics_df['cell_line'] == 'CL2'].iloc[0]
        # Since there is only one class in truth, metrics should be NaN
        self.assertTrue(np.isnan(cl2['roc_auc']))
        self.assertTrue(np.isnan(cl2['pr_auc']))
        self.assertTrue(np.isnan(cl2['f1_score']))
        self.assertTrue(np.isnan(cl2['mcc']))
        self.assertTrue(np.isnan(cl2['balanced_accuracy']))
        self.assertTrue(np.isnan(cl2['n_positive']))
        self.assertTrue(np.isnan(cl2['n_negative']))


if __name__ == "__main__":
    unittest.main()
