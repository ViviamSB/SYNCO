import contextlib
import io
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from synco.features.roc_metrics import calculate_roc_metrics


def _run_roc_metrics(df_exp, df_pred, cell_lines, *, output_path=None, verbose=False):
    return calculate_roc_metrics(
        df_experiment=df_exp,
        df_predictions=df_pred,
        threshold=0.0,
        cell_line_list=cell_lines,
        threshold_offsets=[-1.0, 0.0, 1.0],
        n_bootstrap=0,
        ci_level=0.95,
        verbose=verbose,
        output_path=output_path,
    )


class TestRocMetrics(unittest.TestCase):
    def test_valid_two_class_case(self):
        df_exp = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2'],
                'cell_line': ['CL1', 'CL1'],
                'synergy': [0.5, -0.5],
            }
        )
        df_pred = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2'],
                'CL1': [-0.5, 0.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results'
            output_path.mkdir()
            (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df), skipped = _run_roc_metrics(
                df_exp,
                df_pred,
                ['CL1'],
                output_path=output_path,
            )
            csv_df = pd.read_csv(output_path / 'roc_metrics_df.csv')

        self.assertEqual(skipped, [])
        self.assertEqual(len(traces_roc), 1)
        self.assertEqual(len(traces_pr), 1)
        self.assertEqual(len(rocauc_scores), 1)
        self.assertEqual(len(prauc_scores), 1)

        expected_columns = {
            'cell_line', 'threshold', 'roc_auc', 'pr_auc', 'f1_score', 'mcc', 'balanced_accuracy',
            'roc_auc_ci_low', 'roc_auc_ci_high', 'pr_auc_ci_low', 'pr_auc_ci_high', 'n_positive',
            'n_negative', 'post_filter_positives', 'post_filter_negatives', 'pred_min', 'total_rows',
            'valid_matched_rows', 'missing_predictions', 'invalid_experimental_values', 'invalid_predictions',
            'failure_reason',
        }
        self.assertTrue(expected_columns.issubset(set(metrics_df.columns)))
        self.assertTrue(expected_columns.issubset(set(csv_df.columns)))

        row = metrics_df.iloc[0]
        self.assertEqual(row['roc_auc'], 1.0)
        self.assertEqual(row['pr_auc'], 1.0)
        self.assertEqual(row['f1_score'], 1.0)
        self.assertEqual(row['mcc'], 1.0)
        self.assertEqual(row['balanced_accuracy'], 1.0)
        self.assertEqual(row['n_positive'], 1)
        self.assertEqual(row['n_negative'], 1)
        self.assertEqual(row['post_filter_positives'], 1)
        self.assertEqual(row['post_filter_negatives'], 1)
        self.assertEqual(row['total_rows'], 2)
        self.assertEqual(row['valid_matched_rows'], 2)
        self.assertEqual(row['missing_predictions'], 0)
        self.assertEqual(row['invalid_experimental_values'], 0)
        self.assertEqual(row['invalid_predictions'], 0)
        self.assertEqual(row['failure_reason'], 'success')

    def test_missing_predictions_are_counted_and_skipped(self):
        df_exp = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2', 'P3'],
                'cell_line': ['CL1', 'CL1', 'CL1'],
                'synergy': [0.5, -0.5, -0.25],
            }
        )
        df_pred = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P3'],
                'CL1': [-0.5, 0.5],
            }
        )

        (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df), skipped = _run_roc_metrics(
            df_exp,
            df_pred,
            ['CL1'],
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(traces_roc), 1)
        self.assertEqual(len(traces_pr), 1)
        self.assertEqual(len(rocauc_scores), 1)
        self.assertEqual(len(prauc_scores), 1)

        row = metrics_df.iloc[0]
        self.assertEqual(row['total_rows'], 3)
        self.assertEqual(row['valid_matched_rows'], 2)
        self.assertEqual(row['missing_predictions'], 1)
        self.assertEqual(row['invalid_experimental_values'], 0)
        self.assertEqual(row['invalid_predictions'], 0)
        self.assertEqual(row['failure_reason'], 'success')

    def test_invalid_values_are_skipped_and_counted(self):
        df_exp = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2', 'P3'],
                'cell_line': ['CL1', 'CL1', 'CL1'],
                'synergy': [0.5, np.nan, -0.5],
            }
        )
        df_pred = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2', 'P3'],
                'CL1': [-0.5, 'bad', 0.5],
            }
        )

        (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df), skipped = _run_roc_metrics(
            df_exp,
            df_pred,
            ['CL1'],
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(traces_roc), 1)
        self.assertEqual(len(traces_pr), 1)
        self.assertEqual(len(rocauc_scores), 1)
        self.assertEqual(len(prauc_scores), 1)

        row = metrics_df.iloc[0]
        self.assertEqual(row['total_rows'], 3)
        self.assertEqual(row['valid_matched_rows'], 2)
        self.assertEqual(row['missing_predictions'], 0)
        self.assertEqual(row['invalid_experimental_values'], 1)
        self.assertEqual(row['invalid_predictions'], 1)
        self.assertEqual(row['failure_reason'], 'success')

    def test_single_class_data_remains_in_output_with_nan_metrics(self):
        df_exp = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2'],
                'cell_line': ['CL1', 'CL1'],
                'synergy': [0.5, 0.25],
            }
        )
        df_pred = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2'],
                'CL1': [-0.5, -0.25],
            }
        )

        (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df), skipped = _run_roc_metrics(
            df_exp,
            df_pred,
            ['CL1'],
        )

        self.assertEqual(traces_roc, [])
        self.assertEqual(traces_pr, [])
        self.assertEqual(rocauc_scores, [])
        self.assertEqual(prauc_scores, [])
        self.assertEqual(skipped, [])

        row = metrics_df.iloc[0]
        self.assertTrue(np.isnan(row['roc_auc']))
        self.assertTrue(np.isnan(row['pr_auc']))
        self.assertTrue(np.isnan(row['f1_score']))
        self.assertTrue(np.isnan(row['mcc']))
        self.assertEqual(row['total_rows'], 2)
        self.assertEqual(row['valid_matched_rows'], 2)
        self.assertEqual(row['post_filter_positives'], 2)
        self.assertEqual(row['post_filter_negatives'], 0)
        self.assertEqual(row['failure_reason'], 'single_class')

    def test_duplicate_predictions_use_first_valid_value_and_warn(self):
        df_exp = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P2'],
                'cell_line': ['CL1', 'CL1'],
                'synergy': [0.5, -0.5],
            }
        )
        df_pred = pd.DataFrame(
            {
                'Perturbation': ['P1', 'P1', 'P2'],
                'CL1': [-0.9, 0.8, 0.5],
            }
        )

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            (traces_roc, traces_pr, rocauc_scores, prauc_scores, metrics_df), skipped = _run_roc_metrics(
                df_exp,
                df_pred,
                ['CL1'],
                verbose=True,
            )

        self.assertEqual(skipped, [])
        self.assertEqual(len(traces_roc), 1)
        self.assertEqual(len(traces_pr), 1)
        self.assertEqual(rocauc_scores, [1.0])
        self.assertEqual(prauc_scores, [1.0])
        self.assertIn('multiple distinct valid predictions', captured.getvalue())

        row = metrics_df.iloc[0]
        self.assertEqual(row['roc_auc'], 1.0)
        self.assertEqual(row['pr_auc'], 1.0)
        self.assertEqual(row['valid_matched_rows'], 2)
        self.assertEqual(row['invalid_predictions'], 0)
        self.assertEqual(row['failure_reason'], 'success')


if __name__ == '__main__':
    unittest.main()