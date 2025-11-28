import os
import unittest
import numpy as np
import pandas as pd

# Import directly from the analysis module file
from Src.analysis.analysis import (
    definir_clases_globales,
    extract_imu_acc_features,
    entrenar_random_forest_acc,
)


class TestAnalysisModule(unittest.TestCase):
    """
    Unit tests for a subset of the analysis functions.

    Each test runs on a small synthetic dataset of size N = 300,
    so the effective cost per test is O(N) or O(N log N) depending
    on the underlying algorithm (class definition, FFT features,
    Random Forest training).[web:633]
    """

    def setUp(self):
        """
        Create a small synthetic accelerometer dataset and ensure
        that the output folders for test artifacts exist.

        Complexity: O(N), where N is the number of generated samples.
        """
        n = 300
        self.imu_cols = ["acc_x", "acc_y", "acc_z"]
        self.df = pd.DataFrame({
            "t_pose": np.linspace(0.0, 10.0, n),           # O(N)
            "acc_x": np.random.normal(0.0, 0.01, n),       # O(N)
            "acc_y": np.random.normal(0.0, 0.01, n),       # O(N)
            "acc_z": np.random.normal(1.0, 0.02, n),       # O(N)
        })

        # Output paths for test artifacts (O(1))
        os.makedirs("results/figures", exist_ok=True)
        os.makedirs("results/tables", exist_ok=True)

    def test_define_classes_creates_label_column(self):
        """
        Test that 'definir_clases_globales' creates the label column
        'final_global_mobile_state' without NaNs and only with the
        expected labels: 'repose', 'transition', 'motion'.

        Complexity: dominated by the class-definition function,
        which runs in O(N * K) with K = 3 accelerometer axes.
        """
        df_classes = definir_clases_globales(self.df, self.imu_cols)

        # Column exists and has no NaN
        self.assertIn("final_global_mobile_state", df_classes.columns)
        self.assertFalse(df_classes["final_global_mobile_state"].isna().any())

        # Only expected labels appear
        labels = set(df_classes["final_global_mobile_state"].unique())
        self.assertTrue(labels.issubset({"repose", "transition", "motion"}))

    def test_feature_extraction_returns_one_row_per_axis(self):
        """
        Test that 'extract_imu_acc_features' returns one row per
        accelerometer axis and that basic numeric fields are finite.

        Complexity: O(N * K log N), dominated by the FFT per axis
        inside the feature-extraction routine.
        """
        df_classes = definir_clases_globales(self.df, self.imu_cols)
        feats = extract_imu_acc_features(df_classes, self.imu_cols)

        # One row per accelerometer axis
        self.assertEqual(len(feats), len(self.imu_cols))
        self.assertListEqual(
            sorted(feats["variable"].tolist()),
            sorted(self.imu_cols),
        )

        # Basic numeric fields are finite
        for col in ["mean", "std", "rms"]:
            self.assertTrue(np.isfinite(feats[col]).all())

    def test_random_forest_runs_without_errors(self):
        """
        Test that 'entrenar_random_forest_acc' can train a Random
        Forest on the synthetic data, returns non-empty outputs,
        and saves both the classification report and confusion
        matrix as PNG files.

        Complexity: roughly O(T * N log N), where T is the number of
        trees in the forest and N is the number of samples used for
        training and testing.
        """
        df_classes = definir_clases_globales(self.df, self.imu_cols)

        report_path = "results/tables/test_rf_report.png"
        cm_path = "results/figures/test_rf_cm.png"

        df_rep, cm, labels = entrenar_random_forest_acc(
            df=df_classes,
            feature_cols=self.imu_cols,
            label_col="final_global_mobile_state",
            ruta_tabla_reporte=report_path,
            ruta_figura_cm=cm_path,
            test_size=0.2,
            random_state=0,
        )

        # Model produced outputs
        self.assertGreater(len(df_rep), 0)
        self.assertIsNotNone(cm)
        self.assertEqual(cm.shape[0], len(labels))

        # Artifacts were saved
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(os.path.exists(cm_path))


if __name__ == "__main__":
    # Running through unittest.main keeps the total cost proportional
    # to the sum of the individual test costs, here dominated by the
    # Random Forest training: O(T * N log N) for N = 300.
    unittest.main(verbosity=2)
