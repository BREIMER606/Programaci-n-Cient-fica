import os
import unittest
import numpy as np
import pandas as pd

from Src.analysis import (
    definir_clases_globales,
    extract_imu_acc_features,
    entrenar_random_forest_acc,
)


class TestAnalysisModule(unittest.TestCase):

    def setUp(self):
        # Small synthetic dataset
        n = 300
        self.imu_cols = ["acc_x", "acc_y", "acc_z"]
        self.df = pd.DataFrame({
            "t_pose": np.linspace(0.0, 10.0, n),
            "acc_x": np.random.normal(0.0, 0.01, n),
            "acc_y": np.random.normal(0.0, 0.01, n),
            "acc_z": np.random.normal(1.0, 0.02, n),
        })

        # Output paths for test artifacts
        os.makedirs("results/figures", exist_ok=True)
        os.makedirs("results/tables", exist_ok=True)

    def test_define_classes_creates_label_column(self):
        df_classes = definir_clases_globales(self.df, self.imu_cols)

        # Column exists and has no NaN
        self.assertIn("final_global_mobile_state", df_classes.columns)
        self.assertFalse(df_classes["final_global_mobile_state"].isna().any())

        # Only expected labels appear
        labels = set(df_classes["final_global_mobile_state"].unique())
        self.assertTrue(labels.issubset({"repose", "transition", "motion"}))

    def test_feature_extraction_returns_one_row_per_axis(self):
        df_classes = definir_clases_globales(self.df, self.imu_cols)
        feats = extract_imu_acc_features(df_classes, self.imu_cols)

        # One row per accelerometer axis
        self.assertEqual(len(feats), len(self.imu_cols))
        self.assertListEqual(sorted(feats["variable"].tolist()), sorted(self.imu_cols))

        # Basic numeric fields are finite
        for col in ["mean", "std", "rms"]:
            self.assertTrue(np.isfinite(feats[col]).all())

    def test_random_forest_runs_without_errors(self):
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
    unittest.main()
