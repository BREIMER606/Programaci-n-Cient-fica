"""
Main:

Preprocessing → Visualization → Analysis

FIGURES  → results/figures
TABLES   → results/tables

This script orchestrates the full workflow of Laboratory 3:
1. Data loading and preprocessing.
2. Exploratory visualization.
3. Analytical processing: signal integration, segmentation, feature extraction,
   and classification using Random Forest.

All outputs (figures and tables) are saved as PNG files inside results/.
"""

import os
import pandas as pd

# ---- PREPROCESSING ----
from Src.preprocessing import (
    load_data,
    clean_nan,
    select_columns,
    apply_filter,
    save_csv,
    save_filter_figures,
)

# ---- VISUALIZATION ----
from Src.visualization import (
    generate_heatmap,
    generate_boxplot,
    generate_violinplot,
)

# ---- ANALYSIS ----
from Src.analysis import (
    generate_integration_figure,
    save_segment_summary_png,
    define_global_classes,
    save_class_summary_png,
    extract_acc_features,
    save_features_png,
    train_random_forest_acc,
)

# =====================================================
# 1. PATHS (relative)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

raw_path = os.path.join(BASE_DIR, "data", "raw", "const1-trial2-tdoa2.csv")
processed_path = os.path.join(BASE_DIR, "data", "processed", "const1-trial2-tdoa2_clean.csv")

figures_folder = os.path.join(BASE_DIR, "results", "figures")
tables_folder = os.path.join(BASE_DIR, "results", "tables")

# Visualization files
heatmap_path = os.path.join(figures_folder, "heatmap_variables.png")
boxplot_path = os.path.join(figures_folder, "boxplot_variables.png")
violinplot_path = os.path.join(figures_folder, "violinplot_variables.png")

# Analysis FIGURES
integration_path = os.path.join(figures_folder, "integration_acc.png")
rf_cm_path = os.path.join(figures_folder, "rf_confusion_matrix.png")

# Analysis TABLES
raw_table_path = os.path.join(tables_folder, "table_raw.png")
processed_table_path = os.path.join(tables_folder, "table_processed.png")
features_table_path = os.path.join(tables_folder, "table_features.png")
classes_table_path = os.path.join(tables_folder, "table_classes.png")
rf_report_path = os.path.join(tables_folder, "rf_report.png")

# =====================================================
# 2. PREPROCESSING PARAMETERS
# =====================================================
required_columns = ["tdoa_meas"]

columns_of_interest = [
    "t_pose",
    "acc_x", "acc_y", "acc_z",
]

time_column = "t_pose"
columns_to_filter = ["acc_x", "acc_y", "acc_z"]
window_size = 50

# =====================================================
# 3. VISUALIZATION
# =====================================================
visualization_columns = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "pose_x", "pose_y", "pose_z",
    "baro",
]

# =====================================================
# 4. ANALYSIS
# =====================================================
segment_dict = {
    "UWB": ['t_tdoa', 'idA', 'idB', 'tdoa_meas'],
    "IMU_acc": ['acc_x', 'acc_y', 'acc_z'],
    "IMU_gyro": ['gyro_x', 'gyro_y', 'gyro_z'],
    "Pose": ['pose_x', 'pose_y', 'pose_z'],
}

acc_columns = ['acc_x', 'acc_y', 'acc_z']
filtered_acc_columns = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt']
max_samples_integration = 1000

# =====================================================
# 5. PIPELINE
# =====================================================
def main():
    """
    Executes the complete pipeline of Laboratory 3.

    This includes:
    ---------------------
    1. PREPROCESSING
       - Load raw data.
       - Remove NaNs from required columns.
       - Select relevant sensor channels.
       - Apply smoothing/denoising filters.
       - Save cleaned CSV and filtering figures.

    2. VISUALIZATION
       - Generate heatmap, boxplot, and violinplot for exploratory analysis.

    3. ANALYSIS
       - Integrate acceleration signals to obtain velocity estimates.
       - Compute segment-wise data availability tables (raw and processed).
       - Define global motion classes based on acceleration percentiles.
       - Extract time-domain and frequency-domain features from the IMU.
       - Train and evaluate a Random Forest classifier using ACC signals.

    Saves:
        Figures → results/figures/
        Tables  → results/tables/

    Returns:
        None
    """

    # -------------------- PREPROCESS --------------------
    df = load_data(raw_path)
    df = clean_nan(df, required_columns)
    df_reduced = select_columns(df, columns_of_interest)

    filter_results = apply_filter(
        df=df_reduced,
        columns=columns_to_filter,
        time_column=time_column,
        window_size=window_size,
    )

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    save_csv(df_reduced, processed_path)
    save_filter_figures(filter_results, figures_folder)

    # -------------------- VISUALIZATION --------------------
    df_clean = pd.read_csv(processed_path)
    valid_cols = [c for c in visualization_columns if c in df_clean.columns]

    if valid_cols:
        generate_heatmap(df_clean, valid_cols, heatmap_path)
        generate_boxplot(df_clean, valid_cols, boxplot_path)
        generate_violinplot(df_clean, valid_cols, violinplot_path)

    # -------------------- ANALYSIS --------------------

    # 1) Integration Figure
    acc_cols_integration = []
    for orig, filt in zip(acc_columns, filtered_acc_columns):
        acc_cols_integration.append(filt if filt in df_clean.columns else orig)

    generate_integration_figure(
        df=df_clean,
        time_column=time_column,
        acc_columns=acc_cols_integration,
        max_samples=max_samples_integration,
        figure_path=integration_path,
    )

    # 2) Raw table
    df_raw_full = pd.read_csv(raw_path)
    save_segment_summary_png(df_raw_full, segment_dict, raw_table_path)

    # 3) Processed table
    save_segment_summary_png(df_clean, segment_dict, processed_table_path)

    # 4) Define classes + table
    df_classes = define_global_classes(df_clean, acc_columns)
    save_class_summary_png(df_classes, classes_table_path)

    # 5) ACC features + table
    df_features = extract_acc_features(df_classes, acc_columns)
    save_features_png(df_features, features_table_path)

    # 6) Random Forest (ACC)
    train_random_forest_acc(
        df=df_classes,
        feature_cols=acc_columns,
        label_col="final_global_mobile_state",
        table_output_path=rf_report_path,
        cm_output_path=rf_cm_path,
    )

    print("✔ Full pipeline completed.")


if __name__ == "__main__":
    main()
