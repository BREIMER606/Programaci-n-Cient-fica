"""
analysis:

Analysis module for Laboratory 3.

This module contains reusable functions for:
- Acceleration integration (acc → vel)
- Segment statistics (saved as PNG table)
- Class definition using acceleration magnitude percentiles (PNG table)
- Accelerometer feature extraction (PNG table)
- Random Forest classification using only accelerometer features (PNG table + PNG figure)

IMPORTANT:
- All TABLES must be saved inside results/tables (path provided by main).
- All FIGURES must be saved inside results/figures.
- No print statements and no absolute paths.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
#   UTILITY: Save DataFrame as PNG TABLE
# ============================================================
def df_to_png(df: pd.DataFrame, png_path: str, title: str = None, fontsize: int = 10):
    """
    Save a DataFrame as a PNG table using matplotlib.

    Args:
        df (pd.DataFrame): DataFrame to render as a table.
        png_path (str): Output file path where the PNG will be saved.
        title (str, optional): Optional title displayed above the table.
        fontsize (int): Font size used within the table cells.

    Notes:
        - The directory for png_path is created automatically if it does not exist.
        - No value is returned; the image file is saved directly.
    """
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    rows, cols = df.shape
    height = max(1.5, 0.3 * rows + 1.5)
    width = max(4, 0.8 * cols + 2)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#   1) INTEGRATION acc → vel (FIGURE)
# ============================================================
class IntegratedSignal:
    """
    Class for numerical integration of an acceleration signal using
    the trapezoidal rule to approximate velocity.

    Attributes:
        time (np.ndarray): Time vector.
        values (np.ndarray): Acceleration samples.
        name (str): Optional name for the signal.
    """

    def __init__(self, time: np.ndarray, values: np.ndarray, name: str = ""):
        """
        Initialize the IntegratedSignal object.

        Args:
            time (np.ndarray): Time stamps for the samples.
            values (np.ndarray): Acceleration values.
            name (str): Optional signal name.
        """
        self.time = np.asarray(time)
        self.values = np.asarray(values)
        self.name = name

    def evaluate(self):
        """
        Compute integrated velocity using trapezoidal integration.

        Returns:
            tuple: (t, a, v)
                t (np.ndarray): Clean time vector.
                a (np.ndarray): Clean acceleration vector.
                v (np.ndarray): Integrated velocity vector.
        """
        mask = np.isfinite(self.time) & np.isfinite(self.values)
        t = self.time[mask]
        a = self.values[mask]

        v = np.zeros_like(a)
        if len(t) > 1:
            v[1:] = np.cumsum((a[:-1] + a[1:]) / 2 * np.diff(t))

        return t, a, v


def generate_integration_figure(df: pd.DataFrame,
                                time_column: str,
                                acc_columns: list,
                                max_samples: int,
                                figure_path: str):
    """
    Generate and save a figure with acceleration and integrated velocity
    for each supplied accelerometer axis.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_column (str): Column name of the time vector.
        acc_columns (list): List of acceleration columns to integrate.
        max_samples (int): Maximum number of samples to plot.
        figure_path (str): Output path of the saved figure.

    Notes:
        - This function saves a FIGURE (not a table) to results/figures.
    """
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    valid_cols = [c for c in acc_columns if c in df.columns]
    if len(valid_cols) == 0:
        return

    n = len(valid_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    time = df[time_column].values

    for i, col in enumerate(valid_cols):
        acc = df[col].values
        integrated = IntegratedSignal(time, acc, col)
        t, a, v = integrated.evaluate()

        limit = min(len(t), max_samples)

        axes[i].plot(t[:limit], a[:limit], label=f"{col} (acc)", alpha=0.6)
        axes[i].plot(t[:limit], v[:limit], label=f"{col} (vel)", linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].legend()
        axes[i].grid()

    axes[-1].set_xlabel(time_column)
    plt.suptitle("Integration: Acceleration → Velocity", fontsize=14)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#   2) SEGMENT TABLES (TABLE PNG)
# ============================================================
def segment_summary(df: pd.DataFrame, segments: dict) -> pd.DataFrame:
    """
    Compute summary statistics for sensor data segments.

    Args:
        df (pd.DataFrame): Input DataFrame.
        segments (dict): Dictionary where each key is a segment name
                         and each value is a list of columns.

    Returns:
        pd.DataFrame: A table with columns:
            - Segment
            - Total
            - Missing
            - %Missing
    """
    rows = []
    for name, cols in segments.items():
        valid_cols = [c for c in cols if c in df.columns]

        if len(valid_cols) == 0:
            rows.append([name, 0, 0, np.nan])
            continue

        total = int(np.prod(df[valid_cols].shape))
        missing = int(df[valid_cols].isnull().sum().sum())
        pct = missing / total * 100 if total > 0 else np.nan

        rows.append([name, total, missing, round(pct, 2)])

    return pd.DataFrame(rows, columns=["Segment", "Total", "Missing", "%Missing"])


def save_segment_summary_png(df: pd.DataFrame, segments: dict, png_path: str):
    """
    Save the segment summary table as a PNG.

    Args:
        df (pd.DataFrame): Input dataframe.
        segments (dict): Dictionary of segment → column list.
        png_path (str): Output PNG path.
    """
    summary = segment_summary(df, segments)
    df_to_png(summary, png_path, title="Segment Summary")


# ============================================================
#   3) CLASS DEFINITION (TABLE PNG)
# ============================================================
def define_global_classes(df: pd.DataFrame,
                          acc_columns: list,
                          p_low: float = 0.33,
                          p_high: float = 0.66):
    """
    Define a global motion state label ('repose', 'transition', 'motion')
    based on acceleration magnitude percentiles.

    Args:
        df (pd.DataFrame): Input DataFrame.
        acc_columns (list): Columns used to compute acceleration magnitude.
        p_low (float): Lower percentile threshold.
        p_high (float): Upper percentile threshold.

    Returns:
        pd.DataFrame: Modified DataFrame containing:
            - acc_magnitude
            - final_global_mobile_state
    """
    df_out = df.copy()
    cols = [c for c in acc_columns if c in df_out.columns]

    if len(cols) == 0:
        df_out["final_global_mobile_state"] = np.nan
        return df_out

    acc_sq = np.zeros(len(df_out))
    for c in cols:
        acc_sq += df_out[c].fillna(0).values ** 2

    df_out["acc_magnitude"] = np.sqrt(acc_sq)

    low_thr = df_out["acc_magnitude"].quantile(p_low)
    high_thr = df_out["acc_magnitude"].quantile(p_high)

    def label_from_magnitude(x):
        if x < low_thr:
            return "repose"
        elif x > high_thr:
            return "motion"
        else:
            return "transition"

    df_out["final_global_mobile_state"] = df_out["acc_magnitude"].apply(label_from_magnitude)
    return df_out


def save_class_summary_png(df: pd.DataFrame, png_path: str):
    """
    Save class distribution as PNG table.

    Args:
        df (pd.DataFrame): Input DataFrame with class labels.
        png_path (str): Output PNG path.
    """
    if "final_global_mobile_state" not in df.columns:
        df_to_png(pd.DataFrame(), png_path, title="Empty Classes")
        return

    vc = df["final_global_mobile_state"].value_counts().reset_index()
    vc.columns = ["Class", "Count"]

    df_to_png(vc, png_path, title="Class Distribution")


# ============================================================
#   4) ACC FEATURE EXTRACTION (TABLE PNG)
# ============================================================
def extract_acc_features(df: pd.DataFrame, columns: list, dt: float = 0.01) -> pd.DataFrame:
    """
    Extract ACC features per axis:
    - mean
    - standard deviation
    - RMS (root mean square)
    - mean frequency (FFT weighted)
    - peak frequency (FFT maximum)

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Accelerometer columns.
        dt (float): Sampling period (default 0.01s).

    Returns:
        pd.DataFrame: Feature summary for each axis.
    """
    rows = []

    for col in columns:
        if col not in df.columns:
            rows.append([col, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        arr = df[col].dropna().values
        if len(arr) == 0:
            rows.append([col, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        rms = float(np.sqrt(np.mean(arr ** 2)))

        N = len(arr)
        yf = fft(arr)
        xf = np.linspace(0.0, 1.0 / (2 * dt), N // 2)
        mag = 2.0 / N * np.abs(yf[: N // 2])

        if np.sum(mag) > 0:
            mean_freq = float(np.sum(xf * mag) / np.sum(mag))
        else:
            mean_freq = 0.0

        peak_freq = float(xf[np.argmax(mag)]) if len(mag) > 0 else 0.0

        rows.append([col, mean, std, rms, mean_freq, peak_freq])

    return pd.DataFrame(
        rows,
        columns=["variable", "mean", "std", "rms", "mean_freq", "peak_freq"]
    )


def save_features_png(df_feats: pd.DataFrame, png_path: str):
    """
    Save the extracted accelerometer features as a PNG table.

    Args:
        df_feats (pd.DataFrame): Feature DataFrame.
        png_path (str): Output PNG path.
    """
    df_fmt = df_feats.copy()
    numeric_cols = ["mean", "std", "rms", "mean_freq", "peak_freq"]

    for c in numeric_cols:
        if c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].map(
                lambda x: f"{x:.4f}" if pd.notnull(x) else ""
            )

    df_to_png(df_fmt, png_path, title="ACC Features", fontsize=9)


# ============================================================
#   5) RANDOM FOREST (TABLE PNG + FIGURE PNG)
# ============================================================
def train_random_forest_acc(df: pd.DataFrame,
                            feature_cols: list,
                            label_col: str,
                            table_output_path: str,
                            cm_output_path: str,
                            test_size: float = 0.2,
                            random_state: int = 42):
    """
    Train a Random Forest classifier using only accelerometer axes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list): List of accelerometer columns used as features.
        label_col (str): Name of the label column.
        table_output_path (str): Output path for the classification report PNG.
        cm_output_path (str): Output path for the confusion matrix PNG.
        test_size (float): Fraction of dataset for test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple:
            df_report (pd.DataFrame): Classification report table.
            cm (np.ndarray): Confusion matrix.
            labels (list): Class labels in order.
    """
    df_valid = df.dropna(subset=feature_cols + [label_col])
    if len(df_valid) == 0:
        return pd.DataFrame(), np.array([]), []

    X = df_valid[feature_cols].values
    y = df_valid[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rep_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    df_rep = pd.DataFrame(rep_dict).transpose().reset_index().rename(columns={"index": "label"})
    for c in df_rep.columns:
        if df_rep[c].dtype in [float, int]:
            df_rep[c] = df_rep[c].round(3)

    # Save classification report → TABLE
    df_to_png(df_rep, table_output_path, title="Random Forest - Classification Report")

    # Save confusion matrix → FIGURE
    labels = clf.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Random Forest")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return df_rep, cm, labels
