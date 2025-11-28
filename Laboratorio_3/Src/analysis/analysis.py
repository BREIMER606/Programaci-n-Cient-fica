"""
analysis.py

Módulo reutilizable para análisis del Laboratorio 3:

- integración acc → vel
- tablas de número de datos por segmento (PNG)
- definición de clases por percentiles (PNG)
- extracción de características ACC (PNG)
- Random Forest (ACC) -> reporte PNG + matriz de confusión PNG

IMPORTANTE:
- Las TABLAS .png se guardan SIEMPRE en results/tables (ruta dada como parámetro).
- Las FIGURAS se guardan en results/figures (ruta dada como parámetro).
- Sin prints ni rutas absolutas.
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
#  UTILIDAD: Guardar DataFrame como PNG (TABLA)
# ============================================================
def df_to_png(df: pd.DataFrame, ruta_png: str, title: str = None, fontsize: int = 10):
    """
    Guarda un DataFrame como tabla PNG usando matplotlib.

    TABLAS siempre deben guardarse en results/tables (ruta pasada desde main).

    Args:
        df: DataFrame.
        ruta_png: ruta completa del .png (incluye directorio).
        title: título opcional.
        fontsize: tamaño de fuente.
    """
    os.makedirs(os.path.dirname(ruta_png), exist_ok=True)

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

    plt.savefig(ruta_png, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 1) INTEGRACIÓN ACC → VEL (FIGURA)
# ============================================================
class IntegratedSignal:
    """Integración trapezoidal para señal (acc → vel)."""

    def __init__(self, time: np.ndarray, values: np.ndarray, name: str = ""):
        self.time = np.asarray(time)
        self.values = np.asarray(values)
        self.name = name

    def evaluate(self):
        mask = np.isfinite(self.time) & np.isfinite(self.values)
        t = self.time[mask]
        a = self.values[mask]

        v = np.zeros_like(a)
        if len(t) > 1:
            v[1:] = np.cumsum((a[:-1] + a[1:]) / 2 * np.diff(t))

        return t, a, v


def generar_figura_integracion(df: pd.DataFrame,
                               columna_tiempo: str,
                               columnas_acc: list,
                               max_muestras: int,
                               ruta_figura: str):
    """
    Genera figura de integración (ACC vs VEL).

    Esta es una FIGURA → se guarda en results/figures (ruta pasada desde main).
    """
    os.makedirs(os.path.dirname(ruta_figura), exist_ok=True)

    cols_ok = [c for c in columnas_acc if c in df.columns]
    if len(cols_ok) == 0:
        return

    n = len(cols_ok)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    time = df[columna_tiempo].values

    for i, col in enumerate(cols_ok):
        acc = df[col].values
        integ = IntegratedSignal(time, acc, col)
        t, a, v = integ.evaluate()

        limit = min(len(t), max_muestras)
        axes[i].plot(t[:limit], a[:limit], label=f"{col} (acc)", alpha=0.6)
        axes[i].plot(t[:limit], v[:limit], label=f"{col} (vel)", linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].legend()
        axes[i].grid()

    axes[-1].set_xlabel(columna_tiempo)
    plt.suptitle("Integration: Acceleration → Velocity", fontsize=14)
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 2) TABLAS DE SEGMENTOS (TABLA PNG)
# ============================================================
def resumen_por_segmentos(df: pd.DataFrame, segmentos: dict) -> pd.DataFrame:
    rows = []
    for name, cols in segmentos.items():
        cols_exist = [c for c in cols if c in df.columns]

        if len(cols_exist) == 0:
            rows.append([name, 0, 0, np.nan])
            continue

        total = int(np.prod(df[cols_exist].shape))
        missing = int(df[cols_exist].isnull().sum().sum())
        pct = missing / total * 100 if total > 0 else np.nan

        rows.append([name, total, missing, round(pct, 2)])

    return pd.DataFrame(rows, columns=["Segmento", "Total", "Missing", "%Missing"])


def guardar_resumen_segmentos_png(df: pd.DataFrame, segmentos: dict, ruta_png_tabla: str):
    resumen = resumen_por_segmentos(df, segmentos)
    df_to_png(resumen, ruta_png_tabla, title="Resumen por segmentos")

# ============================================================
# 3) DEFINICIÓN DE CLASES (TABLA PNG) — VERSIÓN EQUILIBRADA
# ============================================================
def definir_clases_globales(df: pd.DataFrame,
                            imu_cols_acc: list,
                            p_low: float = 0.33,
                            p_high: float = 0.66):
    """
    Define la columna 'final_global_mobile_state' usando el
    módulo de la aceleración y percentiles globales.

    - acc_mod baja  -> 'repose'
    - acc_mod media -> 'transition'
    - acc_mod alta  -> 'motion'

    p_low y p_high controlan el balance (~1/3 por clase por defecto).
    """

    df_mod = df.copy()

    # Asegurar que las columnas existen
    cols = [c for c in imu_cols_acc if c in df_mod.columns]
    if len(cols) == 0:
        df_mod["final_global_mobile_state"] = np.nan
        return df_mod

    # Módulo de la aceleración
    acc_sq = np.zeros(len(df_mod))
    for c in cols:
        acc_sq += df_mod[c].fillna(0).values ** 2
    df_mod["acc_mod"] = np.sqrt(acc_sq)

    # Umbrales por percentiles
    low_thr = df_mod["acc_mod"].quantile(p_low)
    high_thr = df_mod["acc_mod"].quantile(p_high)

    def label_from_acc_mod(x):
        if x < low_thr:
            return "repose"
        elif x > high_thr:
            return "motion"
        else:
            return "transition"

    df_mod["final_global_mobile_state"] = df_mod["acc_mod"].apply(label_from_acc_mod)

    return df_mod


def guardar_resumen_clases_png(df: pd.DataFrame, ruta_png_tabla: str):
    if "final_global_mobile_state" not in df.columns:
        df_to_png(pd.DataFrame(), ruta_png_tabla, title="Clases vacías")
        return

    vc = df["final_global_mobile_state"].value_counts().reset_index()
    vc.columns = ["Clase", "Cantidad"]

    df_to_png(vc, ruta_png_tabla, title="Distribución de clases")

# ============================================================
# 4) EXTRACCIÓN DE FEATURES ACC (TABLA PNG)
# ============================================================
def extract_imu_acc_features(df: pd.DataFrame, cols: list, dt: float = 0.01) -> pd.DataFrame:

    rows = []
    for col in cols:
        if col not in df.columns:
            rows.append([col, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        arr = df[col].dropna().values
        if len(arr) == 0:
            rows.append([col, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        rms = float(np.sqrt(np.mean(arr**2)))

        N = len(arr)
        yf = fft(arr)
        xf = np.linspace(0.0, 1.0 / (2 * dt), N // 2)
        mag = 2.0 / N * np.abs(yf[:N // 2])

        if np.sum(mag) > 0:
            mean_freq = float(np.sum(xf * mag) / np.sum(mag))
        else:
            mean_freq = 0.0

        peak_freq = float(xf[np.argmax(mag)]) if len(mag) > 0 else 0.0

        rows.append([col, mean, std, rms, mean_freq, peak_freq])

    return pd.DataFrame(rows, columns=["variable", "mean", "std", "rms", "mean_freq", "peak_freq"])


def guardar_features_png(df_feats: pd.DataFrame, ruta_png_tabla: str):
    """
    Guarda la tabla de features de acelerómetro como PNG, con números
    formateados a 4 decimales para mejorar la legibilidad.
    """
    df_fmt = df_feats.copy()
    num_cols = ["mean", "std", "rms", "mean_freq", "peak_freq"]
    for c in num_cols:
        if c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].map(
                lambda x: f"{x:.4f}" if pd.notnull(x) else ""
            )

    df_to_png(df_fmt, ruta_png_tabla, title="Features IMU (ACC)", fontsize=9)

# ============================================================
# 5) RANDOM FOREST (ACC) → TABLA PNG + FIGURA PNG
# ============================================================
def entrenar_random_forest_acc(df: pd.DataFrame,
                               feature_cols: list,
                               label_col: str,
                               ruta_tabla_reporte: str,
                               ruta_figura_cm: str,
                               test_size: float = 0.2,
                               random_state: int = 42):

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

    # TABLA → results/tables
    df_to_png(df_rep, ruta_tabla_reporte, title="Random Forest - Classification Report")

    # MATRIZ DE CONFUSIÓN → results/figures
    labels = clf.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    os.makedirs(os.path.dirname(ruta_figura_cm), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Matriz de Confusión - Random Forest")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(ruta_figura_cm, dpi=300, bbox_inches="tight")
    plt.close()

    return df_rep, cm, labels
