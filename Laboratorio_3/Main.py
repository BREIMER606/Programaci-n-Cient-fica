"""
Main.py — Pipeline completo:
Preprocessing → Visualization → Analysis 

FIGURAS  → results/figures
TABLAS   → results/tables
"""

import os
import pandas as pd

# ---- PREPROCESSING ----
from Src.preprocessing import (
    cargar_datos,
    limpiar_nan,
    seleccionar_columnas,
    aplicar_filtro,
    guardar_csv,
    guardar_figuras,
)

# ---- VISUALIZATION ----
from Src.visualization import (
    generar_mapa_calor,
    generar_boxplot,
    generar_violinplot,
)

# ---- ANALYSIS ----
from Src.analysis import (
    generar_figura_integracion,
    guardar_resumen_segmentos_png,
    definir_clases_globales,
    guardar_resumen_clases_png,
    extract_imu_acc_features,
    guardar_features_png,
    entrenar_random_forest_acc,
)

# =====================================================
# 1. RUTAS (relativas)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ruta_raw = os.path.join(BASE_DIR, "data", "raw", "const1-trial2-tdoa2.csv")
ruta_processed = os.path.join(BASE_DIR, "data", "processed", "const1-trial2-tdoa2_clean.csv")

carpeta_figuras = os.path.join(BASE_DIR, "results", "figures")
carpeta_tablas = os.path.join(BASE_DIR, "results", "tables")

# Visualizations
ruta_heatmap = os.path.join(carpeta_figuras, "heatmap_variables.png")
ruta_boxplot = os.path.join(carpeta_figuras, "boxplot_variables.png")
ruta_violinplot = os.path.join(carpeta_figuras, "violinplot_variables.png")

# Analysis FIGURES
ruta_integracion = os.path.join(carpeta_figuras, "integracion_acc.png")
ruta_rf_cm = os.path.join(carpeta_figuras, "rf_confusion.png")

# Analysis TABLES
ruta_tabla_crudos = os.path.join(carpeta_tablas, "tabla_crudos.png")
ruta_tabla_procesados = os.path.join(carpeta_tablas, "tabla_procesados.png")
ruta_tabla_features = os.path.join(carpeta_tablas, "tabla_features.png")
ruta_tabla_clases = os.path.join(carpeta_tablas, "tabla_clases.png")
ruta_rf_reporte = os.path.join(carpeta_tablas, "rf_reporte.png")

# =====================================================
# 2. PARÁMETROS PREPROCESAMIENTO
# =====================================================
columnas_obligatorias = ["tdoa_meas"]

columnas_interes = [
    "t_pose",
    "acc_x", "acc_y", "acc_z",
]

columna_tiempo = "t_pose"
columnas_para_filtrar = ["acc_x", "acc_y", "acc_z"]
window_size = 50

# =====================================================
# 3. VISUALIZACIÓN
# =====================================================
columnas_visualizacion = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "pose_x", "pose_y", "pose_z",
    "baro",
]

# =====================================================
# 4. ANÁLISIS
# =====================================================
segmentos_dict = {
    "UWB": ['t_tdoa', 'idA', 'idB', 'tdoa_meas'],
    "IMU_acc": ['acc_x', 'acc_y', 'acc_z'],
    "IMU_gyro": ['gyro_x', 'gyro_y', 'gyro_z'],
    "Pose": ['pose_x', 'pose_y', 'pose_z'],
}

imu_cols_acc = ['acc_x', 'acc_y', 'acc_z']

acc_filtered_candidates = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt']
max_muestras_integracion = 1000

# =====================================================
# 5. PIPELINE
# =====================================================
def main():

    # -------------------- PREPROCESS --------------------
    df = cargar_datos(ruta_raw)
    df = limpiar_nan(df, columnas_obligatorias)
    df_reduc = seleccionar_columnas(df, columnas_interes)

    resultados = aplicar_filtro(
        df=df_reduc,
        columnas=columnas_para_filtrar,
        columna_tiempo=columna_tiempo,
        window_size=window_size,
    )

    os.makedirs(os.path.dirname(ruta_processed), exist_ok=True)
    guardar_csv(df_reduc, ruta_processed)
    guardar_figuras(resultados, carpeta_figuras)

    # -------------------- VISUALIZATION --------------------
    df_clean = pd.read_csv(ruta_processed)
    cols_ok = [c for c in columnas_visualizacion if c in df_clean.columns]

    if cols_ok:
        generar_mapa_calor(df_clean, cols_ok, ruta_heatmap)
        generar_boxplot(df_clean, cols_ok, ruta_boxplot)
        generar_violinplot(df_clean, cols_ok, ruta_violinplot)

    # -------------------- ANALYSIS --------------------

    # 1) INTEGRACIÓN FIGURA
    acc_cols_integracion = []
    for orig, filt in zip(imu_cols_acc, acc_filtered_candidates):
        acc_cols_integracion.append(filt if filt in df_clean.columns else orig)

    generar_figura_integracion(
        df=df_clean,
        columna_tiempo=columna_tiempo,
        columnas_acc=acc_cols_integracion,
        max_muestras=max_muestras_integracion,
        ruta_figura=ruta_integracion,
    )

    # 2) TABLA crudos
    df_raw_full = pd.read_csv(ruta_raw)
    guardar_resumen_segmentos_png(df_raw_full, segmentos_dict, ruta_tabla_crudos)

    # 3) TABLA procesados
    guardar_resumen_segmentos_png(df_clean, segmentos_dict, ruta_tabla_procesados)

    # 4) DEFINIR CLASES + TABLA
    df_classes = definir_clases_globales(df_clean, imu_cols_acc)
    guardar_resumen_clases_png(df_classes, ruta_tabla_clases)

    # 5) FEATURES ACC + TABLA
    df_feats = extract_imu_acc_features(df_classes, imu_cols_acc)
    guardar_features_png(df_feats, ruta_tabla_features)

    # 6) RANDOM FOREST ACC
    entrenar_random_forest_acc(
        df=df_classes,
        feature_cols=imu_cols_acc,
        label_col="final_global_mobile_state",
        ruta_tabla_reporte=ruta_rf_reporte,   # TABLA → tables
        ruta_figura_cm=ruta_rf_cm,            # FIGURA → figures
    )

    print("✔ Pipeline completo finalizado.")


if __name__ == "__main__":
    main()
