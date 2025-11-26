import os
import pandas as pd
import numpy as np
import scipy.signal as signal


# ===========================================
#  UTILIDADES DE RUTA
# ===========================================

def get_project_root():
    """Retorna la carpeta raíz del proyecto (Laboratorio_3)."""
    current = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
    return root


def get_data_paths():
    """Retorna rutas absolutas de raw y processed."""
    root = get_project_root()
    raw_path = os.path.join(root, "Data", "raw")
    processed_path = os.path.join(root, "Data", "processed")
    return raw_path, processed_path


# ===========================================
#   LOADING + CLEANING
# ===========================================

def load_local_csv(filename):
    """Load CSV from Data/raw/."""
    raw_path, _ = get_data_paths()
    file_path = os.path.join(raw_path, filename)
    print(f"📥 Cargando archivo: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    return pd.read_csv(file_path)


def clean_nan_rows(df: pd.DataFrame):
    """Remove rows containing NaN values."""
    return df.dropna()


def apply_percentiles(df, cols, p_low=5, p_high=95):
    """Clip values based on global percentiles."""
    for col in cols:
        low = df[col].quantile(p_low / 100)
        high = df[col].quantile(p_high / 100)
        df[col] = df[col].clip(lower=low, upper=high)
    return df


# ===========================================
#        FILTERING
# ===========================================

class SensorSignal:
    """Filtering class from Lab 1."""
    def __init__(self, time, signal_data):
        self.time = time
        self.signal = signal_data

    def butter_filter(self, cutoff=5, fs=100, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_sig = signal.filtfilt(b, a, self.signal)
        return filtered_sig


# ===========================================
#       INTEGRATION
# ===========================================

class IntegratedSignal:
    """Numerical integration from Lab 1."""
    def __init__(self, time, signal_data):
        self.time = time
        self.signal = signal_data

    def integrate(self):
        dt = np.diff(self.time, prepend=self.time[0])
        integrated = np.cumsum(self.signal * dt)
        return integrated


# ===========================================
#   PIPELINE COMPLETO
# ===========================================

def full_preprocess(filename, output_name="cleaned_output.csv"):
    print("\n==============================")
    print("   EJECUTANDO PREPROCESAMIENTO")
    print("==============================")

    # 1. cargar archivo
    df = load_local_csv(filename)

    # 2. eliminar NaN
    df = clean_nan_rows(df)

    # 3. aplicar percentiles solo a columnas numéricas
    numeric_cols = df.select_dtypes(include="number").columns
    df = apply_percentiles(df, numeric_cols)

    # 4. guardar resultado
    _, processed_path = get_data_paths()
    os.makedirs(processed_path, exist_ok=True)

    output_file = os.path.join(processed_path, output_name)
    df.to_csv(output_file, index=False)

    print(f"📤 Archivo procesado guardado en:\n   {output_file}\n")

    return df


# ===========================================
#   EJECUCIÓN DIRECTA DESDE TERMINAL
# ===========================================

if __name__ == "__main__":
    print("🔧 Ejecutando processing.py como script independiente...")

    full_preprocess(
        filename="const1-trial2-tdoa2.csv",
        output_name="trial2_clean.csv"
    )

    print("✔ Procesamiento completado.\n")

