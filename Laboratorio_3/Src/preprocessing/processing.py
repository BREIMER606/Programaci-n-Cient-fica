"""
--------------
Módulo reutilizable de preprocesamiento y filtrado de señales.

Contiene funciones puras para:
- Cargar datos
- Limpiar valores NaN
- Seleccionar columnas
- Aplicar filtro de media móvil usando POO
- Guardar CSV procesado
- Guardar figuras generadas

No contiene rutas fijas, prints ni lógica específica del dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# 1. CARGA Y GUARDADO
# =====================================================

def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga un archivo CSV en un DataFrame.

    Args:
        ruta_csv (str): Ruta completa al archivo CSV.

    Returns:
        DataFrame: Datos cargados.
    """
    return pd.read_csv(ruta_csv)


def guardar_csv(df: pd.DataFrame, ruta_salida: str) -> None:
    """Guarda un DataFrame en formato CSV.

    Args:
        df (DataFrame): Datos procesados.
        ruta_salida (str): Ruta donde guardar el archivo.
    """
    df.to_csv(ruta_salida, index=False)


# =====================================================
# 2. LIMPIEZA Y SELECCIÓN
# =====================================================

def limpiar_nan(df: pd.DataFrame, columnas_requeridas: list) -> pd.DataFrame:
    """Elimina filas con NaN en columnas específicas.

    Args:
        df (DataFrame): Datos de entrada.
        columnas_requeridas (list): Columnas que deben estar completas.

    Returns:
        DataFrame: Datos sin NaN en las columnas indicadas.
    """
    return df.dropna(subset=columnas_requeridas)


def seleccionar_columnas(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """Selecciona columnas específicas del DataFrame.

    Args:
        df (DataFrame): DataFrame original.
        columnas (list): Lista de columnas a conservar.

    Returns:
        DataFrame: DataFrame reducido.
    """
    return df[columnas].copy()


# =====================================================
# 3. FILTRADO (POO)
# =====================================================

class SensorSignal:
    """Clase base para representar una señal de un sensor."""
    def __init__(self, time, values, name):
        self.time = time
        self.values = values
        self.name = name

    def evaluate(self):
        raise NotImplementedError("Implementar en subclases.")


class OriginalSignal(SensorSignal):
    """Representa una señal original sin procesar."""
    def evaluate(self):
        return self.values


class FilteredSignal(SensorSignal):
    """Señal filtrada mediante filtro de media móvil."""

    def __init__(self, time, values, name, window_size=10):
        super().__init__(time, values, name)
        self.window_size = window_size

    def evaluate(self):
        """Aplica un filtro de promedio móvil."""
        cumsum = np.cumsum(np.insert(self.values, 0, 0))
        filtered = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / self.window_size

        # Ajustar tamaños
        filtered_full = np.full_like(self.values, np.nan)
        filtered_full[self.window_size-1:] = filtered
        return filtered_full


def aplicar_filtro(df: pd.DataFrame, columnas: list, columna_tiempo: str,
                   window_size: int = 50):
    """Aplica el filtro de media móvil a varias columnas usando POO.

    Args:
        df (DataFrame): Datos de entrada.
        columnas (list): Columnas a filtrar.
        columna_tiempo (str): Columna del tiempo.
        window_size (int): Tamaño de la ventana del filtro.

    Returns:
        dict: Diccionario con señales originales y filtradas.
    """
    time = df[columna_tiempo].values
    resultados = {}

    for col in columnas:
        orig = OriginalSignal(time, df[col].values, f"{col}_original")
        filt = FilteredSignal(time, df[col].values, f"{col}_filtered", window_size)

        resultados[col] = {
            "original": orig.evaluate(),
            "filtered": filt.evaluate(),
            "time": time
        }

    return resultados


# =====================================================
# 4. GENERADOR DE FIGURAS
# =====================================================

def guardar_figuras(resultados: dict, carpeta_figuras: str, max_muestras: int = 1000) -> None:
    """Genera y guarda figuras comparando señales originales y filtradas.

    Args:
        resultados (dict): Diccionario generado por aplicar_filtro().
        carpeta_figuras (str): Carpeta donde guardar las figuras.
        max_muestras (int): Muestras máximas para graficar.

    """

    os.makedirs(carpeta_figuras, exist_ok=True)

    for col, data in resultados.items():
        time = data["time"][:max_muestras]
        orig = data["original"][:max_muestras]
        filt = data["filtered"][:max_muestras]

        plt.figure(figsize=(10, 5))
        plt.plot(time, orig, label="Original", alpha=0.5)
        plt.plot(time, filt, label="Filtrada", linewidth=2)
        plt.title(f"Señal {col}: Original vs Filtrada")
        plt.xlabel("Tiempo")
        plt.ylabel(col)
        plt.grid()
        plt.legend()

        ruta = os.path.join(carpeta_figuras, f"{col}_comparacion.png")
        plt.savefig(ruta, dpi=300)
        plt.close()
