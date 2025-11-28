"""
visualization.py
----------------
Módulo reutilizable para generación de visualizaciones estándar:

- Mapa de calor de correlación
- Boxplot de múltiples variables
- Violinplot de múltiples variables

Todas las funciones:
    - Reciben un DataFrame y lista de columnas
    - No contienen rutas absolutas ni prints
    - Guardan la figura en la ruta indicada por parámetro

Requiere seaborn y matplotlib.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1. MAPA DE CALOR
# =====================================================

def generar_mapa_calor(df, columnas, ruta_figura):
    """Genera y guarda un mapa de calor de correlación.

    Args:
        df (DataFrame): Datos limpios.
        columnas (list): Lista de columnas numéricas.
        ruta_figura (str): Ruta donde se guardará la imagen .png
    """
    corr = df[columnas].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Heatmap - Correlación entre variables")

    os.makedirs(os.path.dirname(ruta_figura), exist_ok=True)
    plt.savefig(ruta_figura, dpi=300)
    plt.close()


# =====================================================
# 2. BOXPLOT
# =====================================================

def generar_boxplot(df, columnas, ruta_figura):
    """Genera y guarda un boxplot para múltiples columnas.

    Args:
        df (DataFrame): Datos limpios.
        columnas (list): Variables a visualizar.
        ruta_figura (str): Ruta donde guardar la figura.
    """
    plt.figure(figsize=(12, 8))
    df[columnas].boxplot()
    plt.xticks(rotation=90)
    plt.title("Distribución y outliers de variables físicas")

    os.makedirs(os.path.dirname(ruta_figura), exist_ok=True)
    plt.savefig(ruta_figura, dpi=300)
    plt.close()


# =====================================================
# 3. VIOLINPLOT
# =====================================================

def generar_violinplot(df, columnas, ruta_figura):
    """Genera y guarda un violinplot para múltiples variables.

    Args:
        df (DataFrame): Datos limpios.
        columnas (list): Variables numéricas.
        ruta_figura (str): Ruta donde guardar la figura.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df[columnas], inner="quartile")
    plt.xticks(rotation=90)
    plt.title("Distribución (Violinplot) de variables físicas")

    os.makedirs(os.path.dirname(ruta_figura), exist_ok=True)
    plt.savefig(ruta_figura, dpi=300)
    plt.close()
