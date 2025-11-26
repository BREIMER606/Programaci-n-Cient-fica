import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# =====================================================
# RUTAS
# =====================================================
def get_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_processed(filename):
    root = get_root()
    path = os.path.join(root, "Data", "processed", filename)
    print(f"📥 Cargando: {path}")
    return pd.read_csv(path)


def get_fig_path():
    root = get_root()
    figp = os.path.join(root, "Results", "figures")
    os.makedirs(figp, exist_ok=True)
    return figp


# =====================================================
# FIGURAS
# =====================================================
def plot_heatmap(df):
    figp = get_fig_path()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    file = os.path.join(figp, "heatmap.png")
    plt.savefig(file, dpi=300)
    plt.close()
    print(f"🖼 Heatmap guardado en:\n   {file}")


def plot_box(df):
    figp = get_fig_path()
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    file = os.path.join(figp, "boxplot.png")
    plt.savefig(file, dpi=300)
    plt.close()
    print(f"🖼 Boxplot guardado en:\n   {file}")


def plot_pairplot(df):
    figp = get_fig_path()
    g = sns.pairplot(df)
    file = os.path.join(figp, "pairplot.png")
    g.savefig(file)
    plt.close()
    print(f"🖼 Pairplot guardado en:\n   {file}")


def plot_confusion(cm):
    figp = get_fig_path()
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="magma", fmt="d")
    file = os.path.join(figp, "confusion_matrix.png")
    plt.savefig(file, dpi=300)
    plt.close()
    print(f"🖼 Matriz de confusión guardada en:\n   {file}")


# ●●● EJECUCIÓN DIRECTA ●●●
if __name__ == "__main__":
    df = load_processed("trial2_clean.csv")
    plot_heatmap(df)
    plot_box(df)
    plot_pairplot(df)
