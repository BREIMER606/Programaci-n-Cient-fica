import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# RUTAS
# =====================================================

def get_project_root():
    """Devuelve la ruta raíz del proyecto."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_processed(filename):
    """Carga el dataset limpio desde Data/processed/."""
    root = get_project_root()
    filepath = os.path.join(root, "Data", "processed", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo procesado: {filepath}")

    print(f"📥 Cargando dataset procesado:\n   {filepath}")
    return pd.read_csv(filepath)


def get_results_paths():
    """Crea y devuelve rutas a Results/tables y Results/figures."""
    root = get_project_root()

    tables = os.path.join(root, "Results", "tables")
    figures = os.path.join(root, "Results", "figures")

    os.makedirs(tables, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    return tables, figures


# =====================================================
# CORRELACIÓN
# =====================================================
def compute_correlation(df):
    corr = df.corr(numeric_only=True)
    tables, _ = get_results_paths()
    out = os.path.join(tables, "correlation_matrix.csv")
    corr.to_csv(out)
    print(f"📄 Matriz de correlación guardada en:\n   {out}")
    return corr


# =====================================================
# PCA
# =====================================================
def compute_pca(df, n_components=2):
    numeric = df.select_dtypes(include="number")
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(numeric)

    pca_df = pd.DataFrame(
        pca_data,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    tables, _ = get_results_paths()
    out = os.path.join(tables, "pca_components.csv")
    pca_df.to_csv(out, index=False)
    print(f"📄 PCA guardado en:\n   {out}")

    return pca_df, pca.explained_variance_ratio_


# =====================================================
# FEATURES
# =====================================================
class FeatureExtractor:
    def __init__(self, df):
        self.df = df

    def compute(self):
        feats = {}

        num_cols = self.df.select_dtypes(include="number").columns
        for col in num_cols:
            feats[f"{col}_mean"] = self.df[col].mean()
            feats[f"{col}_std"] = self.df[col].std()
            feats[f"{col}_min"] = self.df[col].min()
            feats[f"{col}_max"] = self.df[col].max()

        feats_df = pd.DataFrame([feats])

        tables, _ = get_results_paths()
        out = os.path.join(tables, "features_summary.csv")
        feats_df.to_csv(out, index=False)

        print(f"📄 Features guardados en:\n   {out}")
        return feats_df


# =====================================================
# RANDOM FOREST
# =====================================================
def train_random_forest(df, target="label"):
    if target not in df.columns:
        raise ValueError(f"El dataset no contiene la columna objetivo '{target}'")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    tables, _ = get_results_paths()
    out = os.path.join(tables, "confusion_matrix.csv")
    pd.DataFrame(cm).to_csv(out, index=False)

    print(f"📄 Matriz de confusión guardada en:\n   {out}")

    return model, cm


# ●●● EJECUCIÓN DIRECTA ●●●
if __name__ == "__main__":
    df = load_processed("trial2_clean.csv")
    compute_correlation(df)
    compute_pca(df)
    FeatureExtractor(df).compute()
