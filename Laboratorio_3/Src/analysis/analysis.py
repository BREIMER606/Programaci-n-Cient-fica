import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# ===========================
#       STATISTICS + PCA
# ===========================

def compute_correlation(df):
    """Return correlation matrix."""
    return df.corr()

def compute_pca(df, n_components=2):
    """Compute PCA using scaled numeric features."""
    X = df.select_dtypes(include=["float64", "int64"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    return components, pca.explained_variance_ratio_


# ===========================
#   FEATURE EXTRACTION (Lab 2)
# ===========================

class FeatureExtractor:
    """Simple feature extraction from Lab 2."""
    def __init__(self, df):
        self.df = df

    def time_domain_features(self, col):
        return {
            "mean": self.df[col].mean(),
            "std": self.df[col].std(),
            "max": self.df[col].max()
        }


# ===========================
#     CLASSIFICATION
# ===========================

def train_random_forest(X, y, test_size=0.3):
    """Train a simple Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return model, y_test, y_pred, cm
