import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


# ===========================
#       BASIC PLOTS
# ===========================

def plot_heatmap(df):
    """Correlation heatmap."""
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_box(df, cols):
    """Boxplot of selected columns."""
    df[cols].boxplot(figsize=(10,6))
    plt.title("Boxplot Distribution")
    plt.show()


# ===========================
#     ADVANCED PLOTS
# ===========================

def plot_pairplot(df):
    """Seaborn pairplot (uses 500 random samples)."""
    sns.pairplot(df.sample(500))
    plt.show()


# ===========================
#      CONFUSION MATRIX
# ===========================

def plot_confusion(cm, model_name="Model"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
