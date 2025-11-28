"""
Visualization:

Provides functions to generate common exploratory data analysis plots:
- Correlation heatmap
- Boxplot for multiple variables
- Violinplot for distribution analysis
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. HEATMAP
# =====================================================

def generate_heatmap(df, columns, figure_path):
    """
    Generate and save a correlation heatmap for selected columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing numeric variables.
        columns (list): List of columns to include in the correlation heatmap.
        figure_path (str): Output path where the PNG file will be saved.

    Returns:
        None
    """
    corr = df[columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    plt.close()


# =====================================================
# 2. BOXPLOT
# =====================================================

def generate_boxplot(df, columns, figure_path):
    """
    Generate and save a boxplot for selected variables.

    Args:
        df (pd.DataFrame): Input DataFrame containing numeric variables.
        columns (list): Columns to be displayed in the boxplot.
        figure_path (str): Output path where the PNG file will be saved.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    df[columns].boxplot()
    plt.xticks(rotation=90)
    plt.title("Variable Distribution and Outliers")

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    plt.close()


# =====================================================
# 3. VIOLINPLOT
# =====================================================

def generate_violinplot(df, columns, figure_path):
    """
    Generate and save a violinplot for selected variables.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to include in the violinplot.
        figure_path (str): Output location for the saved PNG figure.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df[columns], inner="quartile")
    plt.xticks(rotation=90)
    plt.title("Variable Distribution (Violinplot)")

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    plt.close()
