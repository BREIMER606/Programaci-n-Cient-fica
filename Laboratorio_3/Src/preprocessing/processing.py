"""
Preprocessing:

- Load data from CSV
- Remove NaN values from required columns
- Select specific sensor channels
- Apply a moving-average filter using object-oriented programming (OOP)
- Save cleaned CSV files
- Generate comparison figures for original vs filtered signals
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. LOADING AND SAVING
# =====================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(csv_path)


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Output file path.

    Returns:
        None
    """
    df.to_csv(output_path, index=False)


# =====================================================
# 2. CLEANING AND SELECTION
# =====================================================

def clean_nan(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """
    Remove rows containing NaN values in the specified required columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): Columns that must not contain NaN values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    return df.dropna(subset=required_columns)


def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Select a subset of columns from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to keep.

    Returns:
        pd.DataFrame: Reduced DataFrame containing only the selected columns.
    """
    return df[columns].copy()


# =====================================================
# 3. MOVING AVERAGE FILTER (OOP)
# =====================================================

class SensorSignal:
    """
    Base class representing a generic sensor signal.

    Attributes:
        time (np.ndarray): Time vector associated with the signal.
        values (np.ndarray): Sensor values.
        name (str): Name of the signal.
    """

    def __init__(self, time, values, name):
        """
        Initialize the SensorSignal object.

        Args:
            time (array-like): Time samples.
            values (array-like): Signal samples.
            name (str): Identifier for the signal.
        """
        self.time = time
        self.values = values
        self.name = name

    def evaluate(self):
        """
        Evaluate or return the processed form of the signal.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in subclasses.")


class OriginalSignal(SensorSignal):
    """
    Represents the original, unprocessed sensor signal.
    """

    def evaluate(self):
        """
        Return the unmodified original signal.

        Returns:
            np.ndarray: Original signal values.
        """
        return self.values


class FilteredSignal(SensorSignal):
    """
    Represents a moving-average filtered version of a sensor signal.

    Attributes:
        window_size (int): Number of samples used in the moving average.
    """

    def __init__(self, time, values, name, window_size=10):
        """
        Initialize the FilteredSignal object.

        Args:
            time (array-like): Time samples.
            values (array-like): Signal samples.
            name (str): Name of the filtered signal.
            window_size (int, optional): Window size of the moving average. Defaults to 10.
        """
        super().__init__(time, values, name)
        self.window_size = window_size

    def evaluate(self):
        """
        Apply a moving-average filter using cumulative sums.

        Returns:
            np.ndarray: Filtered signal, padded with NaN for initial unavailable samples.
        """
        cumsum = np.cumsum(np.insert(self.values, 0, 0))
        filtered = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / self.window_size

        result = np.full_like(self.values, np.nan)
        result[self.window_size - 1:] = filtered
        return result


def apply_filter(df: pd.DataFrame, columns: list, time_column: str,
                 window_size: int = 50):
    """
    Apply moving-average filtering to selected DataFrame columns using OOP.

    Args:
        df (pd.DataFrame): Input DataFrame containing time and sensor signals.
        columns (list): Columns to filter.
        time_column (str): Name of the time column.
        window_size (int, optional): Window length of the moving-average filter.

    Returns:
        dict: Dictionary mapping each column to its original and filtered signal.
    """
    time = df[time_column].values
    results = {}

    for col in columns:
        original = OriginalSignal(time, df[col].values, f"{col}_original")
        filtered = FilteredSignal(time, df[col].values, f"{col}_filtered", window_size)

        results[col] = {
            "original": original.evaluate(),
            "filtered": filtered.evaluate(),
            "time": time
        }

    return results


# =====================================================
# 4. FIGURE GENERATOR
# =====================================================

def save_filter_figures(results: dict, folder: str, max_samples: int = 1000) -> None:
    """
    Generate and save comparison plots between original and filtered signals.

    Args:
        results (dict): Output dictionary from apply_filter(), containing:
                        - 'time': time vector
                        - 'original': original signal values
                        - 'filtered': filtered signal values
        folder (str): Path to store generated PNG figures.
        max_samples (int, optional): Maximum number of samples to plot.

    Returns:
        None
    """
    os.makedirs(folder, exist_ok=True)

    for col, data in results.items():
        t = data["time"][:max_samples]
        original = data["original"][:max_samples]
        filtered = data["filtered"][:max_samples]

        plt.figure(figsize=(10, 5))
        plt.plot(t, original, label="Original", alpha=0.5)
        plt.plot(t, filtered, label="Filtered", linewidth=2)
        plt.title(f"Signal {col}: Original vs Filtered")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.grid()
        plt.legend()

        path = os.path.join(folder, f"{col}_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
