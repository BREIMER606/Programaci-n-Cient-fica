import pandas as pd
import numpy as np
import scipy.signal as signal

# ===========================
#   LOADING + CLEANING
# ===========================

def load_from_github(url_csv: str):
    """Load CSV directly from GitHub."""
    return pd.read_csv(url_csv)

def clean_nan_rows(df: pd.DataFrame):
    """Remove rows containing NaN values."""
    return df.dropna()

def apply_percentiles(df, cols, p_low=5, p_high=95):
    """Clip values based on global percentiles."""
    for col in cols:
        low = df[col].quantile(p_low/100)
        high = df[col].quantile(p_high/100)
        df[col] = df[col].clip(lower=low, upper=high)
    return df


# ===========================
#        FILTERING
# ===========================

class SensorSignal:
    """Filtering class from Lab 1."""
    def __init__(self, time, signal):
        self.time = time
        self.signal = signal

    def butter_filter(self, cutoff=5, fs=100, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_sig = signal.filtfilt(b, a, self.signal)
        return filtered_sig


# ===========================
#       INTEGRATION
# ===========================

class IntegratedSignal:
    """Numerical integration from Lab 1."""
    def __init__(self, time, signal):
        self.time = time
        self.signal = signal

    def integrate(self):
        dt = np.diff(self.time, prepend=self.time[0])
        integrated = np.cumsum(self.signal * dt)
        return integrated
