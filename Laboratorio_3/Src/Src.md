# Source code (`Src/`)

This directory contains all reusable source code for the project, organized into three main internal packages:

- `preprocessing/`  
  - Functions and classes for loading CSV files, cleaning missing data, selecting relevant columns and applying a moving-average filter to accelerometer signals.  
  - Responsible for generating the processed dataset and comparison figures between original and filtered signals.

- `analysis/`  
  - Utilities for signal integration (acc → vel), summary tables per sensor segment, global motion-state definition (repose / transition / motion), feature extraction in time and frequency domains, and Random Forest training and evaluation.  
  - Produces all analysis tables (PNG) and figures (integration plots, confusion matrix, etc.).

- `visualization/`  
  - High-level plotting functions for exploratory data analysis: correlation heatmap, boxplots and violin plots over selected variables.  
  - Each function receives a pandas DataFrame, a list of columns and an output path, without hard-coded routes or side effects.

All modules are imported and orchestrated by `Main.py`, which runs the complete pipeline:

`Preprocessing → Visualization → Analysis`.
