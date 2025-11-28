# LABORATORIO_3 – Project Documentation

This document summarizes the main steps of the analysis and the most relevant results produced by the pipeline, and points to additional material where the algorithmic complexity and workflow are discussed in more detail.

---

## 1. Objective

The goal of this lab is to design a reusable Python pipeline to process multisensor data (UWB, IMU, pose, barometer) and classify each time sample into three motion states based only on accelerometer measurements: **repose**, **transition**, and **motion**. The code is organized following common recommendations for scientific Python projects, with separate modules for preprocessing, visualization, and analysis.

---

## 2. Methods (short overview)

1. **Preprocessing**
   - Load raw CSV `const1-trial2-tdoa2.csv`.
   - Remove rows with missing values in `tdoa_meas`.
   - Keep `t_pose`, `acc_x`, `acc_y`, `acc_z`.
   - Apply a moving-average filter to each accelerometer axis and save `const1-trial2-tdoa2_clean.csv`.

2. **Exploratory visualization**
   - Correlation heatmap for `acc_x`, `acc_y`, `acc_z` (`results/figures/heatmap_variables.jpg`).
   - Boxplot and violin plot to inspect distribution and outliers (`boxplot_variables.jpg`, `violinplot_variables.jpg`).

3. **Signal analysis and feature extraction**
   - Trapezoidal integration of acceleration to velocity for each axis (`integracion_acc.jpg`).
   - Time- and frequency-domain features per axis (mean, std, RMS, mean frequency, peak frequency), summarized in `results/tables/tabla_features.jpg`.

4. **Motion-state definition and classification**
   - Compute acceleration magnitude and split its distribution using quantiles to obtain three balanced states (`repose`, `transition`, `motion`), summarized in `tabla_clases.jpg`.
   - Train a Random Forest on `acc_x`, `acc_y`, `acc_z` to predict the motion state; evaluate with a confusion matrix and a detailed classification report (`rf_confusion.jpg`, `rf_reporte.jpg`).

---

## 3. Key results

- **Data completeness**
  - `tabla_crudos.jpg` shows that UWB and pose channels contain a large fraction of missing values, whereas IMU accelerations and gyros are complete.
  - `tabla_procesados.jpg` confirms that, after preprocessing, the accelerometer data used for classification contain no missing values.

- **Accelerometer behaviour**
  - The heatmap (`heatmap_variables.jpg`) indicates low correlation between axes, suggesting complementary information.
  - Boxplot and violin plots highlight that `acc_z` is centered around gravity, while `acc_x` and `acc_y` oscillate around zero.
  - Filter comparison figures (`acc_x_comparacion.jpg`, `acc_y_comparacion.jpg`, `acc_z_comparacion.jpg`) show that the moving-average filter smooths high‑frequency noise while preserving the trend.

- **Integration**
  - `integracion_acc.jpg` illustrates the cumulative effect of small accelerations over time, producing smooth velocity curves for each axis.

- **Features**
  - `tabla_features.jpg` reports low mean accelerations in `x` and `y`, higher mean around 1 g in `z`, and characteristic frequency content for each axis.

- **Class distribution**
  - `tabla_clases.jpg` shows an almost balanced number of samples in the three motion states (`repose`, `transition`, `motion`), which is appropriate for supervised learning.

- **Classification performance**
  - The confusion matrix (`rf_confusion.jpg`) shows that most samples are correctly classified, with very few off-diagonal counts.
  - The classification report (`rf_reporte.jpg`) indicates high precision, recall, and F1-score (≈0.99) for all three classes, as well as overall accuracy close to 0.996.

---

## 4. Workflow and Big‑O analysis (notebooks)

In addition to this document, the folder `notebooks/reporting/` contains `reporting.ipynb`, a reporting-style notebook that **does not re‑execute the full pipeline** but reconstructs the workflow step by step. The notebook mirrors the structure of the code (preprocessing → visualization → analysis), embeds the final figures and tables, and explains how each function is used inside the project.

Within the same notebook, each core algorithm is annotated with its **Big‑O time complexity**: loading, cleaning and filtering operations scale linearly with the number of samples \(O(N)\); correlation and integration remain efficient for the dataset size; feature extraction is dominated by FFT computations \(O(N \log N)\); and the Random Forest training behaves approximately as \(O(T \cdot N \log N)\), with \(T\) the number of trees. These results show that the current implementation scales well for the dataset used in this lab and that the main computational cost is concentrated in the spectral features and the classifier, which is acceptable for offline analysis and guides future extensions to larger datasets.

---

## 5. How to reproduce these results

This repository is designed to run inside a dedicated Anaconda virtual environment defined in `environment.yml`. It is strongly recommended to create and activate this environment before executing any script or notebook to ensure that all dependencies and versions match the ones used to generate the results.

From the project root:

conda env create -f environment.yml
conda activate LABORATORIO_3
python Main.py


This command regenerates all processed data, tables, and figures under `results/`. The notebooks in `notebooks/`, and in particular `notebooks/reporting/reporting.ipynb`, can then be used to inspect the workflow, review the Big‑O analysis, and relate each step of the pipeline to the corresponding visual and tabular outputs.
