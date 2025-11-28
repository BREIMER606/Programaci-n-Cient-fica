# LABORATORIO_3 – Project Documentation

This document summarizes the main steps of the analysis and the most relevant results produced by the pipeline.

---

## 1. Objective

The goal of this lab is to design a reusable Python pipeline to process multisensor data (UWB, IMU, pose, barometer) and classify each time sample into three motion states based only on accelerometer measurements: **repose**, **transition**, and **motion**. The code is organized following common recommendations for scientific Python projects, with separate modules for preprocessing, visualization, and analysis.[web:547]

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

## 4. How to reproduce these results

From the project root:

conda activate LABORATORIO_3
python Main.py


This command regenerates all processed data, tables, and figures under `results/`. The notebooks in `notebooks/` can be used to reproduce the same steps interactively and to discuss algorithmic complexity in more detail.
