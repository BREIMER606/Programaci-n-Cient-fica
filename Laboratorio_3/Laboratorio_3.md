# LABORATORIO_3 — Multisensor Motion Analysis

## Repository structure

LABORATORIO_3/
├─ data/
│ ├─ raw/ # Original CSV files (input)
│ └─ processed/ # Cleaned datasets (generated)
├─ results/
│ ├─ figures/ # PNG plots (generated)
│ └─ tables/ # PNG tables (generated)
├─ notebooks/
│ ├─ exploration/ # EDA & algorithm explanation
│ └─ reporting/ # Complexity notes & final write-ups
├─ Src/
│ ├─ preprocessing/ # Load/clean/filter/save
│ ├─ analysis/ # Integration, classes, features, RF
│ └─ visualization/ # Heatmap, boxplot, violinplot
├─ Test/ # Simple examples/tests
├─ Main.py # End-to-end pipeline runner
├─ environment.yml # Conda environment (LABORATORIO_3)
├─ requirements.txt # (Optional) pip-style deps
└─ LABORATORIO_3.md # This file


---

## 1. Theoretical background

Indoor localization is a key challenge in industrial, logistics and autonomous navigation applications because conventional satellite-based systems (e.g., GPS) suffer from poor accuracy in cluttered, indoor environments.[1] Ultra‑wideband (UWB) radio has emerged as a promising technology thanks to its high temporal resolution and robustness to multipath, enabling accurate ranging at relatively low cost.[2] Nevertheless, non‑line‑of‑sight (NLOS) conditions and reflections introduce biased distance estimates that degrade localization performance.[3], [4]

To improve robustness, many works combine UWB with additional sensing modalities such as inertial measurement units (IMUs), odometry or other exteroceptive sensors.[5] Machine learning techniques are increasingly used to model complex propagation phenomena and residual errors beyond classical analytical models, enabling data‑driven corrections and sensor fusion strategies.[5], [6] In parallel, digital signal processing methods such as moving‑average and more advanced filters are commonly applied to denoise IMU signals before feature extraction and classification.[7], [8]

The dataset used in this lab is derived from a UWB‑based localization experiment with an aerial robot, released as part of the UTIL UWB dataset.[9] This project focuses on the accelerometer component of such a multisensor setup. The main objective is to build a reusable scientific‑programming pipeline that: (i) preprocesses and filters IMU and related signals, (ii) defines global motion states (repose, transition, motion) from accelerometer magnitude, and (iii) trains a Random Forest classifier to predict these states from raw accelerometer channels.

---

## 2. Data and methodology

### 2.1 Dataset

The recorded data include synchronized measurements of the drone’s 3‑D position, inertial signals (accelerations and angular velocities), and barometric pressure, together with UWB time‑difference‑of‑arrival (TDOA) measurements. For this lab, the main file is:

- `data/raw/const1-trial2-tdoa2.csv` – raw record containing:
  - UWB: `t_tdoa`, `idA`, `idB`, `tdoa_meas`
  - IMU accelerometer: `acc_x`, `acc_y`, `acc_z`
  - IMU gyroscope: `gyro_x`, `gyro_y`, `gyro_z`
  - Pose estimates: `pose_x`, `pose_y`, `pose_z`
  - Barometer: `baro`

Preprocessing generates:

- `data/processed/const1-trial2-tdoa2_clean.csv` – cleaned subset keeping `t_pose`, `acc_x`, `acc_y`, `acc_z`, removing rows with missing `tdoa_meas`, and serving as input for visualization and analysis stages.

### 2.2 Pipeline overview

#### Main script (`Main.py`)

| Aspect   | Description |
|---------|-------------|
| Purpose | Central entry point of the project. It orchestrates the complete pipeline by calling the functions from `Src/preprocessing`, `Src/visualization` and `Src/analysis` in the correct order (preprocessing → visualization → analysis) and writing all outputs into the `data/processed/` and `results/` folders. |
| Location | Placed in the project root so it can see the whole repository structure (`Src/`, `data/`, `results/`, `notebooks/`) in a single run and can be executed directly with `python Main.py` without changing the Python path or working directory. This makes it clear that it is the main entry point for anyone cloning the repository. |

The source code is organized into three main Python modules, following the lab requirements for scientific programming projects.

1. **Preprocessing (`Src/preprocessing/processing.py`)**

   - Load raw CSV data and drop rows with missing values in mandatory columns (e.g., `tdoa_meas`).
   - Select the columns required for this lab (`t_pose`, `acc_x`, `acc_y`, `acc_z`).
   - Apply a moving‑average filter, implemented via small OOP classes (`SensorSignal`, `OriginalSignal`, `FilteredSignal`), to each accelerometer axis.
   - Save the processed CSV and comparison plots between raw and filtered signals.

2. **Visualization (`Src/visualization/visualization.py`)**

   - Generate a correlation heatmap of accelerometer components to inspect linear relationships between axes.
   - Produce boxplots and violin plots to analyze the distribution and outliers of each accelerometer axis.
   - All functions receive a DataFrame, a list of variables and an explicit output path, without hard‑coded routes.

3. **Analysis (`Src/analysis/analysis.py`)**

   - Integrate acceleration to velocity with a trapezoidal rule, plotting both signals over time for each axis.
   - Summarize total and missing values for sensor groups (UWB, IMU_acc, IMU_gyro, Pose) on raw and processed data.
   - Define a global motion state `final_global_mobile_state` using the acceleration magnitude and quantile‑based thresholds to obtain three classes: `repose`, `transition`, `motion`.
   - Extract time‑ and frequency‑domain features (mean, standard deviation, RMS, mean frequency, peak frequency) for each accelerometer axis.
   - Train and evaluate a Random Forest classifier based only on `acc_x`, `acc_y`, `acc_z` to predict motion state, and export a classification report and confusion matrix.

---

## 3. Project setup and execution

### 3.1 Environment

A Conda environment file `environment.yml` is provided to ensure reproducibility:

conda env create -f environment.yml
conda activate LABORATORIO_3

Main dependencies include Python 3.11, NumPy, pandas, SciPy, matplotlib, seaborn, scikit‑learn, JupyterLab and ipykernel.  
Optionally, a `requirements.txt` file allows installation via pip:

pip install -r requirements.txt


### 3.2 Running the full pipeline

From the project root (`LABORATORIO_3/`), with the environment activated:

python Main.py


This command:

1. Loads the raw dataset from `data/raw/`.
2. Runs the preprocessing module to create `data/processed/const1-trial2-tdoa2_clean.csv` and filter‑comparison figures.
3. Generates heatmap, boxplot and violin plots under `results/figures/`.
4. Produces integration figures, segment summary tables, motion‑class labels, feature table, Random Forest classification report and confusion matrix under `results/`.

# Tests

This folder contains small unit and integration tests that validate the behavior of the code under `Src/`. The tests run a focused subset of what `Main.py` executes end‑to‑end, so they are faster and help catch regressions without rebuilding the whole pipeline.[web:633]

The main file is:

- `test_analysis.py`: imports functions from `Src/analysis/analysis.py`, builds a small synthetic accelerometer dataset and runs class definition, feature extraction and Random Forest training. It checks that the expected columns are created, that numeric outputs are valid and that a classification report and confusion‑matrix PNG are generated under `results/`. This satisfies one of the requirements in the lab guide, which asks for an explicit code example illustrating the general behavior of the implemented algorithms and providing a way to test the overall repository implementation.

## How to run

From the project root:

conda activate LABORATORIO_3
python -m unittest Test.test_analysis

undefined

### Notebooks

This folder contains documentation-oriented notebooks for the project. The end‑to‑end pipeline is executed via `Main.py`; notebooks serve for explanation and reporting.

## Structure
- reporting/
  - reporting.ipynb — Non-executable end-to-end; it presents the pipeline, annotates each core function with its Big‑O complexity, and embeds the figures/tables previously produced by the pipeline for an organized, readable report 

- assignments/
  - Lab1_Pct.ipynb, Lab2_Pct.ipynb — Earlier course work that informed this repository. They are kept as reference material and as an alternative exploratory path.

## Notes
- Use `Main.py` at the project root to regenerate data, figures, and tables; the reporting notebook is intended for inspection only, not for rerunning the analysis 


---

## 4. References

[1] F. Zafari *et al*., “A Survey of Indoor Localization Systems and Technologies,” *IEEE Communications Surveys & Tutorials*, 2019.  
[2] V. Barral *et al*., “NLOS Classification Based on RSS and Ranging Statistics Obtained from Low‑Cost UWB Devices,” 2019.  
[3] I. Güvenç *et al*., “NLOS Identification and Weighted Least Squares Localization for UWB Systems,” *EURASIP Journal on Advances in Signal Processing*, 2008.  
[4] J. Khodjaev *et al*., “Survey of NLOS Identification and Error Mitigation Problems in UWB Positioning Algorithms,” *Annals of Telecommunications*, 2010.  
[5] H. Wymeersch *et al*., “A Machine Learning Approach to Ranging Error Mitigation for UWB Localization,” *IEEE Transactions on Communications*, 2012.  
[6] W. Gan *et al*., “Real‑time Multi‑sensor Joint Fault Diagnosis Method for Traction Motor Bearings,” *Sensors*, vol. 24, no. 9, 2024, Art. 2766.  
[7] A. Ghio, S. Escalante and J. Tarrillo, “Analysis of Moving Average Filter for IMU Measurements on an 8‑bit Microcontroller,” UTEC, Lima, Peru, 2018.  
[8] K. Nirmal *et al*., “Noise Modeling and Analysis of an IMU‑based Attitude Sensor: Improvement of Performance by Filtering and Sensor Fusion,” arXiv:1608.07053, 2016.  
[9] UTIAS Dynamic Systems Lab, “UTIL UWB Dataset,” 2020. Available: https://utiasdsl.github.io/util-uwb-dataset/.
