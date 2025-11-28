# Multisensor Motion Analysis – Scientific Programming Lab 3

This repository contains the final structured project for the **Scientific Programming** course (Master's in Industrial Automation and Control). The goal is to turn exploratory code from previous labs into a **reusable, well-organized pipeline** for preprocessing, analysis, and visualization of a multisensor dataset (UWB, IMU accelerometer and gyroscope, pose and barometer signals).[attached_file:387]

The dataset corresponds to a constrained-motion experiment and includes time-of-arrival measurements from UWB beacons, inertial measurements (accelerations and angular velocities), pose estimates, and barometric pressure. The project focuses on defining motion states from accelerometer data and evaluating a supervised model that classifies each time sample into **repose, transition, or motion**.

The codebase is fully modularized into three main components:

- **Preprocessing:** loading, cleaning, column selection and filtering of raw signals.  
- **Visualization:** generation of heatmaps, boxplots and violin plots for exploratory analysis.  
- **Analysis:** feature extraction, motion-state definition and Random Forest classification, including performance tables and confusion matrix.  

Each component is implemented as a Python module inside `src/`, with clear docstrings and example usage via a single `Main.py` entrypoint.[attached_file:387]

---

## Repository Structure

The repository follows a data-science-style layout to improve reproducibility and reuse.[attached_file:387]

project_root/
├─ data/
│ ├─ raw/ # Archivos CSV originales (p. ej., const1-trial2-tdoa2.csv)
│ └─ cooked/ # Conjuntos de datos limpios generados por la canalización
├─ src/
│ ├─ cooked/ # Biblioteca de preprocesamiento reutilizable (processing.py)
│ ├─ analysis/ # Biblioteca de análisis reutilizable (analysis.py)
│ └─ cooked/ # Biblioteca de visualización reutilizable (visualization.py)
├─ notebooks/
│ ├─ cooked/ # Cuadernos Jupyter para EDA y explicación de algoritmos
│ └─ cooked/ # Cuadernos para análisis de complejidad e informe final
├─ results/
│ ├─ cooked/ # Todos los gráficos (filtros, integración, matriz de confusión, etc.)
│ └─ tablas/ # Tablas exportadas como PNG (segmentos, características, clases, informe de RF)
├─ pruebas/ # Pruebas simples / scripts de ejemplo
├─ documentos/ # PDF con instrucciones de laboratorio y documentos complementarios
├─ requisitos.txt # Dependencias de Python
└─ Main.py # Script principal: ejecuta preprocesamiento → visualización → análisis