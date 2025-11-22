import pandas as pd
from pathlib import Path
import sys
import os

"""
Main script for the spectral analysis of the SMS configuration 
It includes the following steps:

1. loading and pre-processing of transmission spectrum data. 
2. Visualisation of spectra and landslide of the first minimum peak.
3. Calculation of the sensitivity to refractive index variations.
4. Covariance analysis to explore relationships between fibre diameters and different media. 
5. Detection of relevant peaks in the spectrum based on their prominence and spectral width.
6. Statistical analysis of extracted features, including normality tests (Shapiro), 
 comparison (ANOVA) and non-parametric tests (Wilcoxon). 
7. generate graphs (spectra, displacements, heat maps, histograms, KDE). 

All results, graphs and tables are saved in the specified folders within the project.
"""

# Project root directory 
project_root = Path(__file__).resolve().parents[1]  # O(1)
sys.path.append(str(project_root))  # O(1)

from Source.preprocessing.preprocessing_Data import Preprocessing
from Source.analysis.spectrumAnalyze import Spectrum, AnalyzeSpectrum, Sensitivity, AnalyzeCovariance
from Source.analysis.featuresANDstaticalanalyze import FindPeaks, StaticalAnalysis
from Source.visualization.Visualization2 import Visualizer

# Load Data
data_path = Path("Data") / "raw" / "Data_1104.csv"
preprocessor = Preprocessing()
Original_Data = preprocessor.pre_Data_1104(data_path)  # O(N)

# list of values RI for every column
RI = ['1.33', '1.35', '1.37', '1.39', '1.40', '1.41']  # O(1)
unetching = ['RI_Water', 'RI_B', 'RI_C', 'RI_D', 'RI_E', 'RI_F']  # O(1)
etching = ['RI_Water_etching', 'RI_B_etching', 'RI_C_etching', 'RI_D_etching', 'RI_E_etching', 'RI_F_etching']  # O(1)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'gray', 'maroon']  # O(1)

save_figure = "Results/figures"  # O(1)
result_save = "Results/tables" #O(1)
data_processed = "Data/processed" #O(1)

visualizer = Visualizer(save_figure, result_save)  # O(1)


# Spectrum plot
visualizer.plot_transmission_spectra(Original_Data, unetching, etching, RI, colors)  # O(n)

# Landslide
spectra = [Spectrum(Original_Data['Wavelength'], Original_Data[unetching[i]], Original_Data[etching[i]], RI[i]) for i in range(len(RI))]  # O(n)
analyzer = AnalyzeSpectrum(spectra)  # O(1)
result_landslide = analyzer.landslide()  # O(1)
print('First peak movement')  # O(1)
print(result_landslide)  # O(1)
visualizer.plotLandslide(Original_Data, unetching, etching, RI)  # O(n)

# Sensitivity
sensitivity = Sensitivity(spectra)  # O(1)
result_Sensitivity = sensitivity.gradient()  # O(n)
print('Sensitivity')  # O(1)
print(result_Sensitivity)  # O(1)
visualizer.plotSensitivity(Original_Data, unetching, etching, RI)  # O(n)

# AnalyzeCovariance
analyzerCovariance = AnalyzeCovariance(Original_Data, unetching, etching)  # O(1)
print('Covariance')

unetch_cov = analyzerCovariance.unetchingCov() # O(n²)
print("125 µm:\n", unetch_cov)  # O(1)

etch_cov = analyzerCovariance.etchingCov()# O(n²)
print("25 µm:\n", etch_cov)  # O(1)

visualizer.plotHeatmaps(Original_Data, unetching, etching, RI)  # O(n)

# Relevant peaks 
print("Relevant peaks detection")
features_Data = visualizer.plot_relevant_peaks(Original_Data, RI, unetching, etching, colors)  # O(n)
print(features_Data.head())  # O(1)

features_Data.to_csv('Results/tables/features_detected.csv', index=False)  # O(n)
features_Data.to_csv('Data/processed/features_detected.csv', index=False)
# Statical analysis
print("Statical analysis relevant peaks")  # O(1)
stats = StaticalAnalysis(features_Data, result_save="Results/tables")  # O(1)
print(stats.summary_statistics())  # O(n)

# Statical analysis plot
visualizer.plot_histograms(features_Data)  # O(n)
features_Data['MMF Diameter'] = features_Data['MMF Diameter'].astype(str)  # O(n)
visualizer.plot_KDE(features_Data)  # O(n)

# Shapiro-Wilk
shapiro_results = stats.shapiro_test()  # O(n)
print("Shapiro-Wilk Test")  # O(1)
print(shapiro_results)  # O(1)
shapiro_results.to_csv('Results/tables/Shapiro_results.csv')  # O(n)

# ANOVA
ANOVA_Data = stats.ANOVA_Test()  # O(n)
print("ANOVA Test")  # O(1)
print(ANOVA_Data)  # O(1)
ANOVA_Data.to_csv("Results/tables/anova_results.csv", index=False)  # O(n)

# Wilcoxon
Wilcoxon_Data = stats.Wilcoxon_test()  # O(n)
print("Wilcoxon Test")  # O(1)
print(Wilcoxon_Data)  # O(1)
Wilcoxon_Data.to_csv('Results/tables/Wilcoxon_results.csv')  # O(n)

    # This code has a computational time complexity of O(n²)
