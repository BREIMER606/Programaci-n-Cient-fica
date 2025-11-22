# Data

This folder contains two subdirectories. First, **raw/**, which stores the data sets obtained from simulations performed in FIMMWAVE. Both files contain a first column corresponding to the wavelength, which defines the spectral range analysed: in the **Data_SP** file, this range extends from 1.2 to 1.6 µm, while in **Data_1104** it extends from 1.0 to 1.6 µm. The remaining columns represent transmission values as a function of wavelength, where each column corresponds to a medium with a specific diameter. For example, if 7 different media are simulated, 14 columns will be obtained: 7 for a diameter of 125 µm and 7 for 25 µm.

Throughout the project, we worked mainly with the **Data_1104** file, because it covers a wider spectral range and has a higher resolution (more samples), which improves the quality of the analysis.

It is important to note that the refractive indices used in the simulations vary between 1.33 and 1.41, typical values for liquids such as water, ethanol, methanol and propanol, commonly used in experimental tests.