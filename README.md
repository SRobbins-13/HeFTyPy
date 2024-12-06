# <center> HeFTyPy  Data Visualization Module </center>

## Description

HeFTyPy is a Python module, designed for use via a Jupyter Notebook, for visualizing and analyzing HeFTy thermal modeling results. This module provides tools for plotting time-temperature/depth paths, model constraints, age distributions, and comparing measured versus modeled ages for both single-sample and multi-sample HeFTy models.


## Installation

### Required Python Version
To run HeFTyPy, we recommend using Anaconda, a comprehensive data science platform that includes most of the required Python modules (Python 3.6+).

### Full List of Dependencies
```python
numpy
seaborn
scipy
matplotlib
pathlib
collections
typing
os
```

## Single Sample Model Analysis

### Initialize a Single Sample Model
```python
from HeFTyFuncsClasses import SingleSampleModel

# Create a single sample model
model = SingleSampleModel(
    file_name="path/to/hefty/file.txt",
    sample_name="Sample1"
)
```

### Selected Plotting Examples
#### Age Distribution Analysis
```python
# Plot modeled age distributions
model.plot_modeled_age_histograms(
    whatToPlot="both",     # Options: 'both', 'histogram', 'kde'
    pathsToPlot="all",     # Options: 'all', 'good', 'acc'
    ap_x_bounds=(0, 30),  # Custom bounds for apatite ages
    zr_x_bounds=(125, 300)  # Custom bounds for zircon ages
)
```
<p align="center">
  <img src="example_plots/modeled_age_histogram.png" width="900" />
</p>

```python
# Compare measured vs. modeled ages
measured_ages = [15.1, 25.8, 12.8, 15.0, 204.6, 245.2, 243.5]  # Must match number of samples
model.plot_measured_vs_modeled(
    measured_sample_ages=measured_ages,
    pathsToPlot="all",
    color_palette="Set1",
    show_1v1_line = 'both' # Options: 'line', 'point', 'both' or None
)
```
<p align="center">
  <img src="example_plots/measured_vs_modeled ages.png" width="900" />
</p>

#### Basic Path Plotting
```python
# Plot time-temperature paths
model.plotSingleSamplePathData(
    plot_type="paths",
    y_variable="temp",
    pathsToPlot="all",  # Options: 'all', 'good', 'acc'
    plotAgeHistogram = True
)
```
<p align="center">
  <img src="example_plots/time_temp_paths_w_histogram.png" width="900" />
</p>


#### Identify Path Families
```python
# Find paths that pass through specific constraints
matched_paths = model.identifyPathFamilies(
    plot_type="points",
    y_variable="temp",
    c1_x=(30, 0),   # Time constraint: 15-0 Ma
    c1_y=(100, 50),  # Temperature constraint: 100-20°C
    c2_x=(160, 120),  # Second time constraint
    c2_y=None # Second temperature constraint
)
```
<p align="center">
  <img src="example_plots/path_families.png" width="900" />
</p>

## Multi-Sample Model Analysis

### Initialize a Multi-Sample Model
```python
from HeFTyFuncsClasses import MultiSampleModel

# Create a multi-sample model
multi_model = MultiSampleModel(
    folder_path="path/to/hefty/files/",
)

# Access individual samples
sample_model = multi_model.get_sample("Sample1")
```
### Selected Plotting Examples


## References

Example data from [Mackaman-Lofland, C., Lossada, A. C., Fosdick, J. C., Litvak, V. D., Rodríguez, M. P., del Llano, M. B., ... & Giambiagi, L. (2024). Unraveling the tectonic evolution of the Andean hinterland (Argentina and Chile, 30° S) using multi-sample thermal history models. Earth and Planetary Science Letters, 643, 118888.](https://www.sciencedirect.com/science/article/pii/S0012821X24003212)

Other References incldue:
- [Ketcham, R.A., 2005. Forward and inverse modeling of low-temperature thermochronometry data. Rev. Mineral. Geochem. 58 (1), 275–314.](https://pubs.geoscienceworld.org/msa/rimg/article-abstract/58/1/275/87556/Forward-and-Inverse-Modeling-of-Low-Temperature)
- [Ketcham, R.A., 2024. Thermal history inversion from thermochronometric data and complementary information: new methods and recommended practices. Chem. Geol. 653, 122042.](https://www.sciencedirect.com/science/article/pii/S0009254124001220)


## Support

For questions and support, please email samrobbins13@gmail.com with a detailed description of the ask and any accompanying data necessary to provide support. All data will be kept in confidence. 

## License
HeFTyPy is licensed under the Apache License 2.0.