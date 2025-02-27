# Comprehensive Extreme Value Analysis (EVA)

## Table of Contents
- [Title](#title)
- [Description](#description)
- [Input Format](#input-format)
- [Functionality Overview](#functionality-overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Utility Functions](#utility-functions)
- [Main Analysis Workflow](#main-analysis-workflow)
- [License](#license)

## Title
`stats_era5_(swh_mwd_pp1d_wind_dwi).py`

## Description
This Python script performs a Comprehensive Extreme Value Analysis (EVA) on environmental data. Specifically, it includes:
- Reading and rounding CSV data.
- Computing descriptive statistics and joint distributions.
- Fitting Generalized Extreme Value (GEV) distributions for overall data and by 30° sectors.
- Generating windrose plots to visualize the distribution of data against directional variables.
- Creating a landscape PDF report containing formatted tables and plots.

## Input Format
The input CSV file should contain the following columns:
- `datetime`: Timestamps for the data points.
- `swh`: Significant wave height.
- `mwd`: Mean wave direction.
- `pp1d`: 1-day precipitation.
- `wind`: Wind speed.
- `dwi`: Wind direction.

The script expects these columns to be present for successful analysis.

## Functionality Overview
1. **Data Scaling**: Rounds numerical values in specified columns.
2. **Descriptive Statistics**: Calculates basic statistical measures including skewness and kurtosis.
3. **GEV Fitting**: Fits GEV distributions to annual maxima of significant wave height (swh) and wind data.
4. **Joint Distributions**: Computes joint distributions for pairs of variables.
5. **Sub-sector Analysis**: Analyzes GEV distributions within 30° directional sectors.
6. **Visualization**: Generates windrose plots and empirical vs. GEV CDF plots.
7. **Reporting**: Outputs results in a detailed CSV report and compiles findings into a PDF report.

## Dependencies
To run this script, ensure the following packages are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `fpdf`
- `windrose`

You can install the required packages using `pip`:
```bash
pip install numpy pandas matplotlib scipy fpdf windrose
```

## Usage
To execute the script, provide the filename of the input CSV file as an argument:
     
python stats_era5_(swh_mwd_pp1d_wind_dwi).py <filename.csv>

The script will generate two output files:

A detailed CSV report (.rpt.csv)
A PDF report (.pdf) with visualizations and statistics.

## Utility Functions

round_variables(df): Rounds specific columns of a DataFrame to defined decimal places.
format_interval(interval_str): Converts an interval string to a compact format.
make_joint_distribution(df, var1, var2, bins1, bins2): Computes a 2D joint distribution between two variables in percentage.
add_sums_and_highlight(df): Enhances a DataFrame for highlighting maximum values for PDF output.
gev_fit(annual_max_series): Fits a GEV distribution to an annual maximum series.
plot_gev_with_return_lines(...): Generates a plot comparing empirical CDF with a fitted GEV CDF.
plot_windrose(df, var, dir_col, ...): Creates a windrose plot for a given variable against its directional data.
pdf_print_table(pdf, df, ...): Renders a DataFrame as a formatted table onto a PDF page.

## Main Analysis Workflow
extreme_value_analysis(input_csv)

This function encapsulates the entire analysis process. It handles:

Reading of the input CSV.
Data rounding and descriptive statistics computation.
GEV fitting and analysis (both overall and by sector).
Joint distribution calculations.
Visualization creation including CDF plots and windrose plots.
PDF report generation containing all results.
