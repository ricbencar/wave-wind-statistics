# ERA5 Wave and Wind Statistical Report

## Overview

`stats_era5_data.py` is a command-line Python tool for statistical and extreme-value analysis of ERA5 wave and wind time-series data stored in CSV format.

The script reads a single input CSV file, validates the required columns, cleans and rounds the numerical variables, computes descriptive statistics, performs Generalized Extreme Value (GEV) analysis, generates directional plots, and writes a formatted Excel workbook and a landscape PDF report.

![wave-wind-stats](https://github.com/user-attachments/assets/db9eba46-47f1-412a-981f-3a13446241b5)

## Input CSV Format

The input CSV file must contain the following columns:

| Column | Description |
|---|---|
| `datetime` | Date and time of each record. Parsed as a pandas datetime column. |
| `swh` | Significant wave height. |
| `mwp` | Mean wave period. |
| `mwd` | Mean wave direction, in degrees. |
| `wind` | Wind speed. |
| `dwi` | Wind direction, in degrees. |
| `u10` | 10 m wind velocity component in the x/eastward direction. |
| `v10` | 10 m wind velocity component in the y/northward direction. |

Expected column order:

```text
datetime,swh,mwp,mwd,wind,dwi,u10,v10
```

Column names are normalised internally by stripping spaces and converting names to lower case. All required columns must still be present after this normalisation.

## Direction Convention

The script treats `mwd` and `dwi` as directional variables in degrees and normalises them to the range:

```text
0 <= direction < 360
```

The directional sector analyses use 30° classes:

```text
000-030, 030-060, 060-090, ..., 330-360
```

## Main Outputs

For an input file such as:

```text
input.csv
```

running the script generates:

| Output | Description |
|---|---|
| `output.xlsx` | Formatted Excel workbook with all tabular results. Created in the same folder as the input CSV file. |
| `output.pdf` | Landscape A4 PDF report with tables and figures. Created in the same folder as the input CSV file. |
| `figures/` | Folder containing generated PNG figures used in the PDF report. |

The script does not generate a `.rpt.csv` file. Tabular results are written to `output.xlsx`.

## Analyses Performed

### 1. Data Cleaning and Rounding

The script:

- parses `datetime` as a time index;
- sorts records chronologically;
- converts numerical columns to numeric values;
- drops records without valid `datetime`, `swh`, `mwp`, `mwd`, `wind`, or `dwi`;
- rounds `swh`, `wind`, `u10`, and `v10` to 2 decimal places;
- rounds `mwp` to 1 decimal place;
- rounds and normalises `mwd` and `dwi` as integer directions in the range `[0, 360)`.

### 2. Descriptive Statistics

Descriptive statistics are computed for:

```text
swh, mwp, mwd, wind, dwi, u10, v10
```

The reported statistics include:

- count;
- mean;
- standard deviation;
- minimum and maximum;
- 1%, 5%, 25%, 50%, 75%, 95%, and 99% percentiles;
- skewness;
- kurtosis.

### 3. Overall GEV Analysis

The script fits GEV distributions to annual maxima of:

| Variable | Description |
|---|---|
| `swh` | Annual maximum significant wave height. |
| `wind` | Annual maximum wind speed. |

The annual maxima are extracted using year-end resampling.

The following return periods are calculated:

```text
2, 5, 10, 25, 50, 100, 250, 1000 years
```

At least 3 annual maxima are required for each GEV fit.

### 4. Directional-Sector GEV Analysis

Directional GEV analysis is performed by 30° sectors:

| Analysis | Value variable | Direction variable |
|---|---|---|
| Wave-sector extremes | `swh` | `mwd` |
| Wind-sector extremes | `wind` | `dwi` |

For each sector, the script extracts annual maxima and fits a GEV distribution when at least 3 annual maxima are available.

### 5. Joint Distributions

The script generates percentage joint-distribution tables for:

| Table | Variables |
|---|---|
| Wave height by wave direction | `swh` vs `mwd` |
| Wave height by wave period | `swh` vs `mwp` |
| Wind speed by wind direction | `wind` vs `dwi` |

Continuous-variable intervals are rounded to readable limits. The minimum step is 0.5 units, so the resulting class limits are clean values such as:

```text
0-0.5, 0.5-1, 1-1.5, 1.5-2
```

Direction intervals use 30° classes.

### 6. Plots

The script generates:

- empirical CDF versus fitted GEV plots for overall `swh` and `wind`;
- GEV plots by 30° directional sector;
- directional rose plot for `swh` versus `mwd`;
- directional rose plot for `wind` versus `dwi`.

## Excel Workbook Structure

`output.xlsx` contains formatted worksheets including:

| Worksheet | Contents |
|---|---|
| `Summary` | Input file, analysed rows, date range, output names, and variable list. |
| `Descriptive Statistics` | Statistical summary of all numerical variables. |
| `GEV SWH` | GEV parameters and return levels for `swh`. |
| `GEV Wind` | GEV parameters and return levels for `wind`. |
| `SWH Sector GEV` | 30° sector GEV results for `swh` by `mwd`. |
| `Wind Sector GEV` | 30° sector GEV results for `wind` by `dwi`. |
| `Joint SWH MWD` | Joint distribution of `swh` and `mwd`. |
| `Joint SWH MWP` | Joint distribution of `swh` and `mwp`. |
| `Joint Wind DWI` | Joint distribution of `wind` and `dwi`. |

The workbook uses Excel tables, autofilters, frozen panes, adjusted column widths, landscape page setup, and conditional highlighting of maximum numerical values.

## PDF Report Structure

`output.pdf` is generated in landscape A4 format and includes:

1. title page and descriptive statistics;
2. overall GEV analysis for `swh`;
3. overall GEV analysis for `wind`;
4. sector GEV table and plots for `swh`;
5. sector GEV table and plots for `wind`;
6. joint-distribution table for `swh` vs `mwd`;
7. joint-distribution table for `swh` vs `mwp`;
8. joint-distribution table for `wind` vs `dwi`;
9. directional rose plot for `swh` vs `mwd`;
10. directional rose plot for `wind` vs `dwi`.

## Dependencies

The script requires Python 3 and the following packages:

```text
numpy
pandas
matplotlib
scipy
fpdf
windrose
XlsxWriter
```

Install them with:

```bash
python -m pip install numpy pandas matplotlib scipy fpdf windrose XlsxWriter
```

The pip package is `XlsxWriter`; the pandas Excel writer engine used internally is `xlsxwriter`.

## Usage

Run the script from the command line with the input CSV file as the only argument:

```bash
python stats_era5_data.py input.csv
```

Example using an absolute or relative path:

```bash
python stats_era5_data.py data/era5_wave_wind.csv
```

The fixed report names are:

```text
output.xlsx
output.pdf
```

These files are written to the same directory as the input CSV file.

## Command-Line Behaviour

If the input CSV is missing required columns, the script stops with a clear error message listing the missing columns and the full required column set.

If there are fewer than 3 annual maxima available for a required GEV fit, the script stops because the GEV fit would not be statistically meaningful under the script's minimum data rule.

## Adjustable Parameters

The main constants are defined near the top of `stats_era5_data.py`:

| Constant | Purpose |
|---|---|
| `IMAGE_DPI` | Resolution of generated figures. |
| `FIGURES_FOLDER` | Folder where PNG figures are saved. |
| `RETURN_PERIODS` | Return periods used in GEV analysis. |
| `JOINT_DISTRIBUTION_TARGET_BINS` | Approximate number of bins for joint-distribution tables. |
| `JOINT_DISTRIBUTION_MIN_STEP` | Minimum class interval step for readable joint-distribution bins. |
| `MIN_ANNUAL_MAXIMA_FOR_GEV` | Minimum annual maxima required for GEV fitting. |
| `ANNUAL_RESAMPLE_RULE` | Annual resampling rule used to extract annual maxima. |

## Notes on Interpretation

GEV return levels are statistical extrapolations based on annual maxima. Results should be interpreted with engineering judgement, especially for high return periods such as 250 or 1000 years, where uncertainty is normally substantial.

Directional-sector GEV results may be less robust than all-direction results because each sector contains fewer storm events and fewer annual maxima.

Joint-distribution tables are empirical frequency tables expressed as percentages of the available valid records. They are not extreme-value estimates.
