#!/usr/bin/env python
# =============================================================================
# Title:        stats_era5_(swh_mwd_pp1d_wind_dwi).py
# Description:  Comprehensive Extreme Value Analysis (EVA), including:
#               - Reading and rounding CSV data
#               - Computing descriptive statistics and joint distributions
#               - Fitting Generalized Extreme Value (GEV) distributions 
#                 (overall and by 30° sectors)
#               - Generating windrose plots
#               - Creating a landscape PDF report with formatted tables and plots
#
#               The input CSV file is expected to have the columns:
#                 datetime, swh, mwd, pp1d, wind, dwi.
#
#               The script saves both intermediate CSV reports and a final
#               formatted PDF report.
# =============================================================================

# Global parameter for image resolution (set to 180 DPI for images)
IMAGE_DPI = 180

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
import sys
from fpdf import FPDF
from windrose import WindroseAxes

# Create "figures" subfolder if it does not exist.
FIGURES_FOLDER = "figures"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# Define the list of return periods to be used in all analyses.
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 250, 1000]

# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================

def round_variables(df):
    """
    Round numerical values in specific DataFrame columns according to defined rules.
    
    The following columns are processed in-place:
      - 'swh': Rounded to 2 decimal places.
      - 'mwd': Rounded to 0 decimal places and converted to integer.
      - 'pp1d': Rounded to 1 decimal place.
      - 'mwp': Rounded to 1 decimal place.
      - 'wind': Rounded to 2 decimal places.
      - 'dwi': Rounded to 0 decimal places and converted to integer.
      
    Parameters:
      df (pandas.DataFrame): Input DataFrame containing the expected columns.
      
    Returns:
      None. The operation is performed in-place.
    """
    if "swh" in df.columns:
        df["swh"] = df["swh"].round(2)
    if "mwd" in df.columns:
        df["mwd"] = df["mwd"].round(0).astype(int)
    if "pp1d" in df.columns:
        df["pp1d"] = df["pp1d"].round(1)
    if "mwp" in df.columns:
        df["mwp"] = df["mwp"].round(1)
    if "wind" in df.columns:
        df["wind"] = df["wind"].round(2)
    if "dwi" in df.columns:
        df["dwi"] = df["dwi"].round(0).astype(int)
    return

def format_interval(interval_str):
    """
    Convert an interval string to a compact, hyphen-separated format.
    
    For example, converts "[0.0, 30.0]" into "0-30".
    
    Parameters:
      interval_str (str): String representing an interval with square brackets
                          and comma-separated numbers.
    
    Returns:
      A string in the format "a-b" where a and b are integers, or the original
      string if any error occurs.
    """
    try:
        parts = interval_str.strip("[]").split(",")
        a = int(round(float(parts[0].strip())))
        b = int(round(float(parts[1].strip())))
        return f"{a}-{b}"
    except Exception:
        return interval_str

def make_joint_distribution(df, var1, var2, bins1, bins2):
    """
    Compute a 2D joint distribution (in percentage) between two variables.
    
    The function bins the data from columns 'var1' and 'var2' using the provided
    bin edges, converts the resulting bin labels to a compact format using
    format_interval(), and then computes the cross-tabulation (frequency count).
    The frequencies are then normalized to represent percentages.
    
    Parameters:
      df (pandas.DataFrame): The input DataFrame.
      var1 (str): Name of the first variable/column.
      var2 (str): Name of the second variable/column.
      bins1 (array-like): Bin edges to be used for the first variable.
      bins2 (array-like): Bin edges to be used for the second variable.
    
    Returns:
      pandas.DataFrame: A DataFrame representing the percentage joint distribution.
    """
    cat1 = pd.cut(df[var1], bins=bins1, include_lowest=True).astype(str)
    cat2 = pd.cut(df[var2], bins=bins2, include_lowest=True).astype(str)
    cat1 = cat1.map(format_interval)
    cat2 = cat2.map(format_interval)
    freq = pd.crosstab(cat1, cat2)
    freq_percent = freq * 100.0 / freq.values.sum()
    return freq_percent

def add_sums_and_highlight(df):
    """
    Process the DataFrame to facilitate later highlighting in the PDF.
    
    This routine creates a copy of the input DataFrame and searches for the 
    largest numeric cell within it. The function returns:
      - The modified DataFrame.
      - The coordinates (row index, column name) of the cell containing the 
        maximum numeric value.
      
    Parameters:
      df (pandas.DataFrame): The DataFrame to be processed.
      
    Returns:
      tuple: (df_out, highlight_cell) where df_out is the modified DataFrame and 
             highlight_cell is a tuple (row_index, column_label) indicating the 
             location of the maximum value.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    numeric_df = df_out[numeric_cols.tolist() + ["Total"]] if "Total" in df_out.columns else df_out[numeric_cols]
    max_val = numeric_df.max().max()
    highlight_cell = None
    for rindex, row in numeric_df.iterrows():
        for cindex, val in row.items():
            if val == max_val:
                highlight_cell = (rindex, cindex)
                break
        if highlight_cell:
            break
    return df_out, highlight_cell

def gev_fit(annual_max_series):
    """
    Fit a Generalized Extreme Value (GEV) distribution to an annual maximum series.
    
    Uses the scipy.stats.genextreme.fit function to determine the shape, location,
    and scale parameters of the GEV distribution that best fits the input data.
    
    Parameters:
      annual_max_series (pandas.Series or array-like): Series of annual maximum values.
      
    Returns:
      tuple: (shape, loc, scale) parameters of the fitted GEV distribution.
    """
    shape, loc, scale = genextreme.fit(annual_max_series)
    return shape, loc, scale

def plot_gev_with_return_lines(annual_max, shape, loc, scale, outpng, var_label, sector_label):
    """
    Generate and save a plot comparing the empirical CDF with a fitted GEV CDF.
    
    The plot includes:
      - Empirical CDF points of the annual maximum data.
      - The fitted GEV CDF curve.
      - Vertical dashed lines indicating return levels for predefined return periods.
      - Text annotations showing the return period and corresponding level.
    
    Parameters:
      annual_max (array-like): Annual maximum data.
      shape (float): Shape parameter of the fitted GEV.
      loc (float): Location parameter of the fitted GEV.
      scale (float): Scale parameter of the fitted GEV.
      outpng (str): Filename (with path) where the PNG image will be saved.
      var_label (str): Label for the variable (e.g., "swh" or "wind") for axis and annotation.
      sector_label (str): Label indicating the sector (or overall) analysis.
    
    Returns:
      None. The plot is saved as a PNG file using the global IMAGE_DPI resolution.
    """
    plt.figure(figsize=(7, 4))
    x_sorted = np.sort(annual_max)
    cdf_emp = np.arange(1, len(x_sorted) + 1) / (len(x_sorted) + 1)
    plt.plot(x_sorted, cdf_emp, "bo", label="Empirical CDF")
    cdf_theo = genextreme.cdf(x_sorted, shape, loc=loc, scale=scale)
    plt.plot(x_sorted, cdf_theo, "r-", label="GEV Fit")
    # Plot vertical dashed lines for each return period
    for i, T in enumerate(RETURN_PERIODS):
        xT = genextreme.ppf(1 - 1/T, shape, loc=loc, scale=scale)
        plt.axvline(xT, color="gray", linestyle="--")
        plt.text(xT, 0.85 - 0.07 * i, f"T={T}\n{var_label}={xT:.2f}",
                 rotation=90, color="gray", ha="center", va="top",
                 transform=plt.gca().get_xaxis_transform())
    plt.xlabel(f"{var_label} (Annual Max)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"GEV Fit: {var_label.upper()} - {sector_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpng, dpi=IMAGE_DPI)
    plt.close()

def plot_windrose(df, var, dir_col, outpng, var_label, direction_label, bins=None):
    """
    Create and save a windrose plot for a given variable against a directional column.
    
    The windrose plot is a specialized polar bar plot, which displays the frequency
    distribution of the variable as a function of direction.
    
    Parameters:
      df (pandas.DataFrame): Input DataFrame containing the data.
      var (str): Name of the variable column (e.g., "swh" or "wind").
      dir_col (str): Name of the column containing directional data (e.g., "mwd" or "dwi").
      outpng (str): Filename (with path) to save the PNG image.
      var_label (str): Descriptive label for the variable.
      direction_label (str): Descriptive label for the direction.
      bins (array-like, optional): Custom bin edges for the variable; if not provided,
                                   default bins based on the data range are used.
    
    Returns:
      None. The generated plot is saved as a PNG file using the global IMAGE_DPI resolution.
    """
    if bins is None:
        # Define default bins based on the variable range (5 bins)
        vmin = df[var].min()
        vmax = df[var].max()
        bins = np.linspace(vmin, vmax, 6)
    plt.figure(figsize=(6, 6))
    ax = WindroseAxes.from_ax()
    ax.bar(df[dir_col], df[var], bins=bins, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.title(f"Windrose: {var_label} vs {direction_label}")
    plt.savefig(outpng, dpi=IMAGE_DPI)
    plt.close()

def pdf_print_table(pdf, df, title="", decimals=2, highlight_cell=None, font="Courier", size=10):
    """
    Render a pandas DataFrame as a formatted table onto a PDF page.
    
    This routine performs the following:
      - Optionally prints a title.
      - Inserts an "Index" column for row labels.
      - Converts all cell values to strings (with number formatting as needed).
      - Dynamically calculates column widths based on content.
      - Scales the total table width to fit within the available width of a landscape
        A4 page.
      - Replaces overly long header text by truncating and using three periods ("...")
        to indicate abbreviation.
      - Uses the FPDF library to add each cell as a table cell in the PDF.
    
    Parameters:
      pdf (FPDF): An instance of the FPDF class.
      df (pandas.DataFrame): The DataFrame to be printed.
      title (str, optional): Title text to be printed above the table.
      decimals (int, optional): Number of decimals to use for float formatting.
      highlight_cell (tuple, optional): A tuple indicating a cell (row, column) to be
                                        highlighted (currently not used for color).
      font (str, optional): Font family to use (default is "Courier").
      size (int, optional): Font size for the table text.
    
    Returns:
      None. The table is directly printed onto the current page of the PDF document.
    """
    if title:
        pdf.set_font(font, style="B", size=size+2)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, title, ln=True, align="L")
    pdf.ln(1)
    working = df.copy()
    working.insert(0, "Index", working.index)
    working.reset_index(drop=True, inplace=True)
    columns = working.columns.tolist()
    col_aligns = []
    matrix_str = []
    # Determine alignment per column based on type
    for col in columns:
        if col == "Index":
            col_aligns.append("L")
        elif pd.api.types.is_numeric_dtype(working[col]):
            col_aligns.append("R")
        else:
            col_aligns.append("L")
    # Convert each cell value to a formatted string
    for row_i in range(len(working)):
        row_str = []
        for col_i, col_name in enumerate(columns):
            val = working.loc[row_i, col_name]
            if isinstance(val, (int, float, np.number)):
                if float(val).is_integer():
                    s_val = f"{int(val)}"
                else:
                    s_val = f"{val:.{decimals}f}"
            else:
                s_val = str(val)
            row_str.append(s_val)
        matrix_str.append(row_str)
    # Compute initial column widths based on maximum string length
    col_widths = []
    for c, col_name in enumerate(columns):
        if col_name == "Index":
            w = 25  # Fixed width for the index column
        else:
            max_len = len(str(col_name))
            for row_lst in matrix_str:
                txt_len = len(row_lst[c])
                if txt_len > max_len:
                    max_len = txt_len
            w = max(20, min(80, max_len * 3.0))
        col_widths.append(w)
    # Adjust the column widths if the total exceeds the available page width.
    available_width = pdf.w - 2 * pdf.l_margin
    total_width = sum(col_widths)
    if total_width > available_width:
        scale_factor = available_width / total_width
        col_widths = [w * scale_factor for w in col_widths]
    row_h = 8  # Height of each row in the table
    # Print header row with a blue background
    pdf.set_font(font, style="B", size=size)
    pdf.set_fill_color(0, 102, 204)
    pdf.set_text_color(255, 255, 255)
    for c, col_name in enumerate(columns):
        col_txt = str(col_name)
        # Truncate header text if it exceeds the cell width
        if len(col_txt) * 3.0 > col_widths[c]:
            max_chars = int(col_widths[c] / 3.0)
            col_txt = col_txt[:max_chars-1] + "..."
        pdf.cell(col_widths[c], row_h, col_txt, border=1, align="C", fill=True)
    pdf.ln(row_h)
    # Print data rows with normal formatting
    pdf.set_font(font, "", size)
    pdf.set_text_color(0, 0, 0)
    for row_i, row_lst in enumerate(matrix_str):
        for c, cell_val in enumerate(row_lst):
            fill = False
            pdf.cell(col_widths[c], row_h, cell_val, border=1, align=col_aligns[c], fill=fill)
        pdf.ln(row_h)
    pdf.ln(5)

# =============================================================================
# SECTION 2: MAIN ANALYSIS FUNCTION
# =============================================================================

def extreme_value_analysis(input_csv):
    """
    Execute the complete extreme value analysis workflow.
    
    This function performs the following steps:
      1. Reads the CSV file (only the necessary columns) and applies rounding.
      2. Computes descriptive statistics and additional statistics such as skewness 
         and kurtosis.
      3. Fits overall GEV distributions for 'swh' and 'wind' using annual maxima.
      4. Computes joint distribution tables (as percentage frequencies) for:
         - swh vs mwd
         - swh vs pp1d
         - wind vs dwi
      5. Saves all computed results to a CSV file.
      6. Performs GEV analysis for 'swh' and 'wind' on 30° sectors (using 'dwi' for sectoring).
      7. Generates windrose plots for swh (against mwd) and wind (against dwi).
      8. Creates a landscape PDF report (A4) that includes:
         - Title page with input file name and report title.
         - Tables for descriptive statistics, GEV analysis (overall and by sectors),
           joint distributions, and windrose plots.
         - Plots for the GEV fits and sector analyses.
         
    Parameters:
      input_csv (str): Filename (with path) of the input CSV file.
      
    Returns:
      None. The function saves a detailed CSV report and a PDF report.
    """
    print("Reading CSV...")
    # Read only the specified columns, parsing 'datetime' and using it as the index.
    cols_to_use = ["datetime", "swh", "mwd", "pp1d", "wind", "dwi"]
    df = pd.read_csv(input_csv, usecols=cols_to_use, parse_dates=["datetime"], index_col="datetime")
    round_variables(df)
    
    # Prepare base filename (without extension) for saving figures.
    base = os.path.splitext(os.path.basename(input_csv))[0]
    
    # --- Descriptive Statistics ---
    print("\nComputing descriptive statistics...")
    desc_df = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    if "count" in desc_df.index:
        desc_df.loc["count"] = desc_df.loc["count"].astype(int)
    desc_df.loc["skewness"] = df.skew()
    desc_df.loc["kurtosis"] = df.kurt()
    desc_df = desc_df.T
    
    # --- GEV Analysis (Overall) ---
    print("\nPerforming GEV analysis for swh and wind (all directions)...")
    annual_max_swh = df["swh"].resample("Y").max().dropna()
    shape_swh, loc_swh, scale_swh = gev_fit(annual_max_swh)
    rp_swh = {T: genextreme.ppf(1 - 1/T, shape_swh, loc=loc_swh, scale=scale_swh) for T in RETURN_PERIODS}
    
    annual_max_wind = df["wind"].resample("Y").max().dropna()
    shape_wind, loc_wind, scale_wind = gev_fit(annual_max_wind)
    rp_wind = {T: genextreme.ppf(1 - 1/T, shape_wind, loc=loc_wind, scale=scale_wind) for T in RETURN_PERIODS}
    
    # --- Joint Distributions ---
    print("\nComputing joint distributions (swh vs mwd, swh vs pp1d, wind vs dwi)...")
    NBINS = 10
    swh_bins = np.linspace(df["swh"].min(), df["swh"].max(), NBINS+1)
    mwd_bins = np.linspace(0, 360, 13)  # Use 12 bins covering 0 to 360 degrees.
    pp1d_bins = np.linspace(df["pp1d"].min(), df["pp1d"].max(), NBINS+1)
    wind_bins = np.linspace(df["wind"].min(), df["wind"].max(), NBINS+1)
    dwi_bins = np.linspace(0, 360, 13)  # Use 12 bins for directional data.
    
    joint_swh_mwd = make_joint_distribution(df, "swh", "mwd", swh_bins, mwd_bins)
    joint_swh_pp1d = make_joint_distribution(df, "swh", "pp1d", swh_bins, pp1d_bins)
    joint_wind_dwi = make_joint_distribution(df, "wind", "dwi", wind_bins, dwi_bins)
    
    # --- Save All Results to CSV ---
    output_csv = input_csv.replace(".csv", ".rpt.csv")
    print(f"\nSaving results to {output_csv}")
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("# A) Descriptive Statistics\n")
        desc_df.to_csv(f)
        f.write("\n# B) GEV swh (all directions)\n")
        f.write("shape,loc,scale\n")
        f.write(f"{shape_swh:.6f},{loc_swh:.6f},{scale_swh:.6f}\n")
        f.write("ReturnPeriod,ReturnLevel\n")
        for T in RETURN_PERIODS:
            f.write(f"{T},{rp_swh[T]:.6f}\n")
        f.write("\n# C) GEV wind (all directions)\n")
        f.write("shape,loc,scale\n")
        f.write(f"{shape_wind:.6f},{loc_wind:.6f},{scale_wind:.6f}\n")
        f.write("ReturnPeriod,ReturnLevel\n")
        for T in RETURN_PERIODS:
            f.write(f"{T},{rp_wind[T]:.6f}\n")
        f.write("\n# D) Joint Dist: swh vs mwd (%)\n")
        joint_swh_mwd.to_csv(f)
        f.write("\n# E) Joint Dist: swh vs pp1d (%)\n")
        joint_swh_pp1d.to_csv(f)
        f.write("\n# F) Joint Dist: wind vs dwi (%)\n")
        joint_wind_dwi.to_csv(f)
    
    # --- GEV Analysis by 30° Sectors for swh ---
    print("\nPerforming GEV analysis for swh by 30° sectors...")
    bins_dwi = np.arange(0, 361, 30)
    sector_swh_results = []
    sector_swh_plots = []
    for i in range(len(bins_dwi)-1):
        dmin, dmax = bins_dwi[i], bins_dwi[i+1]
        sector_label = f"{dmin:03.0f}-{dmax:03.0f}"
        sector_data = df[(df["dwi"] >= dmin) & (df["dwi"] < dmax)]["swh"].resample("Y").max().dropna()
        if len(sector_data) < 3:
            sector_swh_results.append([sector_label, np.nan, np.nan, np.nan, {}])
            continue
        sh, lc, sc = gev_fit(sector_data)
        rp_dict = {T: genextreme.ppf(1 - 1/T, sh, loc=lc, scale=sc) for T in RETURN_PERIODS}
        outpng = os.path.join(FIGURES_FOLDER, f"{base}_swh_sector_{dmin:03}-{dmax:03}.png")
        plot_gev_with_return_lines(sector_data, sh, lc, sc, outpng, "swh", sector_label)
        sector_swh_plots.append((outpng, sector_label))
        sector_swh_results.append([sector_label, sh, lc, sc, rp_dict])
    
    # --- GEV Analysis by 30° Sectors for wind ---
    print("\nPerforming GEV analysis for wind by 30° sectors...")
    sector_wind_results = []
    sector_wind_plots = []
    for i in range(len(bins_dwi)-1):
        dmin, dmax = bins_dwi[i], bins_dwi[i+1]
        sector_label = f"{dmin:03.0f}-{dmax:03.0f}"
        sector_data = df[(df["dwi"] >= dmin) & (df["dwi"] < dmax)]["wind"].resample("Y").max().dropna()
        if len(sector_data) < 3:
            sector_wind_results.append([sector_label, np.nan, np.nan, np.nan, {}])
            continue
        sh, lc, sc = gev_fit(sector_data)
        rp_dict = {T: genextreme.ppf(1 - 1/T, sh, loc=lc, scale=sc) for T in RETURN_PERIODS}
        outpng = os.path.join(FIGURES_FOLDER, f"{base}_wind_sector_{dmin:03}-{dmax:03}.png")
        plot_gev_with_return_lines(sector_data, sh, lc, sc, outpng, "wind", sector_label)
        sector_wind_plots.append((outpng, sector_label))
        sector_wind_results.append([sector_label, sh, lc, sc, rp_dict])
    
    # --- Generate Windrose Plots ---
    print("\nGenerating windrose plots for swh (using mwd) and wind (using dwi)...")
    plot_swh_windrose = os.path.join(FIGURES_FOLDER, f"{base}_swh_windrose.png")
    plot_wind_windrose = os.path.join(FIGURES_FOLDER, f"{base}_wind_windrose.png")
    plot_windrose(df, "swh", "mwd", plot_swh_windrose, "swh", "mwd")
    plot_windrose(df, "wind", "dwi", plot_wind_windrose, "wind", "dwi")
    
    # --- Generate PDF Report ---
    pdf_file = input_csv.replace(".csv", ".pdf")
    print(f"\nCreating PDF report: {pdf_file}")
    
    # Pre-generate overall GEV plots for swh and wind
    plot_swh_all = os.path.join(FIGURES_FOLDER, f"{base}_swh_all.png")
    plot_gev_with_return_lines(annual_max_swh, shape_swh, loc_swh, scale_swh,
                               plot_swh_all, "swh", "all directions")
    plot_wind_all = os.path.join(FIGURES_FOLDER, f"{base}_wind_all.png")
    plot_gev_with_return_lines(annual_max_wind, shape_wind, loc_wind, scale_wind,
                               plot_wind_all, "wind", "all directions")

    # Create PDF in landscape orientation (A4)
    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Page: Display input file name and report title.
    pdf.set_font("Courier", "B", 22)
    pdf.cell(0, 10, input_csv, ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Courier", "B", 18)
    pdf.cell(0, 10, "Extreme Event Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    # 8.1 Descriptive Statistics Table
    desc_with_sums, desc_highlight = add_sums_and_highlight(desc_df)
    pdf_print_table(pdf, desc_with_sums, title="1) Descriptive Statistics",
                    decimals=2, highlight_cell=desc_highlight, font="Courier", size=10)
    print("Descriptive statistics table added.")
    
    # 8.2 GEV swh (all directions) Table & Plot
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "2) GEV: swh (all directions)", ln=True)
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 6, f"Shape={shape_swh:.4f}, Loc={loc_swh:.4f}, Scale={scale_swh:.4f}", ln=True)
    pdf.ln(2)
    pdf.cell(0, 6, "Return Periods => Return Levels (swh):", ln=True)
    for T in RETURN_PERIODS:
        pdf.cell(0, 6, f"  T={T}: {rp_swh[T]:.2f}", ln=True)
    pdf.ln(3)
    pdf.image(plot_swh_all, x=10, w=220)
    pdf.ln(10)
    print("GEV swh (all directions) table and plot added.")
    
    # 8.3 GEV swh (30° Sectors) Table
    pdf.add_page()
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "3) GEV: swh by 30° Sectors", ln=True)
    pdf.set_font("Courier", "", 10)
    swh_sector_index = []
    swh_sector_rows = []
    for label, sh, lc, sc, rp_dict in sector_swh_results:
        swh_sector_index.append(label)
        if not rp_dict:
            row = [np.nan, np.nan, np.nan] + [np.nan]*len(RETURN_PERIODS)
        else:
            row = [sh, lc, sc] + [rp_dict[t] for t in RETURN_PERIODS]
        swh_sector_rows.append(row)
    swh_cols = ["shape", "loc", "scale"] + [f"RP_{t}" for t in RETURN_PERIODS]
    swh_sector_df = pd.DataFrame(swh_sector_rows, index=swh_sector_index, columns=swh_cols)
    swh_sector_df_sums, swh_sector_highlight = add_sums_and_highlight(swh_sector_df)
    pdf_print_table(pdf, swh_sector_df_sums, title="SWH GEV by Sector",
                    decimals=2, highlight_cell=swh_sector_highlight, font="Courier", size=10)
    print("GEV swh (30° sectors) table added.")
    
    # Insert swh sector plots (2 per page)
    for i in range(0, len(sector_swh_plots), 2):
        pdf.add_page()
        for j in range(2):
            if i+j < len(sector_swh_plots):
                png_file, lbl = sector_swh_plots[i+j]
                pdf.set_font("Courier", "B", 10)
                pdf.cell(0, 6, f"SWH Sector {lbl}", ln=True)
                pdf.image(png_file, x=10, w=220)
                pdf.ln(10)
    print("SWH sector plots added.")
    
    # 8.4 GEV wind (all directions) Table & Plot
    pdf.add_page()
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "4) GEV: wind (all directions)", ln=True)
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 6, f"Shape={shape_wind:.4f}, Loc={loc_wind:.4f}, Scale={scale_wind:.4f}", ln=True)
    pdf.ln(2)
    pdf.cell(0, 6, "Return Periods => Return Levels (wind):", ln=True)
    for T in RETURN_PERIODS:
        pdf.cell(0, 6, f"  T={T}: {rp_wind[T]:.2f}", ln=True)
    pdf.ln(3)
    pdf.image(plot_wind_all, x=10, w=220)
    pdf.ln(10)
    print("GEV wind (all directions) table and plot added.")
    
    # 8.5 GEV wind (30° Sectors) Table
    pdf.add_page()
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "5) GEV: wind by 30° Sectors", ln=True)
    pdf.set_font("Courier", "", 10)
    wind_sector_index = []
    wind_sector_rows = []
    for label, sh, lc, sc, rp_dict in sector_wind_results:
        wind_sector_index.append(label)
        if not rp_dict:
            row = [np.nan, np.nan, np.nan] + [np.nan]*len(RETURN_PERIODS)
        else:
            row = [sh, lc, sc] + [rp_dict[t] for t in RETURN_PERIODS]
        wind_sector_rows.append(row)
    wind_cols = ["shape", "loc", "scale"] + [f"RP_{t}" for t in RETURN_PERIODS]
    wind_sector_df = pd.DataFrame(wind_sector_rows, index=wind_sector_index, columns=wind_cols)
    wind_sector_df_sums, wind_sector_highlight = add_sums_and_highlight(wind_sector_df)
    pdf_print_table(pdf, wind_sector_df_sums, title="WIND GEV by Sector",
                    decimals=2, highlight_cell=wind_sector_highlight, font="Courier", size=10)
    print("GEV wind (30° sectors) table added.")
    
    # Insert wind sector plots (2 per page)
    for i in range(0, len(sector_wind_plots), 2):
        pdf.add_page()
        for j in range(2):
            if i+j < len(sector_wind_plots):
                png_file, lbl = sector_wind_plots[i+j]
                pdf.set_font("Courier", "B", 10)
                pdf.cell(0, 6, f"WIND Sector {lbl}", ln=True)
                pdf.image(png_file, x=10, w=220)
                pdf.ln(10)
    print("WIND sector plots added.")
    
    # 8.6 Joint Distribution Tables
    pdf.add_page()
    j_swh_mwd, swh_mwd_high = add_sums_and_highlight(joint_swh_mwd)
    pdf_print_table(pdf, j_swh_mwd, title="6) Joint Dist: swh vs mwd (%)",
                    decimals=2, highlight_cell=swh_mwd_high, font="Courier", size=10)
    pdf.add_page()
    j_swh_pp1d, swh_pp1d_high = add_sums_and_highlight(joint_swh_pp1d)
    pdf_print_table(pdf, j_swh_pp1d, title="Joint Dist: swh vs pp1d (%)",
                    decimals=2, highlight_cell=swh_pp1d_high, font="Courier", size=10)
    pdf.add_page()
    j_wind_dwi, wind_dwi_high = add_sums_and_highlight(joint_wind_dwi)
    pdf_print_table(pdf, j_wind_dwi, title="Joint Dist: wind vs dwi (%)",
                    decimals=2, highlight_cell=wind_dwi_high, font="Courier", size=10)
    print("Joint distribution tables added.")
    
    # 8.7 Windrose Plots
    pdf.add_page()
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "7) Windrose: swh vs mwd", ln=True)
    pdf.image(plot_swh_windrose, x=10, w=IMAGE_DPI)
    pdf.ln(10)
    pdf.add_page()
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "8) Windrose: wind vs dwi", ln=True)
    pdf.image(plot_wind_windrose, x=10, w=IMAGE_DPI)
    pdf.ln(10)
    print("Windrose plots added.")
    
    # Output the final PDF file.
    pdf.output(pdf_file)
    print(f"PDF report created: {pdf_file}")
        
    print("Analysis complete.")

# =============================================================================
# SECTION 3: SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stats_era5_(swh_mwd_pp1d_wind_dwi).py <filename.csv>")
        sys.exit(1)
    extreme_value_analysis(sys.argv[1])
