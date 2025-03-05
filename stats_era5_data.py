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
#               The input CSV file is expected to have at least the columns:
#                 datetime, swh, mwd, pp1d.
#
#               If the input file uses alternative column names:
#                 - If both local (swh_local, mwd_local) and offshore (swh_offshore, mwd_offshore)
#                   exist, only the local columns will be used.
#                 - Otherwise, if only offshore exist, they will be renamed to swh and mwd.
#
#               The optional wind/dwi columns will be used if available.
#
#               The script saves both intermediate CSV reports and a final
#               formatted PDF report.
# =============================================================================

# Global parameter for image resolution (set to 180 DPI for images)
IMAGE_DPI = 180

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
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

def rename_columns(df):
    """
    Ensure that the expected columns "swh" and "mwd" exist.
    
    - If both "swh_local" and "mwd_local" are present, they are renamed to "swh"
      and "mwd" respectively and any offshore versions are dropped.
    - Otherwise, if only the offshore columns exist (and swh/mwd are missing),
      they are renamed accordingly.
    """
    # Process swh columns:
    if "swh_local" in df.columns:
        df = df.rename(columns={"swh_local": "swh"})
        if "swh_offshore" in df.columns:
            df = df.drop(columns=["swh_offshore"])
    elif "swh" not in df.columns and "swh_offshore" in df.columns:
        df = df.rename(columns={"swh_offshore": "swh"})
    
    # Process mwd columns:
    if "mwd_local" in df.columns:
        df = df.rename(columns={"mwd_local": "mwd"})
        if "mwd_offshore" in df.columns:
            df = df.drop(columns=["mwd_offshore"])
    elif "mwd" not in df.columns and "mwd_offshore" in df.columns:
        df = df.rename(columns={"mwd_offshore": "mwd"})
    
    return df

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
    Returns the modified DataFrame and the coordinates (row, column) of the cell 
    containing the maximum numeric value.
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
    """
    shape, loc, scale = genextreme.fit(annual_max_series)
    return shape, loc, scale

def plot_gev_with_return_lines(annual_max, shape, loc, scale, outpng, var_label, sector_label):
    """
    Generate and save a plot comparing the empirical CDF with a fitted GEV CDF.
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
    Execute the complete extreme value analysis workflow, conditionally
    including wind/dwi calculations if those columns exist.
    """
    print("Reading CSV...")
    # Read CSV file with datetime parsing.
    df = pd.read_csv(input_csv, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    
    # Rename columns if expected names are not found.
    df = rename_columns(df)
    
    # Determine which columns are available.
    available_cols = df.columns.tolist()
    # Always require swh, mwd, and pp1d.
    required_cols = ["swh", "mwd", "pp1d"]
    # Optional wind/dwi columns.
    perform_wind_analysis = ("wind" in available_cols and "dwi" in available_cols)
    if perform_wind_analysis:
        required_cols.extend(["wind", "dwi"])
    else:
        print("Wind and DWI parameters not found. Skipping wind-related analysis.")
    
    # Use only the available (or required) columns.
    df = df[required_cols]
    round_variables(df)
    
    # For sector analysis of swh, determine the directional column.
    # Use "dwi" if available; otherwise, use "mwd" as a proxy.
    dir_col_for_swh = "dwi" if "dwi" in df.columns else "mwd"
    
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
    
    # --- GEV Analysis (Overall) for swh ---
    print("\nPerforming GEV analysis for swh (all directions)...")
    annual_max_swh = df["swh"].resample("Y").max().dropna()
    shape_swh, loc_swh, scale_swh = gev_fit(annual_max_swh)
    rp_swh = {T: genextreme.ppf(1 - 1/T, shape_swh, loc=loc_swh, scale=scale_swh) for T in RETURN_PERIODS}
    
    # --- Optional: GEV Analysis for wind (if available) ---
    if perform_wind_analysis:
        print("\nPerforming GEV analysis for wind (all directions)...")
        annual_max_wind = df["wind"].resample("Y").max().dropna()
        shape_wind, loc_wind, scale_wind = gev_fit(annual_max_wind)
        rp_wind = {T: genextreme.ppf(1 - 1/T, shape_wind, loc=loc_wind, scale=scale_wind) for T in RETURN_PERIODS}
    
    # --- Joint Distributions ---
    print("\nComputing joint distributions (swh vs mwd, swh vs pp1d)...")
    NBINS = 10
    swh_bins = np.linspace(df["swh"].min(), df["swh"].max(), NBINS+1)
    mwd_bins = np.linspace(0, 360, 13)  # 12 bins covering 0 to 360 degrees.
    pp1d_bins = np.linspace(df["pp1d"].min(), df["pp1d"].max(), NBINS+1)
    joint_swh_mwd = make_joint_distribution(df, "swh", "mwd", swh_bins, mwd_bins)
    joint_swh_pp1d = make_joint_distribution(df, "swh", "pp1d", swh_bins, pp1d_bins)
    
    # --- Optional: Joint Distribution for wind vs dwi ---
    if perform_wind_analysis:
        print("\nComputing joint distribution (wind vs dwi)...")
        wind_bins = np.linspace(df["wind"].min(), df["wind"].max(), NBINS+1)
        dwi_bins = np.linspace(0, 360, 13)  # 12 bins for directional data.
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
        if perform_wind_analysis:
            f.write("\n# C) GEV wind (all directions)\n")
            f.write("shape,loc,scale\n")
            f.write(f"{shape_wind:.6f},{loc_wind:.6f},{scale_wind:.6f}\n")
            f.write("ReturnPeriod,ReturnLevel\n")
            for T in RETURN_PERIODS:
                f.write(f"{T},{rp_wind[T]:.6f}\n")
            f.write("\n# D) Joint Dist: wind vs dwi (%)\n")
            joint_wind_dwi.to_csv(f)
        f.write("\n# E) Joint Dist: swh vs mwd (%)\n")
        joint_swh_mwd.to_csv(f)
        f.write("\n# F) Joint Dist: swh vs pp1d (%)\n")
        joint_swh_pp1d.to_csv(f)
    
    # --- GEV Analysis by 30° Sectors for swh ---
    print("\nPerforming GEV analysis for swh by 30° sectors...")
    bins_sector = np.arange(0, 361, 30)
    sector_swh_results = []
    sector_swh_plots = []
    for i in range(len(bins_sector)-1):
        dmin, dmax = bins_sector[i], bins_sector[i+1]
        sector_label = f"{dmin:03.0f}-{dmax:03.0f}"
        # Use the chosen directional column (dwi if available, otherwise mwd)
        sector_data = df[(df[dir_col_for_swh] >= dmin) & (df[dir_col_for_swh] < dmax)]["swh"].resample("Y").max().dropna()
        if len(sector_data) < 3:
            sector_swh_results.append([sector_label, np.nan, np.nan, np.nan, {}])
            continue
        sh, lc, sc = gev_fit(sector_data)
        rp_dict = {T: genextreme.ppf(1 - 1/T, sh, loc=lc, scale=sc) for T in RETURN_PERIODS}
        outpng = os.path.join(FIGURES_FOLDER, f"{base}_swh_sector_{dmin:03}-{dmax:03}.png")
        plot_gev_with_return_lines(sector_data, sh, lc, sc, outpng, "swh", sector_label)
        sector_swh_plots.append((outpng, sector_label))
        sector_swh_results.append([sector_label, sh, lc, sc, rp_dict])
    
    # --- Optional: GEV Analysis by 30° Sectors for wind ---
    if perform_wind_analysis:
        print("\nPerforming GEV analysis for wind by 30° sectors...")
        bins_sector = np.arange(0, 361, 30)
        sector_wind_results = []
        sector_wind_plots = []
        for i in range(len(bins_sector)-1):
            dmin, dmax = bins_sector[i], bins_sector[i+1]
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
    
    # --- Windrose Plots ---
    print("\nGenerating windrose plot for swh (using mwd)...")
    plot_swh_windrose = os.path.join(FIGURES_FOLDER, f"{base}_swh_windrose.png")
    plot_windrose(df, "swh", "mwd", plot_swh_windrose, "swh", "mwd")
    if perform_wind_analysis:
        print("Generating windrose plot for wind (using dwi)...")
        plot_wind_windrose = os.path.join(FIGURES_FOLDER, f"{base}_wind_windrose.png")
        plot_windrose(df, "wind", "dwi", plot_wind_windrose, "wind", "dwi")
    
    # --- Generate PDF Report ---
    pdf_file = input_csv.replace(".csv", ".pdf")
    print(f"\nCreating PDF report: {pdf_file}")
    
    # Pre-generate overall GEV plots for swh and (if available) wind
    plot_swh_all = os.path.join(FIGURES_FOLDER, f"{base}_swh_all.png")
    plot_gev_with_return_lines(annual_max_swh, shape_swh, loc_swh, scale_swh,
                               plot_swh_all, "swh", "all directions")
    if perform_wind_analysis:
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
    
    # 8.3 GEV swh (30° Sectors) Table & Plots
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
    
    # 8.4 Optional: GEV wind (all directions) and sector analysis (if available)
    if perform_wind_analysis:
        # Overall wind analysis
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
    
        # Wind sector analysis
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
    else:
        print("Skipping wind sector analysis due to missing wind/dwi parameters.")
    
    # 8.5 Joint Distribution Tables (swh-based tables always included)
    pdf.add_page()
    j_swh_mwd, swh_mwd_high = add_sums_and_highlight(joint_swh_mwd)
    pdf_print_table(pdf, j_swh_mwd, title="6) Joint Dist: swh vs mwd (%)",
                    decimals=2, highlight_cell=swh_mwd_high, font="Courier", size=10)
    pdf.add_page()
    j_swh_pp1d, swh_pp1d_high = add_sums_and_highlight(joint_swh_pp1d)
    pdf_print_table(pdf, j_swh_pp1d, title="Joint Dist: swh vs pp1d (%)",
                    decimals=2, highlight_cell=swh_pp1d_high, font="Courier", size=10)
    if perform_wind_analysis:
        pdf.add_page()
        j_wind_dwi, wind_dwi_high = add_sums_and_highlight(joint_wind_dwi)
        pdf_print_table(pdf, j_wind_dwi, title="Joint Dist: wind vs dwi (%)",
                        decimals=2, highlight_cell=wind_dwi_high, font="Courier", size=10)
        print("Joint distribution tables (including wind/dwi) added.")
    else:
        print("Joint distribution for wind vs dwi skipped.")
    
    # 8.6 Windrose Plots
    pdf.add_page()
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 8, "7) Windrose: swh vs mwd", ln=True)
    pdf.image(plot_swh_windrose, x=10, w=IMAGE_DPI)
    pdf.ln(10)
    if perform_wind_analysis:
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
