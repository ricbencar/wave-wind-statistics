#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# stats_era5_data.py
#
# Extreme-value and statistical report generator for ERA5 wave and wind time
# series stored in CSV format.
#
# Expected input CSV columns:
#     datetime, swh, mwp, mwd, wind, dwi, u10, v10
#
# Main outputs:
#     output.xlsx       Formatted Excel workbook with all tabular results
#     output.pdf        Landscape A4 PDF report with tables and figures
#     figures/          Generated PNG figures used in the PDF report
#
# Main analyses:
#     - Descriptive statistics for all numerical variables
#     - Overall GEV analysis for significant wave height and wind speed
#     - 30-degree directional-sector GEV analysis for wave and wind extremes
#     - Joint distributions for swh/mwd, swh/mwp and wind/dwi
#     - Directional rose plots for wave height and wind speed
#
# Direction conventions:
#     - mwd is the mean wave direction, in degrees from 0 to 360.
#     - dwi is the wind direction, in degrees from 0 to 360.
#     - Direction values are normalised to the range [0, 360).
# =============================================================================

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy.stats import genextreme
from windrose import WindroseAxes


# =============================================================================
# USER-ADJUSTABLE CONSTANTS
# =============================================================================

IMAGE_DPI = 180
FIGURES_FOLDER = "figures"
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 250, 1000]
JOINT_DISTRIBUTION_TARGET_BINS = 10
JOINT_DISTRIBUTION_MIN_STEP = 0.5
MIN_ANNUAL_MAXIMA_FOR_GEV = 3
ANNUAL_RESAMPLE_RULE = "YE"

EXPECTED_COLUMNS = ["datetime", "swh", "mwp", "mwd", "wind", "dwi", "u10", "v10"]
NUMERIC_COLUMNS = ["swh", "mwp", "mwd", "wind", "dwi", "u10", "v10"]
DIRECTION_COLUMNS = ["mwd", "dwi"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class GEVResult:
    """Container for a fitted GEV distribution and associated return levels."""

    shape: float
    loc: float
    scale: float
    return_levels: dict[int, float]
    n_annual_maxima: int


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, lower-case column names."""
    renamed = {col: str(col).strip().lower() for col in df.columns}
    return df.rename(columns=renamed)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if the input file does not contain the required columns."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        missing_txt = ", ".join(missing)
        required_txt = ", ".join(EXPECTED_COLUMNS)
        raise ValueError(
            f"Input CSV is missing required column(s): {missing_txt}\n"
            f"Required columns: {required_txt}"
        )


def read_input_csv(input_csv: Path) -> pd.DataFrame:
    """Read, validate, clean and index the ERA5 input CSV file."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    df = normalise_column_names(df)
    validate_required_columns(df)

    df = df[EXPECTED_COLUMNS].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["swh", "mwp", "mwd", "wind", "dwi"])

    df["swh"] = df["swh"].round(2)
    df["mwp"] = df["mwp"].round(1)
    df["wind"] = df["wind"].round(2)
    df["u10"] = df["u10"].round(2)
    df["v10"] = df["v10"].round(2)

    for col in DIRECTION_COLUMNS:
        df[col] = np.mod(df[col].round(0), 360).astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after parsing datetime and numerical columns.")

    return df


# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================

def make_joint_distribution_bins(
    series: pd.Series,
    target_bins: int = JOINT_DISTRIBUTION_TARGET_BINS,
    minimum_step: float = JOINT_DISTRIBUTION_MIN_STEP,
) -> np.ndarray:
    """
    Create readable bins for joint-distribution tables.

    Bin limits are rounded to multiples of ``minimum_step``. This avoids table
    classes such as 0.37-0.82 and favours integer or half-unit limits such as
    0-0.5, 0.5-1.0 or 1-2.
    """
    clean = series.dropna()
    if clean.empty:
        raise ValueError("Cannot create bins from an empty series.")

    vmin = float(clean.min())
    vmax = float(clean.max())

    if math.isclose(vmin, vmax):
        lower = math.floor((vmin - minimum_step) / minimum_step) * minimum_step
        upper = math.ceil((vmax + minimum_step) / minimum_step) * minimum_step
        return np.arange(lower, upper + minimum_step * 0.5, minimum_step)

    raw_step = (vmax - vmin) / max(1, target_bins)
    step = max(minimum_step, math.ceil(raw_step / minimum_step) * minimum_step)

    lower = math.floor(vmin / step) * step
    upper = math.ceil(vmax / step) * step
    if upper <= vmax or math.isclose(upper, vmax):
        upper += step
    if math.isclose(lower, upper):
        upper = lower + step

    return np.arange(lower, upper + step * 0.5, step)


def format_bin_limit(value: float) -> str:
    """Format bin limits as integers whenever possible, otherwise with one decimal."""
    rounded = round(float(value), 10)
    if math.isclose(rounded, round(rounded), abs_tol=1e-9):
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


def format_interval(interval_text: str) -> str:
    """Convert pandas interval text to compact integer or half-unit table labels."""
    try:
        cleaned = interval_text.strip("()[]")
        left_text, right_text = cleaned.split(",")
        left = float(left_text.strip())
        right = float(right_text.strip())
        return f"{format_bin_limit(left)}-{format_bin_limit(right)}"
    except Exception:
        return interval_text


def make_joint_distribution(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    bins1: Iterable[float],
    bins2: Iterable[float],
) -> pd.DataFrame:
    """Compute a percentage joint distribution between two variables."""
    work = df[[var1, var2]].dropna().copy()
    if work.empty:
        raise ValueError(f"Cannot compute joint distribution for {var1} and {var2}: no valid data.")

    bins1_array = np.asarray(list(bins1), dtype=float)
    bins2_array = np.asarray(list(bins2), dtype=float)
    labels1 = [f"{format_bin_limit(bins1_array[i])}-{format_bin_limit(bins1_array[i + 1])}" for i in range(len(bins1_array) - 1)]
    labels2 = [f"{format_bin_limit(bins2_array[i])}-{format_bin_limit(bins2_array[i + 1])}" for i in range(len(bins2_array) - 1)]

    cat1 = pd.cut(work[var1], bins=bins1_array, labels=labels1, include_lowest=True, right=False)
    cat2 = pd.cut(work[var2], bins=bins2_array, labels=labels2, include_lowest=True, right=False)

    freq = pd.crosstab(cat1, cat2, dropna=False)
    total = float(freq.values.sum())
    if total <= 0.0:
        raise ValueError(f"Cannot compute joint distribution for {var1} and {var2}: zero observations.")

    freq_percent = freq * 100.0 / total
    freq_percent.loc["Total"] = freq_percent.sum(axis=0)
    freq_percent["Total"] = freq_percent.sum(axis=1)
    return freq_percent.round(3)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for the numerical input variables."""
    desc = df[NUMERIC_COLUMNS].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    desc.loc["skewness"] = df[NUMERIC_COLUMNS].skew(numeric_only=True)
    desc.loc["kurtosis"] = df[NUMERIC_COLUMNS].kurt(numeric_only=True)
    return desc.T


def annual_maxima(series: pd.Series) -> pd.Series:
    """Return annual maxima for a time-indexed series."""
    return series.dropna().resample(ANNUAL_RESAMPLE_RULE).max().dropna()


def fit_gev_from_annual_maxima(maxima: pd.Series) -> GEVResult:
    """Fit a GEV distribution and calculate return levels."""
    maxima = maxima.dropna()
    if len(maxima) < MIN_ANNUAL_MAXIMA_FOR_GEV:
        raise ValueError(
            f"GEV fitting requires at least {MIN_ANNUAL_MAXIMA_FOR_GEV} annual maxima; "
            f"only {len(maxima)} available."
        )

    shape, loc, scale = genextreme.fit(maxima.to_numpy())
    return_levels = {
        period: float(genextreme.ppf(1.0 - 1.0 / period, shape, loc=loc, scale=scale))
        for period in RETURN_PERIODS
    }
    return GEVResult(
        shape=float(shape),
        loc=float(loc),
        scale=float(scale),
        return_levels=return_levels,
        n_annual_maxima=len(maxima),
    )


def fit_sector_gev(
    df: pd.DataFrame,
    value_col: str,
    direction_col: str,
    sector_width: int = 30,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Fit GEV distributions by directional sector and return a table plus plot metadata."""
    rows: list[dict[str, float | str | int]] = []
    plots: list[tuple[str, str]] = []

    for dmin in range(0, 360, sector_width):
        dmax = dmin + sector_width
        label = f"{dmin:03d}-{dmax:03d}"
        sector_df = df[(df[direction_col] >= dmin) & (df[direction_col] < dmax)]
        maxima = annual_maxima(sector_df[value_col])

        row: dict[str, float | str | int] = {
            "sector": label,
            "n_annual_maxima": int(len(maxima)),
            "shape": np.nan,
            "loc": np.nan,
            "scale": np.nan,
        }
        for period in RETURN_PERIODS:
            row[f"RP_{period}"] = np.nan

        if len(maxima) >= MIN_ANNUAL_MAXIMA_FOR_GEV:
            result = fit_gev_from_annual_maxima(maxima)
            row["shape"] = result.shape
            row["loc"] = result.loc
            row["scale"] = result.scale
            for period in RETURN_PERIODS:
                row[f"RP_{period}"] = result.return_levels[period]

            plot_path = str(Path(FIGURES_FOLDER) / f"{value_col}_gev_sector_{label}.png")
            plot_gev_with_return_lines(maxima, result, plot_path, value_col, label)
            plots.append((plot_path, label))

        rows.append(row)

    return pd.DataFrame(rows).set_index("sector"), plots


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_gev_with_return_lines(
    maxima: pd.Series,
    result: GEVResult,
    output_png: str,
    variable_label: str,
    sector_label: str,
) -> None:
    """Save an empirical-vs-fitted GEV CDF plot with return-level lines."""
    values = np.sort(maxima.dropna().to_numpy())
    empirical_cdf = np.arange(1, len(values) + 1) / (len(values) + 1)
    fitted_cdf = genextreme.cdf(values, result.shape, loc=result.loc, scale=result.scale)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(values, empirical_cdf, "o", label="Empirical CDF")
    plt.plot(values, fitted_cdf, "-", label="GEV fit")

    for idx, period in enumerate(RETURN_PERIODS):
        return_level = result.return_levels[period]
        plt.axvline(return_level, linestyle="--", linewidth=0.8)
        y_pos = max(0.05, 0.95 - 0.08 * idx)
        plt.text(
            return_level,
            y_pos,
            f"T={period}\n{variable_label}={return_level:.2f}",
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
            transform=plt.gca().get_xaxis_transform(),
        )

    plt.xlabel(f"Annual maximum {variable_label}")
    plt.ylabel("Cumulative probability")
    plt.title(f"GEV fit: {variable_label.upper()} - {sector_label}")
    plt.grid(True, linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=IMAGE_DPI)
    plt.close()


def plot_windrose(
    df: pd.DataFrame,
    value_col: str,
    direction_col: str,
    output_png: str,
    variable_label: str,
    direction_label: str,
) -> None:
    """Save a directional rose plot for a scalar variable and direction."""
    work = df[[value_col, direction_col]].dropna()
    if work.empty:
        raise ValueError(f"Cannot create windrose for {value_col}/{direction_col}: no valid data.")

    bins = make_joint_distribution_bins(work[value_col], target_bins=5)
    plt.figure(figsize=(6.5, 6.5))
    ax = WindroseAxes.from_ax()
    ax.bar(
        work[direction_col].to_numpy(),
        work[value_col].to_numpy(),
        bins=bins,
        normed=True,
        opening=0.8,
        edgecolor="white",
    )
    ax.set_legend(title=variable_label)
    plt.title(f"Directional rose: {variable_label} vs {direction_label}")
    plt.savefig(output_png, dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close()


# =============================================================================
# PDF REPORT UTILITIES
# =============================================================================

def format_pdf_value(value: object, decimals: int) -> str:
    """Format numbers and strings for compact PDF table cells."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{decimals}f}"
    return str(value)


def truncate_to_cell(text: str, width_mm: float, approx_mm_per_char: float = 2.1) -> str:
    """Trim text so that it fits inside a PDF cell."""
    max_chars = max(3, int(width_mm / approx_mm_per_char))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def pdf_print_table(
    pdf: FPDF,
    df: pd.DataFrame,
    title: str,
    decimals: int = 2,
    font: str = "Courier",
    size: int = 8,
) -> None:
    """Render a pandas DataFrame as a compact landscape PDF table."""
    if title:
        pdf.set_font(font, "B", size + 2)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, title, ln=True, align="L")
        pdf.ln(1)

    working = df.copy()
    working.insert(0, "Index", working.index.astype(str))
    working = working.reset_index(drop=True)

    columns = list(working.columns)
    matrix = [
        [format_pdf_value(working.iloc[row_idx, col_idx], decimals) for col_idx in range(len(columns))]
        for row_idx in range(len(working))
    ]

    widths: list[float] = []
    for col_idx, col in enumerate(columns):
        max_len = len(str(col))
        for row in matrix:
            max_len = max(max_len, len(row[col_idx]))
        width = max(16.0, min(46.0, max_len * 2.3))
        if col == "Index":
            width = max(24.0, min(40.0, width))
        widths.append(width)

    available_width = pdf.w - pdf.l_margin - pdf.r_margin
    total_width = sum(widths)
    if total_width > available_width:
        factor = available_width / total_width
        widths = [width * factor for width in widths]

    row_height = 6.0
    pdf.set_font(font, "B", size)
    pdf.set_fill_color(0, 102, 204)
    pdf.set_text_color(255, 255, 255)
    for col_idx, col in enumerate(columns):
        pdf.cell(widths[col_idx], row_height, truncate_to_cell(str(col), widths[col_idx]), border=1, align="C", fill=True)
    pdf.ln(row_height)

    pdf.set_font(font, "", size)
    pdf.set_text_color(0, 0, 0)
    for row in matrix:
        for col_idx, text in enumerate(row):
            align = "L" if col_idx == 0 else "R"
            pdf.cell(widths[col_idx], row_height, truncate_to_cell(text, widths[col_idx]), border=1, align=align)
        pdf.ln(row_height)

    pdf.ln(4)


def add_gev_summary_to_pdf(pdf: FPDF, title: str, result: GEVResult, plot_path: str, variable_label: str) -> None:
    """Add an overall GEV summary and plot to the PDF report."""
    pdf.add_page()
    pdf.set_font("Courier", "B", 12)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("Courier", "", 10)
    pdf.cell(
        0,
        6,
        f"Annual maxima used: {result.n_annual_maxima} | "
        f"shape={result.shape:.4f}, loc={result.loc:.4f}, scale={result.scale:.4f}",
        ln=True,
    )
    pdf.ln(2)

    rows = [{"Return period": period, f"Return level {variable_label}": result.return_levels[period]} for period in RETURN_PERIODS]
    rp_df = pd.DataFrame(rows).set_index("Return period")
    pdf_print_table(pdf, rp_df, title="Return levels", decimals=2, size=9)
    pdf.image(plot_path, x=10, w=220)


def add_sector_plots_to_pdf(pdf: FPDF, title_prefix: str, plots: list[tuple[str, str]]) -> None:
    """Add sector GEV plots to the PDF, two figures per page."""
    if not plots:
        return

    for idx in range(0, len(plots), 2):
        pdf.add_page()
        for offset in range(2):
            plot_idx = idx + offset
            if plot_idx >= len(plots):
                break
            plot_path, label = plots[plot_idx]
            pdf.set_font("Courier", "B", 10)
            pdf.cell(0, 6, f"{title_prefix} sector {label}", ln=True)
            pdf.image(plot_path, x=10, w=215)
            pdf.ln(8)


# =============================================================================
# REPORT WRITING
# =============================================================================

def output_file(input_csv: Path, filename: str) -> Path:
    """Return a fixed output filename placed next to the input CSV file."""
    return input_csv.resolve().parent / filename


def dataframe_for_excel(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Return a DataFrame prepared for export as a formatted Excel table."""
    out = df.copy()
    out = out.reset_index()
    first_column = out.columns[0]
    if first_column == "index" or str(first_column).strip() == "":
        out = out.rename(columns={first_column: index_name})
    else:
        out = out.rename(columns={first_column: index_name})
    return out


def gev_result_table(result: GEVResult, variable_label: str) -> pd.DataFrame:
    """Return a compact Excel table with GEV parameters and return levels."""
    rows = [
        {"section": "Fit parameters", "metric": "shape", "value": result.shape, "unit": "-"},
        {"section": "Fit parameters", "metric": "loc", "value": result.loc, "unit": variable_label},
        {"section": "Fit parameters", "metric": "scale", "value": result.scale, "unit": variable_label},
        {"section": "Fit parameters", "metric": "annual maxima used", "value": result.n_annual_maxima, "unit": "years"},
    ]
    for period in RETURN_PERIODS:
        rows.append(
            {
                "section": "Return levels",
                "metric": f"RP_{period}",
                "value": result.return_levels[period],
                "unit": variable_label,
            }
        )
    return pd.DataFrame(rows)


def excel_safe_sheet_name(sheet_name: str) -> str:
    """Return a valid Excel worksheet name."""
    invalid = '[]:*?/\\'
    clean = ''.join('_' if char in invalid else char for char in sheet_name)
    return clean[:31]


def column_width_for_series(series: pd.Series, header: str, minimum: int = 10, maximum: int = 26) -> int:
    """Estimate a practical Excel column width."""
    values = series.astype(str).replace("nan", "")
    max_length = max([len(str(header)), *(len(value) for value in values.head(5000))])
    return max(minimum, min(maximum, max_length + 2))


def write_excel_table(
    writer: pd.ExcelWriter,
    workbook: object,
    sheet_name: str,
    df: pd.DataFrame,
    title: str,
    index_name: str | None = None,
    start_row: int = 2,
) -> int:
    """Write one styled table to an Excel worksheet and return the next free row."""
    safe_name = excel_safe_sheet_name(sheet_name)
    worksheet = writer.sheets.get(safe_name)
    if worksheet is None:
        worksheet = workbook.add_worksheet(safe_name)
        writer.sheets[safe_name] = worksheet

    title_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 14,
            "font_color": "#1F4E78",
            "bottom": 1,
            "bottom_color": "#9EADCC",
        }
    )
    note_format = workbook.add_format({"italic": True, "font_color": "#666666"})
    number_format = workbook.add_format({"num_format": "0.000"})
    integer_format = workbook.add_format({"num_format": "0"})
    percent_format = workbook.add_format({"num_format": "0.000"})
    highlight_format = workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"})

    worksheet.write(start_row - 2, 0, title, title_format)
    worksheet.write(start_row - 1, 0, "Generated by stats_era5_data.py", note_format)

    if index_name is not None:
        export_df = dataframe_for_excel(df, index_name)
    else:
        export_df = df.copy().reset_index(drop=True)

    if export_df.empty:
        worksheet.write(start_row, 0, "No data available.")
        return start_row + 3

    export_df.to_excel(
        writer,
        sheet_name=safe_name,
        startrow=start_row + 1,
        startcol=0,
        index=False,
        header=False,
    )

    rows, cols = export_df.shape
    first_row = start_row
    last_row = start_row + rows
    last_col = cols - 1
    table_columns = [{"header": str(column)} for column in export_df.columns]
    worksheet.add_table(
        first_row,
        0,
        last_row,
        last_col,
        {
            "columns": table_columns,
            "style": "Table Style Medium 2",
            "autofilter": True,
        },
    )

    worksheet.freeze_panes(first_row + 1, 1)
    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    worksheet.set_margins(left=0.3, right=0.3, top=0.5, bottom=0.5)

    for col_idx, column in enumerate(export_df.columns):
        width = column_width_for_series(export_df[column], str(column))
        fmt = None
        if pd.api.types.is_integer_dtype(export_df[column]):
            fmt = integer_format
        elif pd.api.types.is_float_dtype(export_df[column]):
            fmt = percent_format if "%" in title else number_format
        worksheet.set_column(col_idx, col_idx, width, fmt)

    numeric_cols = [idx for idx, column in enumerate(export_df.columns) if pd.api.types.is_numeric_dtype(export_df[column])]
    if numeric_cols:
        for col_idx in numeric_cols:
            worksheet.conditional_format(
                first_row + 1,
                col_idx,
                last_row,
                col_idx,
                {"type": "top", "value": 1, "format": highlight_format},
            )

    return last_row + 4


def write_excel_report(
    output_xlsx: Path,
    input_csv: Path,
    df: pd.DataFrame,
    desc: pd.DataFrame,
    swh_gev: GEVResult,
    wind_gev: GEVResult,
    swh_sector: pd.DataFrame,
    wind_sector: pd.DataFrame,
    joint_swh_mwd: pd.DataFrame,
    joint_swh_mwp: pd.DataFrame,
    joint_wind_dwi: pd.DataFrame,
) -> None:
    """Write all tabular results to a formatted Excel workbook."""
    summary = pd.DataFrame(
        [
            {"item": "Input file", "value": str(input_csv.name)},
            {"item": "Rows analysed", "value": len(df)},
            {"item": "Start datetime", "value": str(df.index.min())},
            {"item": "End datetime", "value": str(df.index.max())},
            {"item": "Output PDF", "value": "output.pdf"},
            {"item": "Output workbook", "value": "output.xlsx"},
            {"item": "Variables", "value": ", ".join(NUMERIC_COLUMNS)},
        ]
    )

    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        workbook = writer.book
        write_excel_table(writer, workbook, "Summary", summary, "Run summary", start_row=2)
        write_excel_table(writer, workbook, "Descriptive Statistics", desc, "Descriptive statistics", "variable")
        write_excel_table(writer, workbook, "GEV SWH", gev_result_table(swh_gev, "swh"), "GEV: swh, all directions")
        write_excel_table(writer, workbook, "GEV Wind", gev_result_table(wind_gev, "wind"), "GEV: wind, all directions")
        write_excel_table(writer, workbook, "SWH Sector GEV", swh_sector, "GEV: swh by 30-degree mwd sector", "sector")
        write_excel_table(writer, workbook, "Wind Sector GEV", wind_sector, "GEV: wind by 30-degree dwi sector", "sector")
        write_excel_table(writer, workbook, "Joint SWH MWD", joint_swh_mwd, "Joint distribution: swh vs mwd (%)", "swh interval")
        write_excel_table(writer, workbook, "Joint SWH MWP", joint_swh_mwp, "Joint distribution: swh vs mwp (%)", "swh interval")
        write_excel_table(writer, workbook, "Joint Wind DWI", joint_wind_dwi, "Joint distribution: wind vs dwi (%)", "wind interval")


def write_pdf_report(
    pdf_file: Path,
    input_csv: Path,
    desc: pd.DataFrame,
    swh_gev: GEVResult,
    wind_gev: GEVResult,
    swh_sector: pd.DataFrame,
    wind_sector: pd.DataFrame,
    joint_swh_mwd: pd.DataFrame,
    joint_swh_mwp: pd.DataFrame,
    joint_wind_dwi: pd.DataFrame,
    plot_swh_all: str,
    plot_wind_all: str,
    swh_sector_plots: list[tuple[str, str]],
    wind_sector_plots: list[tuple[str, str]],
    plot_swh_rose: str,
    plot_wind_rose: str,
) -> None:
    """Generate the final landscape PDF report."""
    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page()
    pdf.set_font("Courier", "B", 20)
    pdf.cell(0, 10, "ERA5 Wave and Wind Statistical Report", ln=True, align="C")
    pdf.ln(3)
    pdf.set_font("Courier", "", 11)
    pdf.cell(0, 7, f"Input file: {input_csv.name}", ln=True, align="C")
    pdf.cell(0, 7, "Variables: swh, mwp, mwd, wind, dwi, u10, v10", ln=True, align="C")
    pdf.ln(6)

    pdf_print_table(pdf, desc, title="1) Descriptive statistics", decimals=3, size=8)

    add_gev_summary_to_pdf(pdf, "2) GEV: swh, all directions", swh_gev, plot_swh_all, "swh")
    add_gev_summary_to_pdf(pdf, "3) GEV: wind, all directions", wind_gev, plot_wind_all, "wind")

    pdf.add_page()
    pdf_print_table(pdf, swh_sector, title="4) GEV: swh by 30-degree wave-direction sector", decimals=3, size=7)
    add_sector_plots_to_pdf(pdf, "SWH", swh_sector_plots)

    pdf.add_page()
    pdf_print_table(pdf, wind_sector, title="5) GEV: wind by 30-degree wind-direction sector", decimals=3, size=7)
    add_sector_plots_to_pdf(pdf, "Wind", wind_sector_plots)

    pdf.add_page()
    pdf_print_table(pdf, joint_swh_mwd, title="6) Joint distribution: swh vs mwd (%)", decimals=3, size=7)

    pdf.add_page()
    pdf_print_table(pdf, joint_swh_mwp, title="7) Joint distribution: swh vs mwp (%)", decimals=3, size=7)

    pdf.add_page()
    pdf_print_table(pdf, joint_wind_dwi, title="8) Joint distribution: wind vs dwi (%)", decimals=3, size=7)

    pdf.add_page()
    pdf.set_font("Courier", "B", 12)
    pdf.cell(0, 8, "9) Directional rose: swh vs mwd", ln=True)
    pdf.image(plot_swh_rose, x=35, w=170)

    pdf.add_page()
    pdf.set_font("Courier", "B", 12)
    pdf.cell(0, 8, "10) Directional rose: wind vs dwi", ln=True)
    pdf.image(plot_wind_rose, x=35, w=170)

    pdf.output(str(pdf_file))


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(input_csv: Path) -> None:
    """Execute the complete statistical and extreme-value analysis workflow."""
    figures_dir = Path(FIGURES_FOLDER)
    figures_dir.mkdir(exist_ok=True)

    print(f"Reading input data: {input_csv}")
    df = read_input_csv(input_csv)
    print(f"Valid rows: {len(df):,}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    output_xlsx = output_file(input_csv, "output.xlsx")
    pdf_file = output_file(input_csv, "output.pdf")

    print("Computing descriptive statistics...")
    desc = descriptive_statistics(df)

    print("Fitting GEV distributions...")
    swh_maxima = annual_maxima(df["swh"])
    wind_maxima = annual_maxima(df["wind"])
    swh_gev = fit_gev_from_annual_maxima(swh_maxima)
    wind_gev = fit_gev_from_annual_maxima(wind_maxima)

    print("Generating GEV plots...")
    plot_swh_all = str(figures_dir / "swh_gev_all_directions.png")
    plot_wind_all = str(figures_dir / "wind_gev_all_directions.png")
    plot_gev_with_return_lines(swh_maxima, swh_gev, plot_swh_all, "swh", "all directions")
    plot_gev_with_return_lines(wind_maxima, wind_gev, plot_wind_all, "wind", "all directions")

    print("Fitting directional-sector GEV distributions...")
    swh_sector, swh_sector_plots = fit_sector_gev(df, "swh", "mwd")
    wind_sector, wind_sector_plots = fit_sector_gev(df, "wind", "dwi")

    print("Computing joint distributions...")
    swh_bins = make_joint_distribution_bins(df["swh"])
    mwp_bins = make_joint_distribution_bins(df["mwp"])
    wind_bins = make_joint_distribution_bins(df["wind"])
    direction_bins = np.arange(0, 361, 30)

    joint_swh_mwd = make_joint_distribution(df, "swh", "mwd", swh_bins, direction_bins)
    joint_swh_mwp = make_joint_distribution(df, "swh", "mwp", swh_bins, mwp_bins)
    joint_wind_dwi = make_joint_distribution(df, "wind", "dwi", wind_bins, direction_bins)

    print("Generating directional rose plots...")
    plot_swh_rose = str(figures_dir / "swh_mwd_rose.png")
    plot_wind_rose = str(figures_dir / "wind_dwi_rose.png")
    plot_windrose(df, "swh", "mwd", plot_swh_rose, "swh", "mwd")
    plot_windrose(df, "wind", "dwi", plot_wind_rose, "wind", "dwi")

    print(f"Writing Excel workbook: {output_xlsx}")
    write_excel_report(
        output_xlsx,
        input_csv,
        df,
        desc,
        swh_gev,
        wind_gev,
        swh_sector,
        wind_sector,
        joint_swh_mwd,
        joint_swh_mwp,
        joint_wind_dwi,
    )

    print(f"Writing PDF report: {pdf_file}")
    write_pdf_report(
        pdf_file,
        input_csv,
        desc,
        swh_gev,
        wind_gev,
        swh_sector,
        wind_sector,
        joint_swh_mwd,
        joint_swh_mwp,
        joint_wind_dwi,
        plot_swh_all,
        plot_wind_all,
        swh_sector_plots,
        wind_sector_plots,
        plot_swh_rose,
        plot_wind_rose,
    )

    print("Analysis complete.")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ERA5 wave and wind statistical, GEV, Excel and PDF reports."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file with columns datetime,swh,mwp,mwd,wind,dwi,u10,v10.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        run_analysis(args.input_csv)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
