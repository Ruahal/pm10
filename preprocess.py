"""Load and validate the Guwahati air-quality dataset (data/GHY2020.xlsx).

The workbook stores the data in six blocks on a single sheet (three stacked
vertically, two side by side). Every block has its own header row with the
columns: City, Year, Location, SO2, NO2, PM10. This module scans the sheet for
those header rows and extracts a tidy DataFrame from all blocks, then runs
validation checks (missing values, duplicates, wrong types, impossible values)
and reports everything it does in a preprocessing summary. No rows are removed
silently.
"""

import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "GHY2020.xlsx")

REQUIRED_COLUMNS = ["Year", "Location", "SO2", "NO2", "PM10"]
EXPECTED_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# Plausible physical ranges for ambient concentrations (µg/m³).
PLAUSIBLE_RANGES = {"SO2": (0, 1000), "NO2": (0, 1000), "PM10": (0, 2000)}


def load_raw_data(path=DATA_PATH):
    """Extract every observation block from the workbook into a raw DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place GHY2020.xlsx in the data/ folder."
        )

    workbook = load_workbook(path, data_only=True)
    records = []
    for worksheet in workbook.worksheets:
        header_positions = [
            (cell.row, cell.column)
            for row in worksheet.iter_rows()
            for cell in row
            if isinstance(cell.value, str) and cell.value.strip().lower() == "year"
        ]
        for header_row, year_col in header_positions:
            row = header_row + 1
            while row <= worksheet.max_row:
                year_value = worksheet.cell(row=row, column=year_col).value
                if year_value is None:
                    break
                records.append(
                    {
                        "City": worksheet.cell(row=row, column=year_col - 1).value,
                        "Year": year_value,
                        "Location": worksheet.cell(row=row, column=year_col + 1).value,
                        "SO2": worksheet.cell(row=row, column=year_col + 2).value,
                        "NO2": worksheet.cell(row=row, column=year_col + 3).value,
                        "PM10": worksheet.cell(row=row, column=year_col + 4).value,
                    }
                )
                row += 1

    if not records:
        raise ValueError(
            "No data blocks found in the workbook. Expected header rows containing 'Year'."
        )
    return pd.DataFrame.from_records(records)


def clean_and_validate(raw_df):
    """Validate and clean the raw data.

    Returns (clean_df, summary) where summary documents every check and every
    row that was removed, with the reason.
    """
    df = raw_df.copy()
    summary = {
        "raw_rows": int(len(df)),
        "missing_values_per_column": {},
        "duplicate_rows_removed": 0,
        "non_numeric_cells_coerced": 0,
        "impossible_value_rows_removed": [],
        "rows_removed_missing_values": 0,
        "rows_removed_total": 0,
        "final_rows": 0,
        "years_found": [],
        "stations_found": [],
        "checks_performed": [
            "Missing value check on all required columns",
            "Exact duplicate row check",
            "Data type check (Year, SO2, NO2, PM10 coerced to numeric)",
            "Impossible value check (negative concentrations, out-of-range years)",
        ],
    }

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' missing from dataset.")

    # 1. Missing values
    missing = df[REQUIRED_COLUMNS].isna().sum()
    summary["missing_values_per_column"] = {col: int(n) for col, n in missing.items()}

    # 2. Duplicate rows
    duplicate_mask = df.duplicated(subset=REQUIRED_COLUMNS, keep="first")
    summary["duplicate_rows_removed"] = int(duplicate_mask.sum())
    df = df[~duplicate_mask]

    # 3. Wrong data types -> coerce to numeric, count what had to be converted
    for column in ["Year", "SO2", "NO2", "PM10"]:
        converted = pd.to_numeric(df[column], errors="coerce")
        newly_missing = int(converted.isna().sum() - df[column].isna().sum())
        if newly_missing > 0:
            summary["non_numeric_cells_coerced"] += newly_missing
        df[column] = converted

    if df["Location"].dtype != object:
        df["Location"] = df["Location"].astype(str)
    df["Location"] = df["Location"].str.strip()

    # 4. Impossible values
    impossible_mask = pd.Series(False, index=df.index)
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        bad = df[column].notna() & ((df[column] < low) | (df[column] > high))
        if bad.any():
            for idx in df.index[bad]:
                summary["impossible_value_rows_removed"].append(
                    {
                        "row": int(idx) + 2,
                        "reason": f"{column}={df.at[idx, column]} outside plausible range [{low}, {high}] µg/m³",
                    }
                )
        impossible_mask |= bad

    bad_year = df["Year"].notna() & ~df["Year"].isin(EXPECTED_YEARS)
    if bad_year.any():
        for idx in df.index[bad_year]:
            summary["impossible_value_rows_removed"].append(
                {"row": int(idx) + 2, "reason": f"Year={df.at[idx, 'Year']} outside expected range 2015-2020"}
            )
    impossible_mask |= bad_year

    df = df[~impossible_mask]

    # 5. Drop rows still missing required values (documented, not silent)
    still_missing = df[REQUIRED_COLUMNS].isna().any(axis=1)
    summary["rows_removed_missing_values"] = int(still_missing.sum())
    df = df[~still_missing]

    df = df[["Year", "Location", "SO2", "NO2", "PM10"]].reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)
    df["SO2"] = df["SO2"].astype(float)
    df["NO2"] = df["NO2"].astype(float)
    df["PM10"] = df["PM10"].astype(float)

    summary["rows_removed_total"] = summary["raw_rows"] - len(df)
    summary["final_rows"] = int(len(df))
    summary["years_found"] = sorted(df["Year"].unique().tolist())
    summary["stations_found"] = sorted(df["Location"].unique().tolist())
    return df, summary


def load_dataset(path=DATA_PATH):
    """Convenience entry point used by training and the Flask app."""
    raw_df = load_raw_data(path)
    return clean_and_validate(raw_df)


if __name__ == "__main__":
    data, prep_summary = load_dataset()
    print("=== Preprocessing summary ===")
    for key, value in prep_summary.items():
        print(f"{key}: {value}")
    print("\nFirst rows:")
    print(data.head(10).to_string(index=False))
