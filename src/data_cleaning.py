"""Create dashboard-ready datasets from preserved official source files.

The raw SIPRI workbook is never modified. Run this module from the repository
root after downloading the workbook described in the README:

    py -m src.data_cleaning
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIPRI_WORKBOOK = ROOT / "data" / "raw" / "sipri" / "SIPRI-Milex-data-1949-2025_v1.2.xlsx"
OUTPUT_PATH = ROOT / "data" / "processed" / "military_expenditure.csv"
WORLD_BANK_GDP = ROOT / "data" / "raw" / "world_bank" / "gdp_current_usd.json"
WORLD_BANK_POPULATION = ROOT / "data" / "raw" / "world_bank" / "population.json"
MACRO_OUTPUT_PATH = ROOT / "data" / "processed" / "macro_economic.csv"

SOURCE_TO_DASHBOARD_COUNTRY = {
    "Spain": "Spain",
    "Morocco": "Morocco",
    "France": "France",
    "Germany": "Germany",
    "Italy": "Italy",
    "United Kingdom": "United Kingdom",
    "United States of America": "United States",
}
WORLD_BANK_CODE_TO_COUNTRY = {
    "ESP": "Spain",
    "MAR": "Morocco",
    "FRA": "France",
    "DEU": "Germany",
    "ITA": "Italy",
    "GBR": "United Kingdom",
    "USA": "United States",
}


def read_sipri_metric(sheet_name: str) -> pd.DataFrame:
    """Read a SIPRI wide-format worksheet and return country/year/value rows."""
    raw = pd.read_excel(SIPRI_WORKBOOK, sheet_name=sheet_name, header=None)
    header_row = next(
        row_index
        for row_index, row in raw.iterrows()
        if "Country" in row.astype(str).tolist()
    )
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = raw.iloc[header_row].tolist()
    year_columns = [
        column
        for column in data.columns
        if isinstance(column, (int, float))
        and pd.notna(column)
        and 1900 <= int(column) <= 2100
    ]
    selected = data[data["Country"].isin(SOURCE_TO_DASHBOARD_COUNTRY)].copy()
    long = selected.melt(
        id_vars="Country",
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )
    long["country"] = long["Country"].map(SOURCE_TO_DASHBOARD_COUNTRY)
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[["country", "year", "value"]].dropna(subset=["year"])


def build_military_expenditure() -> pd.DataFrame:
    """Merge comparable SIPRI metrics into the dashboard schema."""
    expenditure = read_sipri_metric("Constant (2024) US$").rename(
        columns={"value": "military_expenditure_constant_usd_millions"}
    )
    share_gdp = read_sipri_metric("Share of GDP").rename(
        columns={"value": "military_expenditure_pct_gdp"}
    )
    # SIPRI's worksheet stores shares as fractions (for example, 0.021 = 2.1%).
    share_gdp["military_expenditure_pct_gdp"] *= 100
    per_capita = read_sipri_metric("Per capita").rename(
        columns={"value": "military_expenditure_per_capita_usd"}
    )
    dataset = expenditure.merge(share_gdp, on=["country", "year"], how="left").merge(
        per_capita, on=["country", "year"], how="left"
    )
    dataset["military_expenditure_constant_usd"] = (
        dataset.pop("military_expenditure_constant_usd_millions") * 1_000_000
    )
    dataset["source_id"] = "D01"
    dataset["source_dataset"] = "SIPRI Military Expenditure Database v1.2"
    dataset["constant_price_base_year"] = 2024
    return dataset.sort_values(["country", "year"]).reset_index(drop=True)


def read_world_bank_indicator(path: Path, value_column: str) -> pd.DataFrame:
    """Convert an official World Bank API response into a tidy indicator series."""
    with path.open(encoding="utf-8-sig") as source_file:
        payload = json.load(source_file)
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError(f"Unexpected World Bank response structure in {path}.")
    records = pd.DataFrame(payload[1])
    records["country"] = records["countryiso3code"].map(WORLD_BANK_CODE_TO_COUNTRY)
    records["year"] = pd.to_numeric(records["date"], errors="coerce").astype("Int64")
    records[value_column] = pd.to_numeric(records["value"], errors="coerce")
    return records[["country", "year", value_column]].dropna(subset=["country", "year"])


def build_macro_economic() -> pd.DataFrame:
    """Create the GDP and population comparison dataset from World Bank data."""
    gdp = read_world_bank_indicator(WORLD_BANK_GDP, "gdp_current_usd")
    population = read_world_bank_indicator(WORLD_BANK_POPULATION, "population")
    dataset = gdp.merge(population, on=["country", "year"], how="outer")
    dataset["source_id"] = "D04"
    dataset["source_dataset"] = "World Development Indicators"
    return dataset.sort_values(["country", "year"]).reset_index(drop=True)


def main() -> None:
    if not SIPRI_WORKBOOK.exists():
        raise FileNotFoundError(
            f"Raw SIPRI workbook not found: {SIPRI_WORKBOOK}. "
            "Download it before running this pipeline."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_military_expenditure()
    dataset.to_csv(OUTPUT_PATH, index=False)
    if not WORLD_BANK_GDP.exists() or not WORLD_BANK_POPULATION.exists():
        raise FileNotFoundError(
            "World Bank raw responses are missing. Download GDP and population "
            "responses before running this pipeline."
        )
    macro = build_macro_economic()
    macro.to_csv(MACRO_OUTPUT_PATH, index=False)
    print(
        f"Created {OUTPUT_PATH} with {len(dataset)} rows, "
        f"{dataset['country'].nunique()} countries and "
        f"{dataset['year'].min()}-{dataset['year'].max()}."
    )
    print(
        f"Created {MACRO_OUTPUT_PATH} with {len(macro)} rows, "
        f"{macro['country'].nunique()} countries and "
        f"{macro['year'].min()}-{macro['year'].max()}."
    )


if __name__ == "__main__":
    main()
