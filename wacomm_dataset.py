"""
wacomm_dataset.py
-----------------
Builds ML training dataset samples by combining:
  - XLS/CSV file with IZS analytical results (E. coli on Mytilus)
  - GeoJSON file of mussel farming zones (banchi)
  - WaComM++ history NetCDF files (via wacomm_profile.py)

For each valid sampling event, produces two CSV files:
  - {scheda}_{t0}.csv         — 72 features (hourly column sums) + label
  - {scheda}_{t0}_matrix.csv  — matrix (d × 72) where d = Copernicus levels
                                within max_depth metres

For plot generation use wacomm_plot.py.

Command-line usage:
    python wacomm_dataset.py <izs_file> <banchi_geojson>
                             [--output-dir DIR] [--max-depth N] [--no-cache]

    izs_file      : XLS or CSV file with IZS analytical results
    banchi_geojson: GeoJSON file of mussel farming zones
    --output-dir  : output directory (default: ./dataset/)
    --max-depth N : maximum depth for the matrix (default: from config.json)
    --no-cache    : disable the history file cache

Example:
    python wacomm_dataset.py esiti_2023.xls banchi.geojson --output-dir ./out/
"""

import sys
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, time

# Make wacomm_profile importable from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wacomm_profile import get_concentration_matrix

from config import (
    N_HOURS,
    DEFAULT_MAX_DEPTH,
    BACTERIA,
    SPECIES,
    TARGET_BINS,
    SAMPLING_HOUR_LOCAL,
    TIMEZONE,
)

# ── Constants derived from configuration ─────────────────────────────────────

ROME_TZ       = pytz.timezone(TIMEZONE)
TARGET_LABELS = [0, 1, 2, 3]


# ── XLS/CSV parsing of IZS results ───────────────────────────────────────────

def _extract_outcome(value) -> int | None:
    """
    Extracts the numeric value from an IZS analytical result.
    E.g.: '290' -> 290, '830 >' -> 830, 'NEGATIVO' -> None
    Replicates the logic of extract_outcome() in talco's routes.py.
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value and value[0].isdigit():
            return int(re.sub(r"\D", "", value))
    return None


def load_izs(filepath: str) -> pd.DataFrame:
    """
    Loads and filters the IZS XLS/CSV results file.

    Applies talco's filters:
      - only rows with PARAMETRO/ANALITA in BACTERIA
      - only rows with MATRICE in SPECIES
      - only numeric outcomes

    Aggregates by NUMERO SCHEDA taking the max outcome
    (same behaviour as uploadMeasurements in talco's routes.py).

    Returns a DataFrame with columns:
        scheda, year, date, sito, lat, lon, outcome, target, t0
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    df = df[
        df["PARAMETRO/ANALITA"].isin(BACTERIA) &
        df["MATRICE"].isin(SPECIES)
    ].copy()

    df["outcome"] = df["ESITO"].apply(_extract_outcome)
    df = df[df["outcome"].notna()].copy()

    if df.empty:
        raise ValueError(
            "No valid rows found after E. coli / Mytilus filter."
        )

    df_agg = df.groupby("NUMERO SCHEDA").agg(
        year    = ("ANNO ACCETTAZIONE", "first"),
        date    = ("DATA PRELIEVO",     "first"),
        sito    = ("SITO",              "first"),
        lat     = ("LATITUDINE",        "first"),
        lon     = ("LONGITUDINE",       "first"),
        outcome = ("outcome",           "max"),
    ).reset_index().rename(columns={"NUMERO SCHEDA": "scheda"})

    df_agg["sito"] = df_agg["sito"].str.strip().str.upper()

    def _to_utc(row) -> datetime:
        d = row["date"]
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime()
        dt_local = datetime.combine(d.date(), time(SAMPLING_HOUR_LOCAL, 0, 0))
        dt_local = ROME_TZ.localize(dt_local)
        return dt_local.astimezone(pytz.utc)

    df_agg["date_utc"] = df_agg.apply(_to_utc, axis=1)
    df_agg["t0"] = df_agg["date_utc"].apply(
        lambda d: d.strftime("%Y%m%dZ%H00")
    )

    df_agg["target"] = pd.cut(
        df_agg["outcome"],
        bins  = [-1] + TARGET_BINS + [float("inf")],
        labels= TARGET_LABELS,
        right = True,
    ).astype(int)

    return df_agg


# ── GeoJSON banchi parsing ────────────────────────────────────────────────────

def load_banchi(filepath: str) -> dict:
    """
    Loads the GeoJSON banchi file and returns a dictionary
        { bank_name_upper: feature }

    Matching with the CSV is done by name (DENOMINAZI field in GeoJSON
    vs SITO field in the CSV), as talco does.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        gj = json.load(f)

    return {
        feat["properties"]["DENOMINAZI"].upper().strip(): feat
        for feat in gj["features"]
    }


# ── Sample building ───────────────────────────────────────────────────────────

def build_sample(row: pd.Series, use_cache: bool = True) -> dict | None:
    """
    For a single IZS sampling event, computes the N_HOURS timeseries
    by calling get_concentration_matrix() and returns a dictionary with
    features + label + metadata + raw data for plotting.

    Returns None if the computation fails entirely.

    The dict contains:
      - metadata: scheda, year, date_utc, t0, sito, lat, lon, outcome, target
      - 72×1 features: h_-71 … h_+00  (column_sums, sum over entire column)
      - raw data (prefix '_', used by wacomm_plot.py):
          _timestamps, _column_sums, _matrix, _depths, _missing
    """
    try:
        result = get_concentration_matrix(
            p         = float(row["lat"]),
            lam       = float(row["lon"]),
            t0        = str(row["t0"]),
            n_hours   = N_HOURS,
            use_cache = use_cache,
        )
    except Exception as e:
        print(f"  [WARN] Could not compute timeseries for {row['scheda']}: {e}",
              file=sys.stderr)
        return None

    column_sums = result["column_sums"]
    timestamps  = result["timestamps"]

    sample = {
        "scheda"   : row["scheda"],
        "year"     : int(row["year"]),
        "date_utc" : str(row["date_utc"]),
        "t0"       : row["t0"],
        "sito"     : row["sito"],
        "lat"      : float(row["lat"]),
        "lon"      : float(row["lon"]),
        "outcome"  : int(row["outcome"]),
        "target"   : int(row["target"]),
    }

    # 72×1 features: full water-column sum for each hour
    for i, ts in enumerate(timestamps):
        hrel = i - (N_HOURS - 1)
        val  = column_sums[i]
        sample[f"h_{hrel:+03d}"] = float(val) if not np.isnan(val) else None

    # Raw data for plotting (not included in CSV output)
    sample["_timestamps"]  = timestamps
    sample["_column_sums"] = column_sums
    sample["_matrix"]      = result["matrix"]
    sample["_depths"]      = result["depths"]
    sample["_missing"]     = result.get("missing_timestamps", [])

    return sample


# ── Save sample CSV (72×1 → MPN) ─────────────────────────────────────────────

def save_sample_csv(sample: dict, output_dir: str) -> str:
    """
    Saves the 72×1 sample as a CSV file.

    Columns: 9 metadata + 72 features (h_-71 … h_+00) + target label
    Filename: {scheda}_{t0}.csv

    Features represent the total concentration in the water column
    (sum over all Copernicus levels) for each of the 72 hours.

    Returns the path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_scheda = sample["scheda"].replace("/", "_").replace("\\", "_")
    filepath    = os.path.join(output_dir, f"{safe_scheda}_{sample['t0']}.csv")

    row = {k: v for k, v in sample.items() if not k.startswith("_")}
    pd.DataFrame([row]).to_csv(filepath, index=False)
    return filepath


# ── Save matrix CSV (72×d → MPN) ─────────────────────────────────────────────

def save_matrix_csv(sample: dict, output_dir: str,
                    max_depth: float = DEFAULT_MAX_DEPTH) -> str:
    """
    Saves the per-level concentration matrix as a CSV file.

    Rows    = Copernicus levels with depth ≤ max_depth (d rows)
    Columns = 72 hours (h_-71 … h_+00)

    Unlike the sample CSV (72×1, full column sum), each row here represents
    a single depth level, allowing the ML model to distinguish the
    contribution of each level.

    Filename: {scheda}_{t0}_matrix.csv

    Returns the path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_scheda = sample["scheda"].replace("/", "_").replace("\\", "_")
    filepath    = os.path.join(output_dir,
                               f"{safe_scheda}_{sample['t0']}_matrix.csv")

    matrix     = np.array(sample["_matrix"])    # (136, 72)
    depths     = np.array(sample["_depths"])    # (136,) in metres
    timestamps = sample["_timestamps"]          # (72,)

    # Keep only levels within max_depth
    in_range = depths <= max_depth
    matrix_r = matrix[in_range, :]
    depths_r = depths[in_range]

    col_labels = [
        f"h_{i-(N_HOURS-1):+03d}_{ts}"
        for i, ts in enumerate(timestamps)
    ]
    df_mat = pd.DataFrame(
        matrix_r,
        index  = [f"{d:.2f}m" for d in depths_r],
        columns= col_labels,
    )
    df_mat.index.name = "depth_m"
    df_mat.to_csv(filepath)
    return filepath


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build ML dataset CSV samples from IZS results + WaComM history files"
    )
    parser.add_argument("izs_file",       help="XLS/CSV file with IZS analytical results")
    parser.add_argument("banchi_geojson", help="GeoJSON file of mussel farming zones")
    parser.add_argument("--output-dir",  default="./dataset",
                        help="Output directory (default: ./dataset/)")
    parser.add_argument("--max-depth",   type=float, default=DEFAULT_MAX_DEPTH,
                        help=f"Maximum depth for the matrix in metres "
                             f"(default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Disable the history file cache")
    args = parser.parse_args()

    use_cache  = not args.no_cache
    output_dir = args.output_dir
    max_depth  = args.max_depth

    # 1. Load IZS results
    print(f"Loading IZS file: {args.izs_file}")
    df_izs = load_izs(args.izs_file)
    print(f"  Valid samples: {len(df_izs)}")

    # 2. Load banchi
    print(f"Loading banchi: {args.banchi_geojson}")
    bank_map = load_banchi(args.banchi_geojson)
    print(f"  Banchi loaded: {len(bank_map)}")

    # 3. Filter samples by site↔banco name match
    df_izs = df_izs[df_izs["sito"].isin(bank_map)].copy()
    print(f"  Samples with site in GeoJSON: {len(df_izs)}")

    if df_izs.empty:
        print("No samples to process. Exiting.")
        sys.exit(0)

    # 4. For each sample: generate the two CSV files
    os.makedirs(output_dir, exist_ok=True)
    n_ok = 0
    n_err = 0

    for idx, row in df_izs.iterrows():
        print(f"\n[{idx+1}/{len(df_izs)}] {row['scheda']}  "
              f"site={row['sito']}  t0={row['t0']}  "
              f"outcome={int(row['outcome'])}  target={row['target']}")

        sample = build_sample(row, use_cache=use_cache)
        if sample is None:
            n_err += 1
            continue

        csv_72x1 = save_sample_csv(sample, output_dir)
        print(f"  72×1 CSV   → {csv_72x1}")

        csv_72xd = save_matrix_csv(sample, output_dir, max_depth=max_depth)
        d = (np.array(sample["_depths"]) <= max_depth).sum()
        print(f"  72×{d} CSV  → {csv_72xd}")

        n_ok += 1

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"Samples processed successfully : {n_ok}")
    print(f"Samples with errors            : {n_err}")
    print(f"Output directory               : {os.path.abspath(output_dir)}")
    print(f"To generate plots use          : wacomm_plot.py dataset ...")


if __name__ == "__main__":
    main()