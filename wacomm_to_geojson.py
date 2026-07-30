"""
wacomm_to_geojson.py
--------------------
Converts WaComM++ dataset CSV samples (produced by wacomm_dataset.py)
to GeoJSON files for loading in QGIS or any GIS application.

Each sample becomes a GeoJSON Point feature with all metadata fields
as properties (scheda, year, date_utc, t0, sito, lat, lon, outcome, target).
The 72 hourly features (h_-71 … h_+00) are excluded as they are not
useful for spatial visualisation.

Three operating modes:

  1. single   — one input CSV  → one output GeoJSON (single feature)
  2. multiple — one input dir  → one GeoJSON per CSV found in the directory
  3. merged   — one input dir  → one GeoJSON containing all samples

Command-line usage:
    python wacomm_to_geojson.py single   <sample_csv>  [output.geojson]
    python wacomm_to_geojson.py multiple <dataset_dir> [output_dir]
    python wacomm_to_geojson.py merged   <dataset_dir> [output.geojson]

    sample_csv   : a single CSV file produced by wacomm_dataset.py
    dataset_dir  : directory containing CSV sample files
    output       : optional output path; defaults described below

Defaults:
    single   → same folder as input CSV, same stem + .geojson
    multiple → same folder as input directory
    merged   → same folder as input directory, named 'dataset_merged.geojson'

Examples:
    python wacomm_to_geojson.py single  ./dataset/1043A-50590-B_20230523Z0800.csv
    python wacomm_to_geojson.py multiple ./dataset/2023/
    python wacomm_to_geojson.py merged   ./dataset/2023/ ./dataset/2023_all.geojson
"""

import sys
import os
import json
import argparse
import pandas as pd


# ── Metadata columns to include in the GeoJSON (h_* features excluded) ───────
METADATA_COLS = ["scheda", "year", "date_utc", "t0", "sito",
                 "lat", "lon", "outcome", "target"]


# ── Core conversion functions ─────────────────────────────────────────────────

def csv_to_feature(csv_path: str) -> dict:
    """
    Reads a single sample CSV and returns a GeoJSON Feature dict.

    The geometry is a Point at (lon, lat) following the GeoJSON spec
    (longitude first, then latitude).
    Properties contain all metadata columns; numeric types are preserved.

    Parameters
    ----------
    csv_path : str — path to the sample CSV file

    Returns
    -------
    dict — a GeoJSON Feature object

    Raises
    ------
    ValueError — if the CSV is missing required columns
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    missing = [c for c in METADATA_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing required columns: {missing}"
        )

    row = df.iloc[0]

    # Build properties, converting numpy/pandas scalars to native Python types
    properties = {}
    for col in METADATA_COLS:
        val = row[col]
        if hasattr(val, "item"):          # numpy scalar → Python native
            val = val.item()
        elif pd.isna(val):
            val = None
        properties[col] = val

    return {
        "type"      : "Feature",
        "geometry"  : {
            "type"       : "Point",
            "coordinates": [properties["lon"], properties["lat"]],
        },
        "properties": properties,
    }


def build_feature_collection(features: list[dict]) -> dict:
    """Wraps a list of GeoJSON Feature dicts into a FeatureCollection."""
    return {
        "type"    : "FeatureCollection",
        "features": features,
    }


def write_geojson(data: dict, path: str) -> None:
    """Writes a GeoJSON dict to file with readable indentation."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_sample_csvs(directory: str) -> list[str]:
    """
    Returns a sorted list of sample CSV paths in the given directory.
    Matrix CSVs (*_matrix.csv) are excluded — only sample CSVs are included.
    """
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".csv") and not f.endswith("_matrix.csv")
    )


# ── Mode implementations ──────────────────────────────────────────────────────

def mode_single(csv_path: str, output_path: str | None) -> None:
    """
    Mode 1 — single CSV → single GeoJSON (one Point feature).
    """
    if output_path is None:
        stem        = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(os.path.dirname(csv_path),
                                   f"{stem}.geojson")

    feature    = csv_to_feature(csv_path)
    collection = build_feature_collection([feature])
    write_geojson(collection, output_path)

    scheda = feature["properties"].get("scheda", "?")
    print(f"  {scheda}  →  {output_path}")


def mode_multiple(dataset_dir: str, output_dir: str | None) -> None:
    """
    Mode 2 — directory of CSVs → one GeoJSON per sample.
    """
    if output_dir is None:
        output_dir = dataset_dir

    csvs = find_sample_csvs(dataset_dir)
    if not csvs:
        print(f"No sample CSVs found in: {dataset_dir}")
        return

    print(f"Found {len(csvs)} samples in: {dataset_dir}")
    n_ok = n_err = 0

    for csv_path in csvs:
        stem        = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(output_dir, f"{stem}.geojson")
        try:
            feature    = csv_to_feature(csv_path)
            collection = build_feature_collection([feature])
            write_geojson(collection, output_path)
            scheda = feature["properties"].get("scheda", "?")
            print(f"  ✓  {scheda}  →  {output_path}")
            n_ok += 1
        except Exception as e:
            print(f"  ✗  {os.path.basename(csv_path)}: {e}", file=sys.stderr)
            n_err += 1

    print(f"\nGenerated: {n_ok}  |  Errors: {n_err}")


def mode_merged(dataset_dir: str, output_path: str | None) -> None:
    """
    Mode 3 — directory of CSVs → single merged GeoJSON (all samples).
    """
    if output_path is None:
        output_path = os.path.join(dataset_dir, "dataset_merged.geojson")

    csvs = find_sample_csvs(dataset_dir)
    if not csvs:
        print(f"No sample CSVs found in: {dataset_dir}")
        return

    print(f"Found {len(csvs)} samples in: {dataset_dir}")
    features = []
    n_err    = 0

    for csv_path in csvs:
        try:
            features.append(csv_to_feature(csv_path))
        except Exception as e:
            print(f"  ✗  {os.path.basename(csv_path)}: {e}", file=sys.stderr)
            n_err += 1

    collection = build_feature_collection(features)
    write_geojson(collection, output_path)

    print(f"Merged {len(features)} features  →  {output_path}")
    if n_err:
        print(f"Errors: {n_err}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert WaComM++ dataset CSV samples to GeoJSON for QGIS"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Mode 1: single
    p_single = subparsers.add_parser(
        "single", help="One CSV → one GeoJSON (single feature)"
    )
    p_single.add_argument("csv_path",
                          help="Sample CSV file produced by wacomm_dataset.py")
    p_single.add_argument("output", nargs="?", default=None,
                          help="Output GeoJSON path (default: same folder as CSV)")

    # Mode 2: multiple
    p_multiple = subparsers.add_parser(
        "multiple", help="Directory of CSVs → one GeoJSON per sample"
    )
    p_multiple.add_argument("dataset_dir",
                            help="Directory containing sample CSV files")
    p_multiple.add_argument("output_dir", nargs="?", default=None,
                            help="Output directory (default: same as input)")

    # Mode 3: merged
    p_merged = subparsers.add_parser(
        "merged", help="Directory of CSVs → one merged GeoJSON"
    )
    p_merged.add_argument("dataset_dir",
                          help="Directory containing sample CSV files")
    p_merged.add_argument("output", nargs="?", default=None,
                          help="Output GeoJSON path "
                               "(default: dataset_dir/dataset_merged.geojson)")

    args = parser.parse_args()

    if args.mode == "single":
        print(f"Mode: single  |  input: {args.csv_path}")
        mode_single(args.csv_path, args.output)

    elif args.mode == "multiple":
        print(f"Mode: multiple  |  input: {args.dataset_dir}")
        mode_multiple(args.dataset_dir, args.output_dir)

    else:  # merged
        print(f"Mode: merged  |  input: {args.dataset_dir}")
        mode_merged(args.dataset_dir, args.output)


if __name__ == "__main__":
    main()