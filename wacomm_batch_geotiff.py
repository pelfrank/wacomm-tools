"""
wacomm_batch_geotiff.py
------------------------
Batch script (runs outside QGIS) that prepares all the data needed for
visualising discarded dataset samples in QGIS:

  For each sample CSV found in the input directory:
    1. Generates 72 hourly GeoTIFFs in a shared folder named after t0
       (if two samples share the same t0, the GeoTIFFs are generated
       only once and reused by both QGIS projects).
    2. Generates a GeoJSON point file for the sampling location.

Run this script first, then run wacomm_qgis_batch_loader.py inside QGIS.

Directory structure produced:
    {geotiff_root}/
    └── {t0}/                        ← shared by all samples with same t0
        ├── wcm3_{ts_-71}.tif
        ├── ...
        └── wcm3_{t0}.tif

    {geojson_dir}/
    └── {scheda}_{t0}.geojson        ← one per sample

Command-line usage:
    python wacomm_batch_geotiff.py <samples_dir>
                                   [--geotiff-root DIR]
                                   [--geojson-dir  DIR]
                                   [--max-depth N]
                                   [--workers N]

    samples_dir    : directory containing the discarded sample CSV files
    --geotiff-root : root directory for GeoTIFF output
                     (default: ./geotiff/)
    --geojson-dir  : directory for GeoJSON output
                     (default: ./geojson/)
    --max-depth N  : maximum depth in metres for the vertical sum
                     (default: from config.json)
    --workers N    : parallel threads for GeoTIFF generation (default: 4)

Example:
    python wacomm_batch_geotiff.py ./dataset/2023/scartati/ \\
        --geotiff-root ./geotiff/ \\
        --geojson-dir  ./geojson/2023/scartati/
"""

import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_MAX_DEPTH, N_HOURS
from wacomm_profile import shift_timestamp, build_history_path
from wacomm_to_geojson import (
    csv_to_feature,
    build_feature_collection,
    write_geojson,
    find_sample_csvs,
)
from wacomm_to_geotiff import netcdf_to_geotiff

DEFAULT_WORKERS = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def geotiff_dir_for_t0(geotiff_root: str, t0: str) -> str:
    """Returns the shared GeoTIFF directory for a given t0."""
    return os.path.join(geotiff_root, t0)


def geojson_path_for_csv(geojson_dir: str, csv_path: str) -> str:
    """Returns the GeoJSON output path for a given sample CSV."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(geojson_dir, f"{stem}.geojson")


def t0_from_csv(csv_path: str) -> str:
    """
    Extracts the t0 timestamp from the CSV filename.
    Filename format: {scheda}_{t0}.csv  e.g. 1043A-50590-B_20230523Z0800.csv
    The t0 is always the last underscore-separated token before .csv.
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return stem.rsplit("_", 1)[-1]


def stem_from_csv(csv_path: str) -> str:
    """Returns filename stem without extension: e.g. '1043A-50590-B_20230523Z0800'."""
    return os.path.splitext(os.path.basename(csv_path))[0]


# ── GeoTIFF generation ────────────────────────────────────────────────────────

def generate_geotiffs_for_t0(t0: str, geotiff_root: str,
                              max_depth: float,
                              workers: int) -> tuple[int, int, int]:
    """
    Generates the 72 hourly GeoTIFFs for a given t0 into the shared
    directory {geotiff_root}/{t0}/.

    Files already present on disk are silently skipped (resume-friendly).

    Returns (n_ok, n_skipped, n_missing) counts.
    """
    out_dir = geotiff_dir_for_t0(geotiff_root, t0)
    os.makedirs(out_dir, exist_ok=True)

    timestamps = [shift_timestamp(t0, -(N_HOURS - 1 - i))
                  for i in range(N_HOURS)]

    # Pre-filter: only timestamps whose GeoTIFF is missing
    to_generate = []
    n_skipped   = 0
    for ts in timestamps:
        out_path = os.path.join(out_dir, f"wcm3_{ts}.tif")
        if os.path.exists(out_path):
            n_skipped += 1
        else:
            to_generate.append(ts)

    if not to_generate:
        return 0, n_skipped, 0

    n_ok      = 0
    n_missing = 0

    def _generate(ts):
        out_path = os.path.join(out_dir, f"wcm3_{ts}.tif")
        ok = netcdf_to_geotiff(ts, out_path, max_depth)
        return ts, ok

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate, ts): ts for ts in to_generate}
        for future in as_completed(futures):
            ts, ok = future.result()
            if ok:
                n_ok += 1
            else:
                n_missing += 1
                print(f"    [MISS] {ts}", flush=True)

    return n_ok, n_skipped, n_missing


# ── GeoJSON generation ────────────────────────────────────────────────────────

def generate_geojson(csv_path: str, geojson_dir: str) -> bool:
    """
    Generates a single-feature GeoJSON for the given sample CSV.
    Returns True on success, False on error.
    """
    out_path = geojson_path_for_csv(geojson_dir, csv_path)
    if os.path.exists(out_path):
        return True   # already present
    try:
        os.makedirs(geojson_dir, exist_ok=True)
        feature    = csv_to_feature(csv_path)
        collection = build_feature_collection([feature])
        write_geojson(collection, out_path)
        return True
    except Exception as e:
        print(f"    [WARN] GeoJSON error for {os.path.basename(csv_path)}: {e}",
              file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate GeoTIFFs and GeoJSONs for discarded "
                    "dataset samples, ready for QGIS visualisation."
    )
    parser.add_argument("samples_dir",
                        help="Directory containing the discarded sample CSV files")
    parser.add_argument("--geotiff-root", default="./geotiff",
                        help="Root directory for GeoTIFF output "
                             "(default: ./geotiff/)")
    parser.add_argument("--geojson-dir", default="./geojson",
                        help="Directory for GeoJSON output "
                             "(default: ./geojson/)")
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH,
                        help=f"Maximum depth in metres (default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download threads (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    # ── 1. Find all sample CSVs ───────────────────────────────────────────────
    csvs = find_sample_csvs(args.samples_dir)
    if not csvs:
        print(f"No sample CSVs found in: {args.samples_dir}")
        sys.exit(0)
    print(f"Found {len(csvs)} sample CSVs in: {args.samples_dir}")

    # ── 2. Group CSVs by t0 to avoid regenerating shared GeoTIFFs ────────────
    t0_to_csvs: dict[str, list[str]] = {}
    for csv_path in csvs:
        t0 = t0_from_csv(csv_path)
        t0_to_csvs.setdefault(t0, []).append(csv_path)

    unique_t0s = sorted(t0_to_csvs)
    print(f"Unique t0 values   : {len(unique_t0s)}")
    print(f"GeoTIFF root       : {os.path.abspath(args.geotiff_root)}")
    print(f"GeoJSON directory  : {os.path.abspath(args.geojson_dir)}")
    print()

    total_ok      = 0
    total_skipped = 0
    total_missing = 0
    total_geojson = 0

    for t0_idx, t0 in enumerate(unique_t0s, start=1):
        t0_csvs = t0_to_csvs[t0]
        print(f"[t0 {t0_idx}/{len(unique_t0s)}] {t0}  "
              f"({len(t0_csvs)} sample{'s' if len(t0_csvs) > 1 else ''})")

        # ── 2a. Generate GeoTIFFs (shared for this t0) ───────────────────
        n_ok, n_skip, n_miss = generate_geotiffs_for_t0(
            t0, args.geotiff_root, args.max_depth, args.workers
        )
        total_ok      += n_ok
        total_skipped += n_skip
        total_missing += n_miss
        print(f"  GeoTIFFs: {n_ok} generated, "
              f"{n_skip} skipped (exist), {n_miss} missing history files")

        # ── 2b. Generate GeoJSON for each sample with this t0 ────────────
        for csv_path in t0_csvs:
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            ok   = generate_geojson(csv_path, args.geojson_dir)
            if ok:
                total_geojson += 1
                print(f"  GeoJSON : ✓  {stem}.geojson")
            else:
                print(f"  GeoJSON : ✗  {stem}")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GeoTIFFs generated    : {total_ok}")
    print(f"GeoTIFFs skipped      : {total_skipped}")
    print(f"Missing history files : {total_missing}")
    print(f"GeoJSONs generated    : {total_geojson}")
    print()
    print("Now run wacomm_qgis_batch_loader.py inside QGIS to create "
          "one project per sample.")


if __name__ == "__main__":
    main()
