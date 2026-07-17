"""
wacomm_history_downloader.py
-----------------------------
Downloads WaComM++ history NetCDF files from the OPeNDAP server for all
sampling events found in an IZS results file.

For each unique t0 timestamp extracted from the IZS file, the script
downloads the 72 hourly history files preceding (and including) t0.
Files already present on disk are silently skipped.

Downloaded files are saved to HISTORY_ROOT (from config.json) following
the same directory structure expected by wacomm_profile.py:
    {HISTORY_ROOT}/yyyy/mm/dd/wcm3_d03_yyyymmddZhh00.nc

Command-line usage:
    python wacomm_history_downloader.py <izs_file> <banchi_geojson>
                                        [--dry-run] [--workers N]

    izs_file      : XLS or CSV file with IZS analytical results
    banchi_geojson: GeoJSON file of mussel farming zones
    --dry-run     : print the list of files to download without downloading
    --workers N   : number of parallel download threads (default: 4)

Example:
    python wacomm_history_downloader.py esiti_2023.xls banchi.geojson
    python wacomm_history_downloader.py esiti_2023.xls banchi.geojson --workers 8
    python wacomm_history_downloader.py esiti_2023.xls banchi.geojson --dry-run
"""

import sys
import os
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# Make config and wacomm_dataset importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HISTORY_ROOT, N_HOURS
from wacomm_dataset import load_izs, load_banchi
from wacomm_profile import shift_timestamp


# ── Configuration ─────────────────────────────────────────────────────────────

# Base URL for direct NetCDF file download via HTTP.
# Files are served by the nginx server at data.meteo.uniparthenope.it/files/
# and are freely accessible without authentication.
# Reference: https://meteo.uniparthenope.it/open-data/
# Verified response: HTTP 200, Content-Type: application/octet-stream (~258 MB per file)
OPENDAP_BASE_URL = (
    "https://data.meteo.uniparthenope.it/files/wcm3/d03/history"
)

# Number of parallel download threads (overridable via --workers)
DEFAULT_WORKERS = 4


# ── URL and path helpers ──────────────────────────────────────────────────────

def _timestamp_to_parts(t: str) -> tuple[str, str, str, str]:
    """
    Splits a timestamp string into (yyyy, mm, dd, hh).
    E.g. '20230418Z0800' → ('2023', '04', '18', '08')
    """
    yyyy = t[0:4]
    mm   = t[4:6]
    dd   = t[6:8]
    hh   = t[9:11]
    return yyyy, mm, dd, hh


def remote_url(t: str) -> str:
    """Returns the full download URL for a given timestamp."""
    yyyy, mm, dd, _ = _timestamp_to_parts(t)
    filename = f"wcm3_d03_{t}.nc"
    return f"{OPENDAP_BASE_URL}/{yyyy}/{mm}/{dd}/{filename}"


def local_path(t: str, history_root: str = HISTORY_ROOT) -> str:
    """
    Returns the local file path for a given timestamp, mirroring the
    directory structure expected by wacomm_profile.py.
    """
    yyyy, mm, dd, _ = _timestamp_to_parts(t)
    filename = f"wcm3_d03_{t}.nc"
    return os.path.join(history_root, yyyy, mm, dd, filename)


# ── Timestamp collection ──────────────────────────────────────────────────────

def collect_timestamps(izs_file: str, banchi_file: str) -> list[str]:
    """
    Reads the IZS file and GeoJSON banchi, applies the same filters as
    wacomm_dataset.py, and returns a sorted list of unique timestamps
    (one per hour) that need to be downloaded.

    For each unique t0 found in the filtered IZS data, the 72 hourly
    timestamps from t0-(N_HOURS-1)h to t0 are included.

    Returns
    -------
    list[str]
        Sorted list of unique yyyymmddZhh00 timestamps.
    """
    df_izs   = load_izs(izs_file)
    bank_map = load_banchi(banchi_file)

    # Apply the same site↔banco filter as wacomm_dataset.py
    df_izs = df_izs[df_izs["sito"].isin(bank_map)].copy()

    if df_izs.empty:
        return []

    # Collect all hourly timestamps for each unique t0
    unique_t0s = df_izs["t0"].unique()
    all_timestamps: set[str] = set()

    for t0 in unique_t0s:
        for h in range(N_HOURS):
            ts = shift_timestamp(t0, -(N_HOURS - 1 - h))
            all_timestamps.add(ts)

    return sorted(all_timestamps)


# ── Download logic ────────────────────────────────────────────────────────────

def download_file(t: str, history_root: str) -> tuple[str, str]:
    """
    Downloads a single history file for timestamp t.

    Returns a tuple (status, message) where status is one of:
        'skipped'    — file already exists on disk
        'downloaded' — file successfully downloaded
        'error'      — download failed (message contains the error)
    """
    dest = local_path(t, history_root)

    # Skip if already present
    if os.path.exists(dest):
        return "skipped", dest

    url = remote_url(t)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"  ↓ {t}  {url}", flush=True)

    try:
        urllib.request.urlretrieve(url, dest)
        return "downloaded", dest
    except urllib.error.HTTPError as e:
        if os.path.exists(dest):
            os.remove(dest)
        return "error", f"{t}: HTTP {e.code} — {e.reason}"
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        return "error", f"{t}: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download WaComM++ history NetCDF files from the OPeNDAP server "
            "for all sampling events in an IZS results file."
        )
    )
    parser.add_argument("izs_file",
                        help="XLS or CSV file with IZS analytical results")
    parser.add_argument("banchi_geojson",
                        help="GeoJSON file of mussel farming zones")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the file list without downloading")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download threads (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    # ── 1. Collect timestamps ─────────────────────────────────────────────────
    print(f"Reading IZS file   : {args.izs_file}")
    print(f"Reading banchi     : {args.banchi_geojson}")
    timestamps = collect_timestamps(args.izs_file, args.banchi_geojson)

    if not timestamps:
        print("No valid samples found after filtering. Exiting.")
        sys.exit(0)

    # Separate already-present from missing files
    to_download = [t for t in timestamps
                   if not os.path.exists(local_path(t, HISTORY_ROOT))]
    already_present = len(timestamps) - len(to_download)

    print(f"\nTotal hourly timestamps needed : {len(timestamps)}")
    print(f"Already on disk (will skip)    : {already_present}")
    print(f"To download                    : {len(to_download)}")
    print(f"Destination root               : {HISTORY_ROOT}")

    if args.dry_run:
        print("\n[DRY RUN] Files that would be downloaded:")
        for t in to_download:
            print(f"  {remote_url(t)}")
            print(f"  → {local_path(t, HISTORY_ROOT)}")
        sys.exit(0)

    if not to_download:
        print("\nAll files already present. Nothing to download.")
        sys.exit(0)

    # ── 2. Download in parallel ───────────────────────────────────────────────
    print(f"\nDownloading with {args.workers} parallel thread(s)...\n")

    n_downloaded = 0
    n_skipped    = 0
    n_errors     = 0
    errors       = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, t, HISTORY_ROOT): t
            for t in to_download
        }
        for i, future in enumerate(as_completed(futures), start=1):
            t = futures[future]
            status, msg = future.result()

            if status == "downloaded":
                n_downloaded += 1
                print(f"  ✓ [{i}/{len(to_download)}] {t}", flush=True)
            elif status == "skipped":
                n_skipped += 1
            else:
                n_errors += 1
                errors.append(msg)
                print(f"  ✗ [{i}/{len(to_download)}] {msg}", file=sys.stderr, flush=True)

    # ── 3. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Downloaded successfully : {n_downloaded}")
    print(f"Skipped (already exist) : {already_present + n_skipped}")
    print(f"Errors                  : {n_errors}")

    if errors:
        print(f"\nFailed downloads:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()