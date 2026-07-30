"""
wacomm_to_geotiff.py
---------------------
Converts WaComM++ history NetCDF files to GeoTIFF rasters for
visualisation in QGIS with the Temporal Controller.

For each hourly history file, the script extracts the concentration
field (sum over all Copernicus levels within max_depth metres, i.e. the
same quantity used as features in wacomm_dataset.py) and saves it as a
single-band float32 GeoTIFF on the regular lat/lon grid.

The 72 output GeoTIFFs can be loaded in QGIS, given temporal properties,
and animated with the Temporal Controller alongside the sampling point
GeoJSON produced by wacomm_to_geojson.py.

Command-line usage:
    python wacomm_to_geotiff.py <t0> [--output-dir DIR]
                                      [--max-depth N]
                                      [--no-cache]

    t0           : sampling timestamp in yyyymmddZhh00 format
                   (e.g. 20230523Z0800). The 72 hours preceding and
                   including t0 will be converted.

    --output-dir : directory where GeoTIFF files are saved
                   (default: ./geotiff/{t0}/)
    --max-depth  : maximum depth in metres for the vertical sum
                   (default: from config.json)
    --no-cache   : bypass the wacomm_profile cache

Output files:
    {output_dir}/wcm3_{timestamp}.tif   — one file per hour
    {output_dir}/wcm3_{timestamp}.tfw   — world file (for older GIS software)
    {output_dir}/timestamps.txt         — list of timestamps in order

QGIS workflow:
    1. Load all 72 .tif files into QGIS (drag & drop or Add Raster Layer)
    2. For each layer: Layer Properties → Temporal → set mode to
       'Fixed Time Range', start = timestamp, end = timestamp + 1h
    3. Open View → Panels → Temporal Controller
    4. Press Play to animate the 72-hour sequence
    5. Load the sampling point GeoJSON on top

Example:
    python wacomm_to_geotiff.py 20230523Z0800 --output-dir ./geotiff/fusaro/
"""

import sys
import os
import argparse
import numpy as np

# Make wacomm_profile importable from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wacomm_profile import (
    build_history_path,
    shift_timestamp,
    FILL_VALUE,
    COPERNICUS_DEPTHS,
)
from config import N_HOURS, DEFAULT_MAX_DEPTH

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    print("Error: rasterio is required. Install it with: pip install rasterio",
          file=sys.stderr)
    sys.exit(1)

try:
    from netCDF4 import Dataset
except ImportError:
    print("Error: netCDF4 is required. Install it with: pip install netCDF4",
          file=sys.stderr)
    sys.exit(1)


# ── Core extraction and conversion ───────────────────────────────────────────

def extract_surface_concentration(filepath: str,
                                   max_depth: float) -> tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]:
    """
    Reads a history NetCDF file and returns the vertically integrated
    concentration field (sum over all Copernicus levels within max_depth),
    mapped onto the regular lat/lon grid.

    This mirrors the logic in wacomm_profile.get_concentration_matrix():
    - horizontal remapping from curvilinear to regular grid (nearest-neighbour)
    - vertical integration limited to max_depth metres

    Parameters
    ----------
    filepath  : str   — path to the history NetCDF file
    max_depth : float — maximum depth in metres for the vertical sum

    Returns
    -------
    (conc_2d, lats, lons) :
        conc_2d : np.ndarray (n_lat, n_lon), float32
                  Integrated concentration; NaN on land/below bathymetry
        lats    : np.ndarray (n_lat,)  — regular latitude axis
        lons    : np.ndarray (n_lon,)  — regular longitude axis
    """
    try:
        from util.Distributor import Distrib3D
    except ImportError:
        raise ImportError(
            "util.Distributor not found. Make sure the util/ directory from "
            "ccmmma-postpro is present in the same folder as this script."
        )

    nc = Dataset(filepath, "r")
    try:
        lat_rho  = nc.variables["lat_rho"][:]
        lon_rho  = nc.variables["lon_rho"][:]
        s_rho    = nc.variables["s_rho"][:]
        mask_rho = nc.variables["mask_rho"][:]
        h        = nc.variables["h"][:]
        conc_4d  = nc.variables["conc"][:]   # (1, s_rho, eta_rho, eta_xi)
    finally:
        nc.close()

    # Build regular destination grid (identical to postpro-wcm3.py)
    dst_lon = np.linspace(lon_rho.min(), lon_rho.max(), lon_rho.shape[1])
    dst_lat = np.linspace(lat_rho.min(), lat_rho.max(), lat_rho.shape[0])

    # Horizontal remapping + conservative vertical redistribution
    # sigma → 136 Copernicus depth levels in metres
    distributor = Distrib3D(lon_rho, lat_rho, dst_lon, dst_lat,
                            s_rho, mask_rho, h)
    conc_dist = distributor.distrib(conc_4d)
    # conc_dist: (1, 136, n_lat, n_lon)

    depths = np.array(list(COPERNICUS_DEPTHS))   # (136,)
    in_range = depths <= max_depth               # levels within max_depth

    # Sum over levels within max_depth (same as column_sums in wacomm_profile)
    conc_3d = conc_dist[0, :, :, :]             # (136, n_lat, n_lon)
    conc_3d_r = conc_3d[in_range, :, :]         # (d, n_lat, n_lon)

    # Convert fill values to NaN before summing
    conc_3d_r = conc_3d_r.astype(np.float32)
    conc_3d_r[conc_3d_r >= FILL_VALUE * 0.9] = np.nan

    # Sum vertically; cells where all levels are NaN stay NaN
    all_nan = np.all(np.isnan(conc_3d_r), axis=0)
    conc_2d = np.nansum(conc_3d_r, axis=0).astype(np.float32)
    conc_2d[all_nan] = np.nan

    return conc_2d, dst_lat, dst_lon


def save_geotiff(conc_2d: np.ndarray,
                 lats: np.ndarray,
                 lons: np.ndarray,
                 output_path: str) -> None:
    """
    Saves a 2D concentration array as a single-band float32 GeoTIFF
    in WGS84 (EPSG:4326).

    The raster origin is the top-left corner (north-up convention),
    so the latitude axis is flipped before writing.

    Parameters
    ----------
    conc_2d     : np.ndarray (n_lat, n_lon) — concentration values
    lats        : np.ndarray (n_lat,)       — latitude axis (south→north)
    lons        : np.ndarray (n_lon,)       — longitude axis (west→east)
    output_path : str — destination .tif file path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n_lat, n_lon = conc_2d.shape

    # GeoTIFF uses north-up convention: flip the array so row 0 = northernmost
    conc_northup = np.flipud(conc_2d)

    # Affine transform: maps pixel coordinates to geographic coordinates.
    # from_bounds(left, bottom, right, top, width, height)
    transform = from_bounds(
        west   = float(lons.min()),
        south  = float(lats.min()),
        east   = float(lons.max()),
        north  = float(lats.max()),
        width  = n_lon,
        height = n_lat,
    )

    with rasterio.open(
        output_path,
        mode     = "w",
        driver   = "GTiff",
        height   = n_lat,
        width    = n_lon,
        count    = 1,                         # single band
        dtype    = rasterio.float32,
        crs      = CRS.from_epsg(4326),       # WGS84
        transform= transform,
        nodata   = np.nan,
        compress = "lzw",                     # lossless compression
    ) as dst:
        dst.write(conc_northup, 1)            # band 1


def netcdf_to_geotiff(timestamp: str, output_path: str,
                       max_depth: float) -> bool:
    """
    Converts a single history NetCDF file to a GeoTIFF.

    Returns True on success, False if the file is missing or corrupted.
    """
    filepath = build_history_path(timestamp)

    if not os.path.exists(filepath):
        return False

    try:
        conc_2d, lats, lons = extract_surface_concentration(filepath, max_depth)
        save_geotiff(conc_2d, lats, lons, output_path)
        return True
    except Exception as e:
        print(f"  [WARN] {timestamp}: {e}", file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert WaComM++ history NetCDF files to GeoTIFF rasters "
            "for QGIS Temporal Controller visualisation."
        )
    )
    parser.add_argument("t0",
                        help="Sampling timestamp in yyyymmddZhh00 format "
                             "(e.g. 20230523Z0800). The 72 preceding hours "
                             "will be converted.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for GeoTIFF files "
                             "(default: ./geotiff/{t0}/)")
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH,
                        help=f"Maximum depth in metres for the vertical sum "
                             f"(default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the wacomm_profile cache (not used "
                             "directly here but kept for CLI consistency)")
    args = parser.parse_args()

    t0         = args.t0
    max_depth  = args.max_depth
    output_dir = args.output_dir or os.path.join("geotiff", t0)

    # Build list of 72 timestamps (oldest → newest, t0 last)
    timestamps = [
        shift_timestamp(t0, -(N_HOURS - 1 - i))
        for i in range(N_HOURS)
    ]

    os.makedirs(output_dir, exist_ok=True)

    print(f"t0            : {t0}")
    print(f"Hours         : {N_HOURS}  ({timestamps[0]} → {timestamps[-1]})")
    print(f"Max depth     : {max_depth} m")
    print(f"Output dir    : {output_dir}")
    print()

    n_ok      = 0
    n_missing = 0
    n_err     = 0
    written_timestamps = []

    for i, ts in enumerate(timestamps, start=1):
        out_path = os.path.join(output_dir, f"wcm3_{ts}.tif")

        # Skip if GeoTIFF already exists
        if os.path.exists(out_path):
            print(f"  [SKIP] [{i}/{N_HOURS}] {ts} — already exists")
            n_ok += 1
            written_timestamps.append(ts)
            continue

        print(f"  [{i}/{N_HOURS}] {ts} ...", end=" ", flush=True)
        ok = netcdf_to_geotiff(ts, out_path, max_depth)

        if ok:
            print(f"✓  {out_path}")
            n_ok += 1
            written_timestamps.append(ts)
        else:
            filepath = build_history_path(ts)
            if not os.path.exists(filepath):
                print(f"✗  missing history file")
                n_missing += 1
            else:
                print(f"✗  error")
                n_err += 1

    # Write a timestamps.txt listing the converted files in order
    ts_list_path = os.path.join(output_dir, "timestamps.txt")
    with open(ts_list_path, "w") as f:
        for ts in written_timestamps:
            f.write(f"{ts}\n")

    # Print QGIS instructions
    print(f"\n{'='*60}")
    print(f"GeoTIFFs generated : {n_ok}")
    print(f"Missing files      : {n_missing}")
    print(f"Errors             : {n_err}")
    print(f"Output directory   : {os.path.abspath(output_dir)}")
    print(f"Timestamp list     : {ts_list_path}")
    print()
    print("QGIS workflow:")
    print(f"  1. Drag & drop all .tif files from {output_dir} into QGIS")
    print( "  2. For each layer: right-click → Properties → Temporal")
    print( "     → Fixed Time Range → set start/end to the file timestamp")
    print( "  3. View → Panels → Temporal Controller → Play")
    print( "  4. Load your sampling point GeoJSON as a vector layer on top")


if __name__ == "__main__":
    main()