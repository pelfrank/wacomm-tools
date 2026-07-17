"""
wacomm_profile.py
-----------------
Functions for extracting particle concentration data from WaComM++ history files.

Step 1 — vertical profile at a single timestamp:
    python wacomm_profile.py profile <p> <lambda> <t>

Step 2 — matrix (136 Copernicus levels × 72 hours) centred on t_0:
    python wacomm_profile.py matrix <p> <lambda> <t0>

Common arguments:
    p       : latitude  (degrees north, float, e.g. 40.85)
    lambda  : longitude (degrees east,  float, e.g. 14.27)
    t / t0  : timestamp in the format yyyymmddZhh00 (e.g. 20260601Z0600)

History files are looked up at:
    /storage/ccmmma/prometeo/data/opendap/wcm3/d03/history/yyyy/mm/dd/
    wcm3_d03_yyyymmddZhh00.nc

On-disk cache
-------------
get_concentration_profile() and get_concentration_matrix() automatically
save/load results to/from ./cache/ (.npz files) to avoid re-reading and
re-computing on every run. The 'matrix' cache already includes 'column_sums'
(the vector used by 'totals'), so no separate cache is needed for totals.

File naming: {timestamp}_{p|m}_{n_hours}h.npz
    p = profile, m = matrix   (n_hours is always 1 for 'profile')

To disable the cache: pass use_cache=False to the functions, or use
--no-cache from the command line.
"""

import sys
import os
import re
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset

# Make the util/ package from ccmmma-postpro importable.
# Update POSTPRO_UTIL_DIR if the path differs in your environment.
POSTPRO_UTIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "util")
if os.path.dirname(POSTPRO_UTIL_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(POSTPRO_UTIL_DIR))

from util.Distributor import Distrib3D
from util.Interpolator import depths as COPERNICUS_DEPTHS   # 136 depth levels in metres
from config import HISTORY_ROOT, CACHE_DIR, FILL_VALUE


# ── Helper functions ──────────────────────────────────────────────────────────

def build_history_path(t: str) -> str:
    """
    Builds the full path to a history NetCDF file given timestamp t.

    Parameters
    ----------
    t : str
        Timestamp in the format yyyymmddZhh00 (e.g. '20260601Z0600').

    Returns
    -------
    str
        Absolute path to the history NetCDF file.

    Raises
    ------
    ValueError
        If the timestamp format is invalid.
    """
    pattern = r"^(\d{4})(\d{2})(\d{2})Z(\d{2})00$"
    m = re.match(pattern, t)
    if not m:
        raise ValueError(
            f"Invalid timestamp: '{t}'. "
            "Expected format: yyyymmddZhh00 (e.g. 20260601Z0600)"
        )
    yyyy, mm, dd, hh = m.group(1), m.group(2), m.group(3), m.group(4)
    filename = f"wcm3_d03_{t}.nc"
    return f"{HISTORY_ROOT}/{yyyy}/{mm}/{dd}/{filename}"


def find_nearest_rho_point(lat_rho: np.ndarray, lon_rho: np.ndarray,
                            p: float, lam: float) -> tuple[int, int]:
    """
    Finds the (j, i) indices of the RHO point on the curvilinear grid
    nearest to the geographic point (p, lam), using Euclidean distance
    on lat/lon (valid approximation for small to medium domains).

    Parameters
    ----------
    lat_rho : np.ndarray, shape (eta_rho, eta_xi)
        Latitudes of RHO grid points.
    lon_rho : np.ndarray, shape (eta_rho, eta_xi)
        Longitudes of RHO grid points.
    p   : float
        Target latitude (degrees north).
    lam : float
        Target longitude (degrees east).

    Returns
    -------
    (j, i) : tuple[int, int]
        Indices of the nearest RHO point (j = eta_rho axis, i = eta_xi axis).
    """
    dist2 = (lat_rho - p) ** 2 + (lon_rho - lam) ** 2
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(j), int(i)


# ── On-disk cache ─────────────────────────────────────────────────────────────

def _cache_path(t: str, kind: str, n_hours: int = 1,
                cache_dir: str = CACHE_DIR) -> str:
    """
    Builds the cache file path.

    File naming convention: {timestamp}_{kind}_{n_hours}h.npz
        kind = 'p' for profile, 'm' for matrix

    Parameters
    ----------
    t         : str — timestamp (t for profile, t0 for matrix)
    kind      : str — 'p' or 'm'
    n_hours   : int — number of hours (1 for profile, 72 by default for matrix)
    cache_dir : str — cache directory

    Returns
    -------
    str — full path to the .npz file
    """
    filename = f"{t}_{kind}_{n_hours}h.npz"
    return os.path.join(cache_dir, filename)


def _save_cache(path: str, result: dict) -> None:
    """
    Saves a result dict to a .npz file, converting lists
    (e.g. 'depths', 'timestamps') to numpy-compatible arrays.
    Scalar values (float/int) are saved as 0-d arrays.

    Note: the 'file' key (present in get_concentration_profile results)
    is renamed to '_file' inside the .npz file because it conflicts with
    the positional 'file' parameter of np.savez_compressed.
    _load_cache() restores it correctly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {}
    for key, value in result.items():
        npz_key = "_file" if key == "file" else key
        arrays[npz_key] = np.asarray(value)
    try:
        np.savez_compressed(path, **arrays)
    except OSError as e:
        # The cache is an optimisation only: if saving fails
        # (e.g. disk full, permissions) it must not block execution.
        print(f"Warning: could not save cache to {path}: {e}",
              file=sys.stderr)


def _load_cache(path: str) -> dict:
    """
    Loads a result dict from a .npz file previously saved by _save_cache().
    Restores known fields to their original Python types
    (string lists, float/int scalars).

    Raises
    ------
    FileNotFoundError — if the cache file does not exist
    """
    with np.load(path, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}

    # Restore the 'file' key that was renamed to '_file' during saving
    if "_file" in result:
        result["file"] = result.pop("_file")

    # Restore scalar/list types from 0-d numpy containers
    scalar_keys = ("lat_found", "lon_found", "lat_idx", "lon_idx", "file")
    for key in scalar_keys:
        if key in result:
            result[key] = result[key].item()

    list_keys = ("timestamps", "missing_timestamps", "depths")
    for key in list_keys:
        if key in result:
            result[key] = result[key].tolist()

    return result


def get_concentration_profile(p: float, lam: float, t: str,
                              use_cache: bool = True,
                              cache_dir: str = CACHE_DIR) -> dict:
    """
    Extracts the vertical concentration profile over the 136 Copernicus
    physical depth levels (in metres) for geographic point (p, lam) at time t.

    The pipeline mirrors postpro-wcm3.py:
      1. Reads from the history file: lat_rho, lon_rho, s_rho, mask_rho, h, conc
      2. Builds the regular destination grid (linspace over lat/lon)
      3. Applies Distrib3D.distrib() which:
           a) remaps horizontally level by level (nearest-neighbour)
              from the curvilinear source grid to the regular grid
           b) redistributes vertically in a conservative manner
              from sigma s_rho levels to the 136 physical Copernicus levels (metres)
      4. Finds the grid point nearest to (p, lam) on the resulting regular grid
      5. Returns the vertical profile (136,) at that point

    Parameters
    ----------
    p         : float — latitude of the point of interest (degrees north)
    lam       : float — longitude of the point of interest (degrees east)
    t         : str   — timestamp in the format yyyymmddZhh00 (e.g. '20260601Z0600')
    use_cache : bool  — if True (default), reads/writes the result from/to cache
    cache_dir : str   — cache directory (default ./cache/)

    Returns
    -------
    dict with keys:
        - 'conc'      : np.ndarray (136,) — concentration per Copernicus level;
                        NaN on levels below local bathymetry or on land
        - 'depths'    : list[float] (136,) — Copernicus depths in metres
        - 'lat_found' : float — latitude of the point on the regular grid
        - 'lon_found' : float — longitude of the point on the regular grid
        - 'lat_idx'   : int   — latitude index on the regular grid
        - 'lon_idx'   : int   — longitude index on the regular grid
        - 'file'      : str   — path of the history file read

    Raises
    ------
    FileNotFoundError  — history file not found
    ValueError         — invalid timestamp
    """
    cache_file = _cache_path(t, "p", n_hours=1, cache_dir=cache_dir)
    if use_cache and os.path.exists(cache_file):
        return _load_cache(cache_file)

    filepath = build_history_path(t)

    try:
        nc = Dataset(filepath, "r")
    except FileNotFoundError:
        raise FileNotFoundError(f"History file not found: {filepath}")

    try:
        # 1. Read variables from the history file
        lat_rho  = nc.variables["lat_rho"][:]   # (eta_rho, eta_xi)
        lon_rho  = nc.variables["lon_rho"][:]   # (eta_rho, eta_xi)
        s_rho    = nc.variables["s_rho"][:]     # (30,)
        mask_rho = nc.variables["mask_rho"][:]  # (eta_rho, eta_xi)
        h        = nc.variables["h"][:]         # (eta_rho, eta_xi)
        conc_4d  = nc.variables["conc"][:]      # (1, 30, eta_rho, eta_xi)
    finally:
        nc.close()

    # 2. Build the regular destination grid (identical to postpro-wcm3.py)
    dst_lon = np.linspace(lon_rho.min(), lon_rho.max(), lon_rho.shape[1])
    dst_lat = np.linspace(lat_rho.min(), lat_rho.max(), lat_rho.shape[0])

    # 3. Apply Distrib3D: horizontal remapping + conservative vertical
    #    redistribution sigma → 136 Copernicus depth levels in metres
    distributor = Distrib3D(lon_rho, lat_rho, dst_lon, dst_lat,
                            s_rho, mask_rho, h)
    conc_dist = distributor.distrib(conc_4d)
    # conc_dist: (1, 136, len(dst_lat), len(dst_lon))

    # 4. Find the nearest point to (p, lam) on the 1D regular grid
    lat_idx = int(np.argmin(np.abs(dst_lat - p)))
    lon_idx = int(np.argmin(np.abs(dst_lon - lam)))

    # 5. Extract the vertical profile (136,) at that point
    profile = np.array(conc_dist[0, :, lat_idx, lon_idx], dtype=np.float64)
    # Convert fill values to NaN
    profile[profile >= FILL_VALUE * 0.9] = np.nan

    result = {
        "conc"      : profile,
        "depths"    : COPERNICUS_DEPTHS,
        "lat_found" : float(dst_lat[lat_idx]),
        "lon_found" : float(dst_lon[lon_idx]),
        "lat_idx"   : lat_idx,
        "lon_idx"   : lon_idx,
        "file"      : filepath,
    }

    #if use_cache:
    _save_cache(cache_file, result)

    return result


def shift_timestamp(t: str, hours: int) -> str:
    """
    Shifts a timestamp by `hours` hours (may be negative).

    Parameters
    ----------
    t     : str  — timestamp in the format yyyymmddZhh00
    hours : int  — hours to add (negative = back in time)

    Returns
    -------
    str — new timestamp in the format yyyymmddZhh00
    """
    pattern = r"^(\d{4})(\d{2})(\d{2})Z(\d{2})00$"
    m = re.match(pattern, t)
    if not m:
        raise ValueError(
            f"Invalid timestamp: '{t}'. "
            "Expected format: yyyymmddZhh00 (e.g. 20260601Z0600)"
        )
    dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                  int(m.group(4)))
    dt_shifted = dt + timedelta(hours=hours)
    return dt_shifted.strftime("%Y%m%dZ%H00")


def get_concentration_matrix(p: float, lam: float, t0: str,
                              n_hours: int = 72,
                              use_cache: bool = True,
                              cache_dir: str = CACHE_DIR) -> dict:
    """
    Builds the concentration matrix (n_levels × n_hours) for the
    `n_hours` hours preceding (and including) t0.

    Element [k, col] of the matrix is the Copernicus level-k concentration
    at hour t0 - (n_hours - 1 - col), i.e.:
        column 0         → t0 - (n_hours-1) h  (oldest hour)
        column n_hours-1 → t0                  (most recent hour)

    For each hour, the same pipeline as postpro-wcm3.py is applied:
      1. Read the history file (lat_rho, lon_rho, s_rho, mask_rho, h, conc)
      2. Build the regular destination grid
      3. Apply Distrib3D (horizontal remapping + conservative vertical
         redistribution sigma → 136 Copernicus depth levels in metres)
      4. Extract the profile at the point nearest to (p, lam) on the regular grid

    Missing files on disk produce a NaN column without interrupting execution;
    they are tracked in 'missing_timestamps'.

    Parameters
    ----------
    p         : float — latitude of the point of interest (degrees north)
    lam       : float — longitude of the point of interest (degrees east)
    t0        : str   — final timestamp (format yyyymmddZhh00)
    n_hours   : int   — number of hours to cover, default 72
    use_cache : bool  — if True (default), reads/writes the result from/to cache
                        (includes 'column_sums', so 'totals' reuses the same
                        cache without recomputing anything)
    cache_dir : str   — cache directory (default ./cache/)

    Returns
    -------
    dict with keys:
        - 'matrix'             : np.ndarray (136, n_hours), float64;
                                 NaN where the file was unavailable or below
                                 local bathymetry
        - 'column_sums'        : np.ndarray (n_hours,), float64 — for each hour,
                                 sum of matrix[:, col] over all 136 levels
                                 (total concentration in the water column).
                                 NaN only if the entire column was NaN (missing file).
        - 'depths'             : list[float] (136,) — Copernicus depths in metres
        - 'timestamps'         : list[str] (n_hours,) — timestamp list,
                                 column 0 = oldest, -1 = t0
        - 'lat_found'          : float — point latitude on the regular grid
        - 'lon_found'          : float — point longitude on the regular grid
        - 'lat_idx'            : int
        - 'lon_idx'            : int
        - 'missing_timestamps' : list[str] — hours for which the file was missing

    Raises
    ------
    RuntimeError
        If no history file is available in the requested time range.
    """
    cache_file = _cache_path(t0, "m", n_hours=n_hours, cache_dir=cache_dir)
    if use_cache and os.path.exists(cache_file):
        return _load_cache(cache_file)

    # Build the ordered list of timestamps (oldest to most recent)
    timestamps = [
        shift_timestamp(t0, -(n_hours - 1 - col))
        for col in range(n_hours)
    ]

    n_levels = len(COPERNICUS_DEPTHS)
    matrix   = np.full((n_levels, n_hours), np.nan, dtype=np.float64)
    missing  = []

    # Variables for the found point (set on the first available file)
    lat_found, lon_found, lat_idx, lon_idx = None, None, None, None

    # For each hour: apply the full Interpolator + Distributor pipeline
    for col, ts in enumerate(timestamps):
        try:
            filepath = build_history_path(ts)
            nc = Dataset(filepath, "r")
            lat_rho  = nc.variables["lat_rho"][:]
            lon_rho  = nc.variables["lon_rho"][:]
            s_rho    = nc.variables["s_rho"][:]
            mask_rho = nc.variables["mask_rho"][:]
            h        = nc.variables["h"][:]
            conc_4d  = nc.variables["conc"][:]
            nc.close()

            # Regular destination grid (identical to postpro-wcm3.py)
            dst_lon = np.linspace(lon_rho.min(), lon_rho.max(), lon_rho.shape[1])
            dst_lat = np.linspace(lat_rho.min(), lat_rho.max(), lat_rho.shape[0])

            # Pipeline: horizontal remapping + vertical redistribution
            # sigma → 136 Copernicus depth levels in metres
            distributor = Distrib3D(lon_rho, lat_rho, dst_lon, dst_lat,
                                    s_rho, mask_rho, h)
            conc_dist = distributor.distrib(conc_4d)
            # conc_dist: (1, 136, len(dst_lat), len(dst_lon))

            # Find the point on the 1D regular grid (only on the first file)
            if lat_idx is None:
                lat_idx   = int(np.argmin(np.abs(dst_lat - p)))
                lon_idx   = int(np.argmin(np.abs(dst_lon - lam)))
                lat_found = float(dst_lat[lat_idx])
                lon_found = float(dst_lon[lon_idx])

            # Extract the vertical profile (136,) and convert fill values to NaN
            profile = np.array(conc_dist[0, :, lat_idx, lon_idx], dtype=np.float64)
            profile[profile >= FILL_VALUE * 0.9] = np.nan
            matrix[:, col] = profile

        except (FileNotFoundError, OSError):
            missing.append(ts)

    if lat_idx is None:
        raise RuntimeError(
            f"No history files available in the range "
            f"{timestamps[0]} — {timestamps[-1]}"
        )

    # Vector (n_hours,): sum of concentrations over all levels for each hour.
    # column_sums[col] = sum(matrix[:, col]), ignoring NaN.
    # If ALL levels for a given hour are NaN (e.g. missing file), the sum is
    # NaN (not 0), to distinguish "no data" from "zero particles".
    all_nan_cols = np.all(np.isnan(matrix), axis=0)
    column_sums = np.nansum(matrix, axis=0)
    column_sums[all_nan_cols] = np.nan

    result = {
        "matrix"             : matrix,
        "column_sums"        : column_sums,
        "depths"             : COPERNICUS_DEPTHS,
        "timestamps"         : timestamps,
        "lat_found"          : lat_found,
        "lon_found"          : lon_found,
        "lat_idx"            : lat_idx,
        "lon_idx"            : lon_idx,
        "missing_timestamps" : missing,
    }

    if use_cache:
        _save_cache(cache_file, result)

    return result


# ── Command-line interface ────────────────────────────────────────────────────

def _print_profile(result: dict, p: float, lam: float, t: str) -> None:
    """Prints the concentration profile to stdout in a readable format."""
    print(f"\nFile read        : {result['file']}")
    print(f"Requested point  : lat={p:.4f}°N  lon={lam:.4f}°E")
    print(f"Found point      : lat={result['lat_found']:.4f}°N  "
          f"lon={result['lon_found']:.4f}°E  "
          f"(lat_idx={result['lat_idx']}, lon_idx={result['lon_idx']})")
    print(f"\n{'Level':>8}  {'Depth (m)':>16}  {'conc':>12}")
    print("-" * 44)
    for k, (depth_m, c) in enumerate(zip(result["depths"], result["conc"])):
        c_str = f"{c:12.4f}" if not np.isnan(c) else "        (land)"
        print(f"{k:>8d}  {depth_m:>16.4f}  {c_str}")
    print()


def _print_matrix_summary(result: dict, p: float, lam: float) -> None:
    """Prints the matrix summary and full contents (136 rows × 72 columns)."""
    mat    = result["matrix"]
    ts     = result["timestamps"]
    depths = result["depths"]

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\nRequested point  : lat={p:.4f}°N  lon={lam:.4f}°E")
    print(f"Found point      : lat={result['lat_found']:.4f}°N  "
          f"lon={result['lon_found']:.4f}°E  "
          f"(lat_idx={result['lat_idx']}, lon_idx={result['lon_idx']})")
    print(f"Time range       : {ts[0]}  →  {ts[-1]}")
    print(f"Matrix shape     : {mat.shape[0]} levels × {mat.shape[1]} hours")
    print(f"min={np.nanmin(mat):.2f}  max={np.nanmax(mat):.2f}  "
          f"NaN={int(np.isnan(mat).sum())}")

    if result["missing_timestamps"]:
        print(f"Missing files ({len(result['missing_timestamps'])}):",
              ", ".join(result["missing_timestamps"]))

    # ── Column headers (timestamps) ──────────────────────────────────────────
    print()
    col_w = 5   # value column width
    lbl_w = 12  # level label column width (e.g. "136.4m")

    header_date = " " * (lbl_w + 2)
    header_hour = " " * (lbl_w + 2)
    for col, stamp in enumerate(ts):
        date_part = stamp[:8]
        hour_part = stamp[9:11]
        if col == 0 or ts[col][:8] != ts[col - 1][:8]:
            header_date += f"{date_part:>{col_w}}"
        else:
            header_date += " " * col_w
        header_hour += f"{hour_part:>{col_w}}"

    print(header_date)
    print(header_hour)
    print("-" * (lbl_w + 2 + col_w * mat.shape[1]))

    # ── Matrix rows (one Copernicus level per row) ───────────────────────────
    for k in range(mat.shape[0]):
        depth_lbl = f"{depths[k]:.1f}m"
        row_str = f"{depth_lbl:>{lbl_w}}  "
        for col in range(mat.shape[1]):
            v = mat[k, col]
            if np.isnan(v):
                row_str += f"{'NaN':>{col_w}}"
            else:
                row_str += f"{int(v):>{col_w}}"
        print(row_str)

    # ── Column-sum vector (total concentration per hour) ─────────────────────
    print("-" * (lbl_w + 2 + col_w * mat.shape[1]))
    sums = result["column_sums"]
    sum_row = f"{'TOT':>{lbl_w}}  "
    for col in range(mat.shape[1]):
        v = sums[col]
        if np.isnan(v):
            sum_row += f"{'NaN':>{col_w}}"
        else:
            sum_row += f"{int(v):>{col_w}}"
    print(sum_row)
    print()


if __name__ == "__main__":
    USAGE = (
        "Usage:\n"
        "  Single profile : python wacomm_profile.py profile <p> <lambda> <t>  [--no-cache]\n"
        "  72-hour matrix : python wacomm_profile.py matrix  <p> <lambda> <t0> [--no-cache]\n"
        "\n"
        "  p / lambda : latitude and longitude (e.g. 40.85  14.27)\n"
        "  t / t0     : timestamp in the format yyyymmddZhh00 (e.g. 20260601Z0600)\n"
        "  --no-cache : bypass the on-disk cache (./cache/)\n"
    )

    raw_args  = sys.argv[1:]
    use_cache = "--no-cache" not in raw_args
    args      = [a for a in raw_args if a != "--no-cache"]

    if len(args) != 4:
        print(USAGE)
        sys.exit(1)

    subcommand = args[0]
    if subcommand not in ("profile", "matrix"):
        print(f"Unknown subcommand: '{subcommand}'\n")
        print(USAGE)
        sys.exit(1)

    try:
        p_arg   = float(args[1])
        lam_arg = float(args[2])
        t_arg   = args[3]

        if subcommand == "profile":
            result = get_concentration_profile(p_arg, lam_arg, t_arg,
                                               use_cache=use_cache)
            _print_profile(result, p_arg, lam_arg, t_arg)

        else:  # matrix
            result = get_concentration_matrix(p_arg, lam_arg, t_arg,
                                              use_cache=use_cache)
            _print_matrix_summary(result, p_arg, lam_arg)

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)