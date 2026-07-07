"""
wacomm_plot.py
--------------
Visualisation of results from wacomm_profile.py and samples
produced by wacomm_dataset.py.

Command-line usage:
    python wacomm_plot.py profile      <p> <lambda> <t>          [output.png] [--print] [--max-depth N] [--no-cache]
    python wacomm_plot.py matrix       <p> <lambda> <t0>         [output.png] [--print] [--max-depth N] [--no-cache]
    python wacomm_plot.py matrix-lines <p> <lambda> <t0>         [output.png] [--print] [--max-depth N] [--no-cache]
    python wacomm_plot.py totals       <p> <lambda> <t0>         [output.png] [--print] [--no-cache]
    python wacomm_plot.py dataset      <sample_csv> <matrix_csv> [output_dir] [--max-depth N]

--print          : also print numerical data to the terminal.
--max-depth N    : maximum depth (m) shown on the Y axis (default from config.json).
--no-cache       : bypass the on-disk cache (./cache/).

The concentration colour scale uses the same levels and colours as the app
(metacharts.json, fields 'clevels' and 'ccolors').
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make wacomm_profile importable from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wacomm_profile import (
    get_concentration_profile,
    get_concentration_matrix,
    COPERNICUS_DEPTHS,
    _print_profile,
    _print_matrix_summary,
)

from config import METACHARTS_PATH, DEFAULT_MAX_DEPTH, PLOT_Y_MAX


# ── Concentration colour scale (shared with the app) ─────────────────────────

def load_concentration_colormap(path: str = METACHARTS_PATH):
    """
    Loads the discrete concentration colour scale from metacharts.json,
    using the same 'clevels' (thresholds) and 'ccolors' (RGBA colours) as the app.

    The clevels represent the upper edges of each colour interval
    (36 levels → 36 colours → 37 boundaries, including 0 as the lower
    boundary of the first interval).

    Parameters
    ----------
    path : str — path to metacharts.json

    Returns
    -------
    (cmap, norm, unit, label) :
        cmap  : matplotlib.colors.ListedColormap — discrete colours
        norm  : matplotlib.colors.BoundaryNorm   — interval boundaries
        unit  : str — unit of measure (e.g. "#")
        label : str — colour bar label (e.g. "Number of Particles")

    Raises
    ------
    FileNotFoundError — if the file does not exist
    KeyError          — if 'clevels' or 'ccolors' fields are missing
    """
    with open(path, "r") as f:
        meta = json.load(f)["meta-chart"]

    clevels = meta["clevels"]
    ccolors = meta["ccolors"]   # RGBA 0-255

    if len(clevels) != len(ccolors):
        raise ValueError(
            f"clevels ({len(clevels)}) and ccolors ({len(ccolors)}) "
            "must have the same length"
        )

    # Colours normalised to 0-1 for ListedColormap
    colors_norm = [[c / 255.0 for c in rgba] for rgba in ccolors]
    cmap = mcolors.ListedColormap(colors_norm)

    # Interval boundaries: 0 as lower boundary, then the clevels.
    # N levels → N+1 boundaries (the last clevel is already the upper
    # boundary of the last colour, so boundaries = [0] + clevels).
    boundaries = [0] + list(clevels)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    unit  = meta.get("unit_bars", "")
    label = meta.get("title_bars", "Concentration")

    return cmap, norm, unit, label


# ── Step 1: vertical profile ──────────────────────────────────────────────────

def plot_profile(result: dict, p: float, lam: float, t: str,
                 save_path: str = None,
                 max_depth: float = DEFAULT_MAX_DEPTH) -> None:
    """
    Vertical concentration profile plot over Copernicus levels
    down to max_depth metres.

    X axis : concentration [# particles]  (zeros excluded / transparent)
    Y axis : depth in metres, LINEAR scale, increasing downward,
             limited to [0, max_depth]
    Marker colour : same discrete scale as the app (metacharts.json)

    Parameters
    ----------
    result    : dict returned by get_concentration_profile()
    p, lam    : coordinates of the requested point
    t         : timestamp
    save_path : if provided, saves to file instead of displaying on screen
    max_depth : maximum Y-axis depth in metres (default 50)
    """
    depths = np.array(result["depths"])            # (136,) in metres
    conc   = np.array(result["conc"], dtype=float) # (136,)

    # Keep only levels within max_depth
    in_range = depths <= max_depth
    depths_r = depths[in_range]
    conc_r   = conc[in_range]

    # Zeros and NaN → not plotted (transparent)
    valid = (~np.isnan(conc_r)) & (conc_r > 0)

    cmap, norm, unit, cbar_label = load_concentration_colormap()

    fig, ax = plt.subplots(figsize=(6, 8))

    # Thin baseline connecting the points (only where data is available)
    ax.plot(conc_r[valid], depths_r[valid],
            color="lightgray", linewidth=1, zorder=1)

    # Markers coloured according to the app's scale
    sc = ax.scatter(conc_r[valid], depths_r[valid],
                    c=conc_r[valid], cmap=cmap, norm=norm,
                    s=40, edgecolors="black", linewidths=0.4, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"{cbar_label} [{unit}]", fontsize=9)

    # Y axis: LINEAR, increasing downward, limited to max_depth
    ax.set_ylim(max_depth, 0)
    ax.set_ylabel("Depth [m] ↓", fontsize=11)
    ax.set_xlabel("Concentration [#]", fontsize=11)
    ax.set_title(
        f"Vertical concentration profile (0–{max_depth:.0f} m)\n"
        f"lat={result['lat_found']:.4f}°N   lon={result['lon_found']:.4f}°E\n"
        f"t = {t}",
        fontsize=11
    )
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotation: valid levels / total (within the shown range)
    ax.text(0.98, 0.02,
            f"{valid.sum()}/{in_range.sum()} valid levels (conc > 0) "
            f"within {max_depth:.0f}m",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Step 2: depth × time matrix ──────────────────────────────────────────────

def plot_matrix(result: dict, p: float, lam: float, t0: str,
                save_path: str = None,
                max_depth: float = DEFAULT_MAX_DEPTH) -> None:
    """
    Heatmap of the concentration matrix, limited to levels within max_depth.

    X axis : hours relative to t0, from -71 (left) to 0 (right = t0)
    Y axis : depth in metres, LINEAR scale, increasing downward,
             limited to [0, max_depth]
    Colour : same discrete scale as the app (metacharts.json);
             0 = transparent

    Parameters
    ----------
    result    : dict returned by get_concentration_matrix()
    p, lam    : coordinates of the requested point
    t0        : final timestamp
    save_path : if provided, saves to file instead of displaying on screen
    max_depth : maximum Y-axis depth in metres (default 50)
    """
    mat_full   = result["matrix"]               # (136, 72)
    timestamps = result["timestamps"]           # list of 72 yyyymmddZhh00 strings
    depths     = np.array(result["depths"])     # (136,) in metres
    n_hours    = mat_full.shape[1]               # 72

    # Keep only levels within max_depth
    in_range = depths <= max_depth
    depths_r = depths[in_range]
    mat      = mat_full[in_range, :]

    # ── X axis: relative hours, from -(n_hours-1) to 0 ───────────────────────
    x_centers = np.arange(-(n_hours - 1), 1)           # [-71, -70, ..., 0]
    x_edges   = np.arange(-(n_hours - 1) - 0.5, 0.6)  # 73 bin edges

    # ── Y axis: depth bin edges (only levels within max_depth) ───────────────
    d = depths_r
    if len(d) >= 2:
        d_edges = np.concatenate([
            [max(0.0, d[0] - (d[1] - d[0]) / 2)],
            (d[:-1] + d[1:]) / 2,
            [d[-1] + (d[-1] - d[-2]) / 2]
        ])
    else:
        d_edges = np.array([0.0, max_depth])

    # ── Concentration 0 → NaN (transparent) ──────────────────────────────────
    mat_plot = mat.copy()
    mat_plot[mat_plot <= 0] = np.nan

    # ── Discrete colour map shared with the app ───────────────────────────────
    cmap, norm, unit, cbar_label = load_concentration_colormap()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))

    im = ax.pcolormesh(
        x_edges, d_edges, mat_plot,
        cmap=cmap, norm=norm, shading="flat"
    )

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(f"{cbar_label} [{unit}]", fontsize=10)

    # ── Y axis: LINEAR, increasing downward, limited to max_depth ────────────
    ax.set_ylim(max_depth, 0)
    ax.set_ylabel("Depth [m] ↓", fontsize=11)

    # ── X axis: negative values every 6 hours + date label on day change ─────
    step = 6
    tick_positions = x_centers[::step]
    # Ensure 0 is always present as the last label
    if 0 not in tick_positions:
        tick_positions = np.append(tick_positions, 0)

    tick_labels = []
    for rel_h in tick_positions:
        col = int(rel_h + (n_hours - 1))        # index in the timestamps list
        stamp = timestamps[col]
        hh    = stamp[9:11]
        # Show day/month only on the first label or when the day changes
        prev_col = col - step
        if col == 0 or (prev_col >= 0 and timestamps[col][:8] != timestamps[prev_col][:8]):
            tick_labels.append(f"{stamp[6:8]}/{stamp[4:6]}\n{hh}:00 ({int(rel_h):+d}h)")
        else:
            tick_labels.append(f"{hh}:00\n({int(rel_h):+d}h)")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Hours relative to t₀ (UTC)", fontsize=11)

    ax.set_title(
        f"Concentration (0–{max_depth:.0f} m)  —  lat={result['lat_found']:.4f}°N   "
        f"lon={result['lon_found']:.4f}°E\n"
        f"t₀ = {t0}  |  range: {timestamps[0]}  →  {timestamps[-1]}",
        fontsize=11
    )
    ax.grid(True, which="major", linestyle="--", alpha=0.3, color="gray")

    # Missing file annotation
    if result["missing_timestamps"]:
        ax.text(0.01, 0.02,
                f"Missing files: {len(result['missing_timestamps'])}",
                transform=ax.transAxes, fontsize=8, color="red")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Step 2b: all depths as overlaid lines ────────────────────────────────────

def plot_matrix_lines(result: dict, p: float, lam: float, t0: str,
                      save_path: str = None,
                      max_depth: float = DEFAULT_MAX_DEPTH) -> None:
    """
    Line chart of concentration over time, one line per Copernicus depth
    level within max_depth (all on the same axes).

    X axis : hours relative to t0, from -71 (left) to 0 (right = t0)
    Y axis : concentration [# particles]
    Line colour : continuous scale based on depth
                  (light = surface, dark = deeper)

    Parameters
    ----------
    result    : dict returned by get_concentration_matrix()
    p, lam    : coordinates of the requested point
    t0        : final timestamp
    save_path : if provided, saves to file instead of displaying on screen
    max_depth : maximum depth of the levels shown, in metres (default 50)
    """
    mat_full   = result["matrix"]               # (136, n_hours)
    timestamps = result["timestamps"]
    depths     = np.array(result["depths"])     # (136,) in metres
    n_hours    = mat_full.shape[1]

    # Keep only levels within max_depth
    in_range = depths <= max_depth
    depths_r = depths[in_range]                 # e.g. (14,) for max_depth=50
    mat      = mat_full[in_range, :]             # (n_levels_r, n_hours)
    n_levels_r = mat.shape[0]

    x = np.arange(-(n_hours - 1), 1)             # [-71, -70, ..., 0]

    # ── Continuous colour map based on depth ─────────────────────────────────
    # Light (surface) → dark (deeper). Normalised over the range of shown
    # levels (not the full 0-136 scale) so differences are visible even with
    # few selected levels.
    cmap = plt.get_cmap("turbo")
    if n_levels_r > 1:
        depth_norm = mcolors.Normalize(vmin=depths_r.min(), vmax=depths_r.max())
    else:
        depth_norm = mcolors.Normalize(vmin=0, vmax=max(depths_r[0], 1))

    def color_for_depth(d):
        t = depth_norm(d)
        return cmap(0.0 + 1.0 * t)

    fig, ax = plt.subplots(figsize=(14, 7))

    for k in range(n_levels_r):
        y = mat[k, :]
        color = color_for_depth(depths_r[k])
        ax.plot(x, y, color=color, linewidth=1.3,
               label=f"{depths_r[k]:.1f} m")

    # ── X axis: same convention as plot_matrix / plot_column_sums ────────────
    step = 6
    tick_positions = list(x[::step])
    if 0 not in tick_positions:
        tick_positions.append(0)

    tick_labels = []
    for rel_h in tick_positions:
        col = int(rel_h + (n_hours - 1))
        stamp = timestamps[col]
        hh    = stamp[9:11]
        prev_col = col - step
        if col == 0 or (prev_col >= 0 and timestamps[col][:8] != timestamps[prev_col][:8]):
            tick_labels.append(f"{stamp[6:8]}/{stamp[4:6]}\n{hh}:00 ({int(rel_h):+d}h)")
        else:
            tick_labels.append(f"{hh}:00\n({int(rel_h):+d}h)")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Hours relative to t₀ (UTC)", fontsize=11)

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Concentration [#]", fontsize=11)
    ax.set_title(
        f"Concentration by depth level (0–{max_depth:.0f} m)\n"
        f"lat={result['lat_found']:.4f}°N   lon={result['lon_found']:.4f}°E\n"
        f"t₀ = {t0}",
        fontsize=11
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legend: one entry per level, ordered by increasing depth ─────────────
    # With max_depth=50 there are roughly 14 entries: manageable in a side column.
    ax.legend(title="Depth", loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, title_fontsize=9, frameon=False)

    # Missing file annotation
    if result["missing_timestamps"]:
        ax.text(0.01, 0.97,
                f"Missing files: {len(result['missing_timestamps'])} "
                "(gaps in the series)",
                transform=ax.transAxes, fontsize=8, color="red", va="top")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Step 3: total water-column concentration time series ─────────────────────

def plot_column_sums(result: dict, p: float, lam: float, t0: str,
                     save_path: str = None) -> None:
    """
    Area chart of the total water-column concentration, hour by hour.

    Uses the 'column_sums' vector already computed by get_concentration_matrix()
    (sum of all 136 levels for each hour).

    X axis : hours relative to t0, from -71 (left) to 0 (right = t0)
    Y axis : total concentration [# particles] (sum over entire column)
    The last point (t0, the sampling hour) is highlighted in red,
    while the rest of the series is drawn as a filled green area,
    similar to a continuous function plot (not a histogram).

    Parameters
    ----------
    result    : dict returned by get_concentration_matrix()
                (must contain the 'column_sums' key)
    p, lam    : coordinates of the requested point
    t0        : final timestamp (sampling hour)
    save_path : if provided, saves to file instead of displaying on screen
    """
    sums       = np.array(result["column_sums"], dtype=float)  # (72,)
    timestamps = result["timestamps"]                          # (72,)
    n_hours    = len(sums)

    x = np.arange(-(n_hours - 1), 1)   # [-71, -70, ..., 0]

    # NaN values (hours with missing files) are not drawn: the line/area
    # breaks at those points, creating a visible gap.
    y = sums.copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Green filled area for all points except the last ─────────────────────
    # (the last point, t0, is highlighted separately in red)
    x_main, y_main = x[:-1], y[:-1]
    x_last, y_last = x[-1],  y[-1]

    ax.fill_between(x_main, 0, y_main, color="#3a9b3a", alpha=0.85,
                    linewidth=0, zorder=1)
    ax.plot(x_main, y_main, color="#2e7d32", linewidth=1.3, zorder=2)

    # ── Last point (t0, sampling hour) highlighted in red ────────────────────
    # Drawn as a small red area/bar bridging the last green value to the
    # final point, to match the reference design.
    if not np.isnan(y_last):
        ax.fill_between([x_main[-1], x_last], 0,
                        [y_main[-1] if not np.isnan(y_main[-1]) else 0, y_last],
                        color="#e53935", alpha=0.95, linewidth=0, zorder=3)
        ax.plot([x_main[-1], x_last],
               [y_main[-1] if not np.isnan(y_main[-1]) else 0, y_last],
               color="#c62828", linewidth=1.3, zorder=4)
        ax.scatter([x_last], [y_last], color="#c62828", s=25, zorder=5)

    # ── X axis: same convention as plot_matrix (hours relative to t0) ────────
    step = 6
    tick_positions = list(x[::step])
    if 0 not in tick_positions:
        tick_positions.append(0)

    tick_labels = []
    for rel_h in tick_positions:
        col = int(rel_h + (n_hours - 1))
        stamp = timestamps[col]
        hh    = stamp[9:11]
        prev_col = col - step
        if col == 0 or (prev_col >= 0 and timestamps[col][:8] != timestamps[prev_col][:8]):
            tick_labels.append(f"{stamp[6:8]}/{stamp[4:6]}\n{hh}:00 ({int(rel_h):+d}h)")
        else:
            tick_labels.append(f"{hh}:00\n({int(rel_h):+d}h)")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Hours relative to t₀ (UTC)", fontsize=11)

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Total concentration [#]", fontsize=11)
    ax.set_title(
        f"Total water-column concentration\n"
        f"lat={result['lat_found']:.4f}°N   lon={result['lon_found']:.4f}°E\n"
        f"t₀ = {t0}  (red bar)",
        fontsize=11
    )
    ax.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Missing file annotation
    if result["missing_timestamps"]:
        ax.text(0.01, 0.95,
                f"Missing files: {len(result['missing_timestamps'])} "
                "(gaps in the series)",
                transform=ax.transAxes, fontsize=8, color="red", va="top")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Dataset sample plots (from wacomm_dataset.py) ────────────────────────────

def plot_sample(sample: dict, save_path: str = None) -> str | None:
    """
    72-hour WaComM concentration time series + IZS value at t₀.

    X axis         : hours relative to t0, from -71 (left) to 0 (right = t0)
    Left Y axis    : total WaComM concentration [#], scale 0–PLOT_Y_MAX
    Right Y axis   : E. coli [MPN/100g], same scale
    Last bar (t₀)  : actual IZS value in red

    Parameters
    ----------
    sample    : dict returned by wacomm_dataset.build_sample()
    save_path : path to the PNG file; if None, displays on screen

    Returns the path of the saved file, or None.
    """
    timestamps  = sample["_timestamps"]
    sums        = np.array(sample["_column_sums"], dtype=float)
    outcome     = sample["outcome"]
    n_hours     = len(sums)
    x           = np.arange(-(n_hours - 1), 1)

    fig, ax = plt.subplots(figsize=(15, 5))

    # Green area for hours from -71 to -1 (WaComM features)
    x_feat, y_feat = x[:-1], sums[:-1]
    ax.fill_between(x_feat, 0, y_feat,
                    where=~np.isnan(y_feat),
                    color="#3a9b3a", alpha=0.85, linewidth=0, zorder=1)
    ax.plot(x_feat, y_feat, color="#2e7d32", linewidth=1.3, zorder=2)

    # Red final segment: from the last WaComM hour to the IZS value at t₀
    y_prev = y_feat[-1] if not np.isnan(y_feat[-1]) else 0.0
    ax.fill_between([x_feat[-1], 0], 0, [y_prev, outcome],
                    color="#e53935", alpha=0.95, linewidth=0, zorder=3)
    ax.plot([x_feat[-1], 0], [y_prev, outcome],
            color="#c62828", linewidth=1.5, zorder=4)

    target_colors = {0: "#4caf50", 1: "#ff9800", 2: "#f44336", 3: "#7b1fa2"}
    ax.scatter([0], [outcome],
               color=target_colors.get(sample["target"], "#c62828"),
               s=80, zorder=6, edgecolors="black", linewidths=0.7,
               label=f"IZS outcome={outcome} CFU/100g (class {sample['target']})")

    # X axis with relative hours
    step = 6
    tick_positions = list(x[::step])
    if 0 not in tick_positions:
        tick_positions.append(0)
    tick_labels = []
    for rel_h in tick_positions:
        col   = int(rel_h + (n_hours - 1))
        stamp = timestamps[col]
        hh    = stamp[9:11]
        prev  = col - step
        if col == 0 or (prev >= 0 and timestamps[col][:8] != timestamps[prev][:8]):
            tick_labels.append(f"{stamp[6:8]}/{stamp[4:6]}\n{hh}:00 ({int(rel_h):+d}h)")
        else:
            tick_labels.append(f"{hh}:00\n({int(rel_h):+d}h)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(x[0], 0)
    ax.set_xlabel("Hours relative to sampling t₀ (UTC)", fontsize=11)

    ax.set_ylim(bottom=0, top=PLOT_Y_MAX if PLOT_Y_MAX is not None else None)
    ax.set_ylabel("Total concentration [#]", fontsize=10)

    ax_r = ax.twinx()
    if PLOT_Y_MAX is not None:
        ax_r.set_ylim(0, PLOT_Y_MAX)
    else:
        ax_r.set_ylim(ax.get_ylim())
    ax_r.set_ylabel("E. coli [MPN/100g]", fontsize=10)

    ax.set_title(
        f"ML dataset sample  —  {sample['sito']}\n"
        f"lat={sample['lat']:.4f}°N   lon={sample['lon']:.4f}°E   "
        f"t₀={sample['t0']}   scheda={sample['scheda']}",
        fontsize=11
    )
    ax.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax_r.spines["top"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)

    if sample["_missing"]:
        ax.text(0.01, 0.95,
                f"Missing hours: {len(sample['_missing'])} (gaps in the series)",
                transform=ax.transAxes, fontsize=8, color="red", va="top")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return save_path


def plot_sample_matrix(sample: dict, save_path: str = None,
                       max_depth: float = DEFAULT_MAX_DEPTH) -> str | None:
    """
    Heatmap of the dataset sample concentration matrix,
    limited to levels within max_depth metres.

    Reuses plot_matrix() by building a compatible dict from the sample.

    Parameters
    ----------
    sample    : dict returned by wacomm_dataset.build_sample()
    save_path : path to the PNG file; if None, displays on screen
    max_depth : maximum Y-axis depth in metres

    Returns the path of the saved file, or None.
    """
    result = {
        "matrix"             : sample["_matrix"],
        "depths"             : sample["_depths"],
        "timestamps"         : sample["_timestamps"],
        "lat_found"          : sample["lat"],
        "lon_found"          : sample["lon"],
        "lat_idx"            : 0,
        "lon_idx"            : 0,
        "missing_timestamps" : sample["_missing"],
    }
    plot_matrix(result, sample["lat"], sample["lon"], sample["t0"],
                save_path=save_path, max_depth=max_depth)
    return save_path


# ── Utilities ─────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str = None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    USAGE = (
        "Usage:\n"
        "  python wacomm_plot.py profile      <p> <lambda> <t>          [output.png] [--print] [--max-depth N] [--no-cache]\n"
        "  python wacomm_plot.py matrix       <p> <lambda> <t0>         [output.png] [--print] [--max-depth N] [--no-cache]\n"
        "  python wacomm_plot.py matrix-lines <p> <lambda> <t0>         [output.png] [--print] [--max-depth N] [--no-cache]\n"
        "  python wacomm_plot.py totals       <p> <lambda> <t0>         [output.png] [--print] [--no-cache]\n"
        "  python wacomm_plot.py dataset      <sample_csv> <matrix_csv> [output_dir] [--max-depth N]\n"
        "\n"
        "  For profile/matrix/matrix-lines/totals:\n"
        "    p / lambda    : latitude and longitude (e.g. 40.85  14.27)\n"
        "    t / t0        : timestamp yyyymmddZhh00  (e.g. 20230523Z0800)\n"
        "    output.png    : optional — if absent, displays on screen\n"
        "    --print       : also print numerical data to the terminal\n"
        "    --max-depth N : maximum Y-axis depth in metres (default from config.json)\n"
        "    --no-cache    : bypass the on-disk cache\n"
        "\n"
        "  For dataset:\n"
        "    dataset_dir   : folder produced by wacomm_dataset.py (processes all samples)\n"
        "    or:\n"
        "    sample_csv    : single sample CSV produced by wacomm_dataset.py\n"
        "    matrix_csv    : single matrix CSV produced by wacomm_dataset.py\n"
        "    output_dir    : output directory (default: same folder as the CSV files)\n"
        "    --max-depth N : maximum depth for the heatmap\n"
    )

    raw_args = sys.argv[1:]

    do_print = "--print" in raw_args
    if do_print:
        raw_args = [a for a in raw_args if a != "--print"]

    use_cache = "--no-cache" not in raw_args
    if not use_cache:
        raw_args = [a for a in raw_args if a != "--no-cache"]

    max_depth = DEFAULT_MAX_DEPTH
    if "--max-depth" in raw_args:
        idx = raw_args.index("--max-depth")
        try:
            max_depth = float(raw_args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --max-depth requires a numeric value.\n")
            print(USAGE)
            sys.exit(1)
        del raw_args[idx:idx + 2]

    args = raw_args

    if len(args) < 1:
        print(USAGE)
        sys.exit(1)

    subcommand = args[0]
    if subcommand not in ("profile", "matrix", "matrix-lines", "totals", "dataset"):
        print(f"Unknown subcommand: '{subcommand}'\n")
        print(USAGE)
        sys.exit(1)

    # ── dataset subcommand ────────────────────────────────────────────────────
    if subcommand == "dataset":
        if len(args) < 2:
            print(
                "Usage:\n"
                "  python wacomm_plot.py dataset <dataset_dir>             [output_dir]\n"
                "  python wacomm_plot.py dataset <sample_csv> <matrix_csv> [output_dir]\n"
            )
            sys.exit(1)

        def _build_sample_from_csvs(s_path: str, m_path: str) -> dict:
            """Reconstructs the sample dict from the two CSVs produced by wacomm_dataset."""
            df_s = pd.read_csv(s_path)
            df_m = pd.read_csv(m_path, index_col=0)
            row   = df_s.iloc[0]
            n_hrs = len([c for c in df_s.columns if c.startswith("h_")])
            sums  = np.array([row.get(f"h_{i-(n_hrs-1):+03d}", np.nan)
                              for i in range(n_hrs)], dtype=float)
            timestamps = [c.split("_", 2)[2] for c in df_m.columns]
            depths_r   = np.array([float(d.replace("m", "")) for d in df_m.index])
            all_depths = np.array(list(COPERNICUS_DEPTHS))
            full_mat   = np.full((len(all_depths), len(timestamps)), np.nan)
            for ki, d in enumerate(depths_r):
                idx2 = np.where(np.abs(all_depths - d) < 0.01)[0]
                if len(idx2):
                    full_mat[idx2[0], :] = df_m.values[ki, :]
            return {
                "scheda"       : str(row.get("scheda", "unknown")),
                "year"         : int(row.get("year", 0)),
                "date_utc"     : str(row.get("date_utc", "")),
                "t0"           : str(row.get("t0", "")),
                "sito"         : str(row.get("sito", "")),
                "lat"          : float(row.get("lat", 0)),
                "lon"          : float(row.get("lon", 0)),
                "outcome"      : int(row.get("outcome", 0)),
                "target"       : int(row.get("target", 0)),
                "_timestamps"  : timestamps,
                "_column_sums" : sums,
                "_matrix"      : full_mat,
                "_depths"      : list(COPERNICUS_DEPTHS),
                "_missing"     : [],
            }

        def _plot_sample_pair(sample: dict, out_dir: str) -> None:
            """Generates the two plots for a single sample."""
            os.makedirs(out_dir, exist_ok=True)
            safe = sample["scheda"].replace("/", "_")
            t0   = sample["t0"]
            p1   = os.path.join(out_dir, f"{safe}_{t0}_plot.png")
            p2   = os.path.join(out_dir, f"{safe}_{t0}_matrix_plot.png")
            plot_sample(sample, save_path=p1)
            plot_sample_matrix(sample, save_path=p2, max_depth=max_depth)
            print(f"  Sample plot → {p1}")
            print(f"  Matrix plot → {p2}")

        # Case A: first argument is a directory → process all samples inside
        if os.path.isdir(args[1]):
            dataset_dir = args[1]
            output_dir  = args[2] if len(args) >= 3 else dataset_dir

            # Find all pairs ({stem}.csv, {stem}_matrix.csv)
            csvs = sorted(f for f in os.listdir(dataset_dir)
                          if f.endswith(".csv") and "_matrix" not in f
                          and not f.endswith("_matrix.csv"))
            if not csvs:
                print(f"No sample CSVs found in: {dataset_dir}")
                sys.exit(1)

            print(f"Found {len(csvs)} samples in: {dataset_dir}")
            n_ok = n_err = 0
            for csv_name in csvs:
                stem       = csv_name[:-4]           # strip .csv
                s_path     = os.path.join(dataset_dir, csv_name)
                m_path     = os.path.join(dataset_dir, f"{stem}_matrix.csv")
                if not os.path.exists(m_path):
                    print(f"  [SKIP] {csv_name}: matrix CSV not found")
                    n_err += 1
                    continue
                print(f"\n{csv_name}")
                try:
                    sample = _build_sample_from_csvs(s_path, m_path)
                    _plot_sample_pair(sample, output_dir)
                    n_ok += 1
                except Exception as e:
                    print(f"  [WARN] {e}", file=sys.stderr)
                    n_err += 1

            print(f"\n{'='*50}")
            print(f"Plots generated: {n_ok}  |  Errors: {n_err}")

        # Case B: first argument is a CSV file → single sample
        else:
            if len(args) < 3:
                print("Usage: python wacomm_plot.py dataset <sample_csv> <matrix_csv> [output_dir]\n")
                sys.exit(1)
            s_path     = args[1]
            m_path     = args[2]
            output_dir = args[3] if len(args) >= 4 else os.path.dirname(
                             os.path.abspath(s_path))
            sample = _build_sample_from_csvs(s_path, m_path)
            _plot_sample_pair(sample, output_dir)

        sys.exit(0)

    # ── profile / matrix / matrix-lines / totals subcommands ─────────────────
    if len(args) not in (4, 5):
        print(USAGE)
        sys.exit(1)

    try:
        p_arg    = float(args[1])
        lam_arg  = float(args[2])
        t_arg    = args[3]
        save_arg = args[4] if len(args) == 5 else None

        if subcommand == "profile":
            result = get_concentration_profile(p_arg, lam_arg, t_arg,
                                               use_cache=use_cache)
            if do_print:
                _print_profile(result, p_arg, lam_arg, t_arg)
            plot_profile(result, p_arg, lam_arg, t_arg,
                        save_path=save_arg, max_depth=max_depth)
        elif subcommand == "matrix":
            result = get_concentration_matrix(p_arg, lam_arg, t_arg,
                                              use_cache=use_cache)
            if do_print:
                _print_matrix_summary(result, p_arg, lam_arg)
            plot_matrix(result, p_arg, lam_arg, t_arg,
                       save_path=save_arg, max_depth=max_depth)
        elif subcommand == "matrix-lines":
            result = get_concentration_matrix(p_arg, lam_arg, t_arg,
                                              use_cache=use_cache)
            if do_print:
                _print_matrix_summary(result, p_arg, lam_arg)
            plot_matrix_lines(result, p_arg, lam_arg, t_arg,
                              save_path=save_arg, max_depth=max_depth)
        else:  # totals
            result = get_concentration_matrix(p_arg, lam_arg, t_arg,
                                              use_cache=use_cache)
            if do_print:
                _print_matrix_summary(result, p_arg, lam_arg)
            plot_column_sums(result, p_arg, lam_arg, t_arg, save_path=save_arg)

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)