"""
wacomm_qgis_loader.py
----------------------
PyQGIS script to be run from the QGIS Python Console.

Loads all WaComM++ GeoTIFF files produced by wacomm_to_geotiff.py into
the current QGIS project, configures their temporal properties for use
with the Temporal Controller, applies a consistent colour ramp, and
optionally loads a sampling point GeoJSON on top.

Usage:
    1. Open QGIS and create a new or existing project.
    2. Open the Python Console: Plugins → Python Console
    3. Click the "Show Editor" button (the script icon in the console toolbar)
    4. Open this file in the editor, or paste its contents.
    5. Edit the CONFIGURATION section below to match your paths.
    6. Click "Run Script" (the green play button).

After running:
    - All 72 GeoTIFF layers will appear in the layer panel.
    - Open View → Panels → Temporal Controller.
    - Press Play to animate the 72-hour concentration sequence.
    - The sampling point GeoJSON (if provided) stays always visible on top.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths before running
# ═══════════════════════════════════════════════════════════════════════════

# Directory containing the GeoTIFF files produced by wacomm_to_geotiff.py
GEOTIFF_DIR = "/home/francesco/Scrivania/wacomm-tools/data/history_geotiff/2023_single/"

# (Optional) Path to the sampling point GeoJSON produced by wacomm_to_geojson.py.
# Set to None or "" to skip loading the GeoJSON.
GEOJSON_PATH = "/home/francesco/Scrivania/wacomm-tools/dataset/2023_single/1043A-50590-B_20230523Z0800.geojson"

# Name of the QGIS group that will contain all 72 raster layers
GROUP_NAME = "1043A-50590-B_20230523Z0800"

# Path to metacharts.json — same file used by wacomm_plot.py.
# The colour scale (clevels + ccolors) is read from this file so that
# QGIS uses the exact same colours as the Python concentration plots.
METACHARTS_PATH = "/home/francesco/Scrivania/wacomm-tools/metacharts.json"

# Temporal step: duration of each frame in the animation (hours).
# Keep at 1 to match the hourly resolution of WaComM history files.
STEP_HOURS = 1

# ═══════════════════════════════════════════════════════════════════════════
# Script — do not edit below this line
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import json
from datetime import datetime, timezone, timedelta

from qgis.core import (
    Qgis,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsRasterLayerTemporalProperties,
    QgsDateTimeRange,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)
from qgis.PyQt.QtCore import QDateTime, QDate, QTime, Qt
from qgis.PyQt.QtGui import QColor


def timestamp_to_qdatetime(ts: str) -> QDateTime:
    """
    Converts a WaComM timestamp string (yyyymmddZhh00) to a QDateTime in UTC.
    E.g. '20230523Z0800' → QDateTime(2023, 5, 23, 8, 0, 0, UTC)
    Compatible with QGIS 4.x (PyQt6): uses Qt.TimeSpec.UTC instead of Qt.UTC.
    """
    m = re.match(r"^(\d{4})(\d{2})(\d{2})Z(\d{2})00$", ts)
    if not m:
        raise ValueError(f"Invalid WaComM timestamp: {ts!r}")
    yyyy, mm, dd, hh = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    # PyQt6 (QGIS 4.x) uses Qt.TimeSpec.UTC; PyQt5 (QGIS 3.x) uses Qt.UTC.
    # Try both for forward/backward compatibility.
    try:
        utc_spec = Qt.TimeSpec.UTC      # PyQt6 / QGIS 4.x
    except AttributeError:
        utc_spec = Qt.UTC               # PyQt5 / QGIS 3.x
    return QDateTime(QDate(yyyy, mm, dd), QTime(hh, 0, 0), utc_spec)


def load_metacharts_colormap(metacharts_path: str):
    """
    Reads clevels and ccolors from metacharts.json and returns a list of
    QgsColorRampShader.ColorRampItem objects — one per level — that reproduce
    the exact same discrete colour scale used by wacomm_plot.py.

    clevels: list of 36 threshold values (upper boundary of each colour band)
    ccolors: list of 36 RGBA colours [R, G, B, A] in 0-255 range
    """
    with open(metacharts_path, "r") as f:
        meta = json.load(f)["meta-chart"]

    clevels = meta["clevels"]   # [1, 3, 6, 8, ..., 46000]
    ccolors = meta["ccolors"]   # [[R,G,B,A], ...]

    items = []
    for level, rgba in zip(clevels, ccolors):
        r, g, b, a = rgba
        color = QColor(r, g, b, a)
        items.append(QgsColorRampShader.ColorRampItem(float(level), color,
                                                       str(level)))
    return items, float(clevels[0]), float(clevels[-1])


def make_pseudocolor_renderer(layer, metacharts_path):
    """
    Creates a single-band pseudocolor renderer using the discrete colour scale
    from metacharts.json — identical to the one used in the Python plots.

    The colour scale is discrete (Exact mode): each clevels value maps to
    its exact colour, matching the ListedColormap used by wacomm_plot.py.
    Values below the first clevel are transparent; values above the last
    clevel use the last colour.

    Compatible with QGIS 4.x (PyQt6, Qt6).
    """
    color_items, vmin, vmax = load_metacharts_colormap(metacharts_path)

    ramp_shader = QgsColorRampShader(vmin, vmax)
    # Discrete mode: each interval has a fixed colour, no interpolation.
    # This mirrors the ListedColormap / BoundaryNorm used in wacomm_plot.py.
    ramp_shader.setColorRampType(Qgis.ShaderInterpolationMethod.Discrete)
    ramp_shader.setColorRampItemList(color_items)

    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(ramp_shader)

    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, raster_shader
    )
    return renderer


def load_geotiffs(geotiff_dir, group_name, metacharts_path, step_hours):
    """
    Loads all wcm3_*.tif files from geotiff_dir into a QGIS layer group,
    sets temporal properties and colour renderer on each layer.
    Returns the number of layers loaded.
    """
    # Find and sort all WaComM GeoTIFF files
    tif_files = sorted(
        f for f in os.listdir(geotiff_dir)
        if f.startswith("wcm3_") and f.endswith(".tif")
    )
    if not tif_files:
        print(f"No wcm3_*.tif files found in: {geotiff_dir}")
        return 0

    print(f"Found {len(tif_files)} GeoTIFF files in: {geotiff_dir}")

    # Create or get the layer group in the layer panel
    root  = QgsProject.instance().layerTreeRoot()
    group = root.findGroup(group_name)
    if group is None:
        group = root.insertGroup(0, group_name)

    n_ok = 0
    for tif_name in tif_files:
        # Extract timestamp from filename: wcm3_20230523Z0800.tif
        ts = tif_name[len("wcm3_"):-len(".tif")]
        tif_path = os.path.join(geotiff_dir, tif_name)

        # Load the raster layer
        layer = QgsRasterLayer(tif_path, f"wcm3 {ts}")
        if not layer.isValid():
            print(f"  [WARN] Could not load: {tif_path}")
            continue

        # ── Colour renderer (metacharts.json discrete scale) ──────────────
        renderer = make_pseudocolor_renderer(layer, metacharts_path)
        layer.setRenderer(renderer)

        # ── Temporal properties ───────────────────────────────────────────
        t_start = timestamp_to_qdatetime(ts)
        t_end   = t_start.addSecs(step_hours * 3600)

        tp = layer.temporalProperties()
        # QGIS 4.x: mode enum moved to Qgis.RasterTemporalMode
        tp.setMode(Qgis.RasterTemporalMode.FixedTemporalRange)
        tp.setFixedTemporalRange(QgsDateTimeRange(t_start, t_end))
        tp.setIsActive(True)

        # ── Add to project and group ──────────────────────────────────────
        QgsProject.instance().addMapLayer(layer, addToLegend=False)
        group.addLayer(layer)

        print(f"  ✓  {ts}")
        n_ok += 1

    return n_ok


def load_geojson(geojson_path):
    """Loads the sampling point GeoJSON as a vector layer on top of all rasters."""
    if not geojson_path:
        return
    layer = QgsVectorLayer(geojson_path,
                           os.path.splitext(os.path.basename(geojson_path))[0],
                           "ogr")
    if not layer.isValid():
        print(f"[WARN] Could not load GeoJSON: {geojson_path}")
        return
    QgsProject.instance().addMapLayer(layer)
    print(f"GeoJSON loaded: {geojson_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("WaComM QGIS Loader")
print("=" * 60)

n = load_geotiffs(
    geotiff_dir    = GEOTIFF_DIR,
    group_name     = GROUP_NAME,
    metacharts_path= METACHARTS_PATH,
    step_hours     = STEP_HOURS,
)

if GEOJSON_PATH:
    load_geojson(GEOJSON_PATH)

print()
print(f"Layers loaded       : {n}")
print()
print("Next steps:")
print("  1. View → Panels → Temporal Controller")
print("  2. Set the time step to 1 hour")
print("  3. Press Play to animate the concentration sequence")
