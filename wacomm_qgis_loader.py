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
GEOTIFF_DIR = "/path/to/geotiff/20230523Z0800"

# (Optional) Path to the sampling point GeoJSON produced by wacomm_to_geojson.py.
# Set to None or "" to skip loading the GeoJSON.
GEOJSON_PATH = "/path/to/dataset/1043A-50590-B_20230523Z0800.geojson"

# Name of the QGIS group that will contain all 72 raster layers
GROUP_NAME = "WaComM concentration"

# Colour ramp for the concentration values.
# Options: 'YlOrRd', 'Blues', 'Greens', 'RdYlBu', 'viridis', 'plasma', etc.
# Any name valid in QgsColorRampShader works.
COLOR_RAMP = "YlOrRd"

# Minimum and maximum values for the colour scale.
# Set both to None to use automatic scaling per layer (less consistent).
CONC_MIN = 0.0
CONC_MAX = 3000.0

# Temporal step: duration of each frame in the animation (hours).
# Keep at 1 to match the hourly resolution of WaComM history files.
STEP_HOURS = 1

# ═══════════════════════════════════════════════════════════════════════════
# Script — do not edit below this line
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
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
    QgsStyle,
)
from qgis.PyQt.QtCore import QDateTime, QDate, QTime, Qt


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


def make_pseudocolor_renderer(layer, vmin, vmax, ramp_name):
    """
    Creates a single-band pseudocolor renderer with the requested colour ramp,
    applied to the given raster layer between vmin and vmax.
    Compatible with QGIS 4.x (PyQt6, Qt6).
    """
    # Build the colour ramp from QGIS built-in styles
    style      = QgsStyle.defaultStyle()
    color_ramp = style.colorRamp(ramp_name)
    if color_ramp is None:
        print(f"  [WARN] Colour ramp '{ramp_name}' not found; using Spectral.")
        color_ramp = style.colorRamp("Spectral")

    # QGIS 4.x: color ramp is passed to the constructor via setSourceColorRamp,
    # interpolation type is now Qgis.ShaderInterpolationMethod
    ramp_shader = QgsColorRampShader(vmin, vmax)
    ramp_shader.setColorRampType(Qgis.ShaderInterpolationMethod.Linear)
    ramp_shader.setSourceColorRamp(color_ramp)
    ramp_shader.classifyColorRamp()

    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(ramp_shader)

    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, raster_shader
    )
    return renderer


def load_geotiffs(geotiff_dir, group_name, color_ramp, vmin, vmax, step_hours):
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

    # Determine colour scale (auto if vmin/vmax not set)
    use_auto_scale = (vmin is None or vmax is None)

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

        # ── Colour renderer ───────────────────────────────────────────────
        if use_auto_scale:
            stats = layer.dataProvider().bandStatistics(1)
            cur_min = stats.minimumValue
            cur_max = stats.maximumValue
        else:
            cur_min, cur_max = vmin, vmax

        renderer = make_pseudocolor_renderer(layer, cur_min, cur_max, color_ramp)
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

        print(f"  ✓  {ts}  [{cur_min:.0f} – {cur_max:.0f}]")
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
    geotiff_dir = GEOTIFF_DIR,
    group_name  = GROUP_NAME,
    color_ramp  = COLOR_RAMP,
    vmin        = CONC_MIN,
    vmax        = CONC_MAX,
    step_hours  = STEP_HOURS,
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