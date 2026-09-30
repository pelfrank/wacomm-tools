"""
wacomm_qgis_batch_loader.py
----------------------------
PyQGIS batch script to be run from the QGIS Python Console.

For each discarded sample CSV found in the input directory, creates a
dedicated QGIS project (.qgz) that contains:
  - The 72 hourly GeoTIFF rasters (Temporal Controller ready)
  - The sampling point GeoJSON
  - The concentration colour scale from metacharts.json

Projects are named {scheda}_{t0}.qgz and saved to the output directory.

Prerequisites:
    Run wacomm_batch_geotiff.py first to generate the GeoTIFFs and GeoJSONs.

Usage:
    1. Open QGIS.
    2. Open the Python Console: Plugins → Python Console
    3. Click "Show Editor" and open this file.
    4. Edit the CONFIGURATION section below.
    5. Click "Run Script".

Tested on QGIS 4.2.0-Belém do Pará (PyQt6 / Qt6).
"""

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths before running
# ═══════════════════════════════════════════════════════════════════════════

# Directory containing the discarded sample CSV files
SAMPLES_DIR = "/path/to/dataset/2023/scartati/"

# Root directory containing the shared GeoTIFF folders (one per t0)
# produced by wacomm_batch_geotiff.py
GEOTIFF_ROOT = "/path/to/geotiff/"

# Directory containing the GeoJSON files produced by wacomm_batch_geotiff.py
GEOJSON_DIR = "/path/to/geojson/2023/scartati/"

# Directory where QGIS project files (.qgz) will be saved
PROJECTS_DIR = "/path/to/qgis_projects/2023/scartati/"

# Path to metacharts.json for the concentration colour scale
METACHARTS_PATH = "/path/to/wacomm-tools/metacharts.json"

# Temporal step: duration of each frame in hours (keep at 1)
STEP_HOURS = 1

# ═══════════════════════════════════════════════════════════════════════════
# Script — do not edit below this line
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import json

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


# ── Helpers (identical to wacomm_qgis_loader.py) ─────────────────────────────

def timestamp_to_qdatetime(ts: str) -> QDateTime:
    """Converts yyyymmddZhh00 → QDateTime in UTC. Compatible with QGIS 4.x."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})Z(\d{2})00$", ts)
    if not m:
        raise ValueError(f"Invalid WaComM timestamp: {ts!r}")
    yyyy, mm, dd, hh = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    try:
        utc_spec = Qt.TimeSpec.UTC      # PyQt6 / QGIS 4.x
    except AttributeError:
        utc_spec = Qt.UTC               # PyQt5 / QGIS 3.x
    return QDateTime(QDate(yyyy, mm, dd), QTime(hh, 0, 0), utc_spec)


def load_metacharts_colormap(metacharts_path: str):
    """Reads clevels/ccolors from metacharts.json → QgsColorRampShader items."""
    with open(metacharts_path, "r") as f:
        meta = json.load(f)["meta-chart"]
    clevels = meta["clevels"]
    ccolors = meta["ccolors"]
    items = []
    for level, rgba in zip(clevels, ccolors):
        r, g, b, a = rgba
        color = QColor(r, g, b, a)
        items.append(QgsColorRampShader.ColorRampItem(float(level), color,
                                                       str(level)))
    return items, float(clevels[0]), float(clevels[-1])


def make_pseudocolor_renderer(layer, metacharts_path: str):
    """Discrete pseudocolor renderer from metacharts.json. QGIS 4.x compatible."""
    color_items, vmin, vmax = load_metacharts_colormap(metacharts_path)
    ramp_shader = QgsColorRampShader(vmin, vmax)
    ramp_shader.setColorRampType(Qgis.ShaderInterpolationMethod.Discrete)
    ramp_shader.setColorRampItemList(color_items)
    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(ramp_shader)
    return QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, raster_shader
    )


def find_sample_csvs(directory: str) -> list:
    """Returns sorted list of sample CSVs (excluding *_matrix.csv)."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".csv") and not f.endswith("_matrix.csv")
    )


def t0_from_csv(csv_path: str) -> str:
    """Extracts t0 from filename: {scheda}_{t0}.csv → t0."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return stem.rsplit("_", 1)[-1]


def stem_from_csv(csv_path: str) -> str:
    """Returns filename stem without extension: e.g. '1043A-50590-B_20230523Z0800'."""
    return os.path.splitext(os.path.basename(csv_path))[0]


# ── Per-sample project creation ───────────────────────────────────────────────

def create_project_for_sample(csv_path: str,
                               geotiff_root: str,
                               geojson_dir: str,
                               projects_dir: str,
                               metacharts_path: str,
                               step_hours: int) -> bool:
    """
    Creates a QGIS project for a single discarded sample:
      - Loads the 72 GeoTIFFs from the shared t0 folder
      - Loads the sampling point GeoJSON
      - Applies the metacharts colour scale and temporal properties
      - Saves the project as {stem}.qgz

    Returns True on success, False if GeoTIFF folder or GeoJSON is missing.
    """
    stem       = stem_from_csv(csv_path)         # e.g. 1043A-50590-B_20230523Z0800
    t0         = t0_from_csv(csv_path)           # e.g. 20230523Z0800
    tif_dir    = os.path.join(geotiff_root, t0)  # shared GeoTIFF folder
    geojson    = os.path.join(geojson_dir, f"{stem}.geojson")
    project_path = os.path.join(projects_dir, f"{stem}.qgz")

    # Skip if project already exists
    if os.path.exists(project_path):
        print(f"  [SKIP] {stem}.qgz already exists")
        return True

    # Check prerequisites
    if not os.path.isdir(tif_dir):
        print(f"  [WARN] GeoTIFF folder not found: {tif_dir}")
        return False
    if not os.path.exists(geojson):
        print(f"  [WARN] GeoJSON not found: {geojson}")
        return False

    # Create a new blank QGIS project
    project = QgsProject.instance()
    project.clear()
    project.setTitle(stem)

    # ── Load GeoTIFFs ─────────────────────────────────────────────────────
    tif_files = sorted(
        f for f in os.listdir(tif_dir)
        if f.startswith("wcm3_") and f.endswith(".tif")
    )
    if not tif_files:
        print(f"  [WARN] No GeoTIFFs found in: {tif_dir}")
        return False

    root  = project.layerTreeRoot()
    group = root.insertGroup(0, f"WaComM {t0}")

    for tif_name in tif_files:
        ts       = tif_name[len("wcm3_"):-len(".tif")]
        tif_path = os.path.join(tif_dir, tif_name)

        layer = QgsRasterLayer(tif_path, f"wcm3 {ts}")
        if not layer.isValid():
            continue

        # Colour renderer from metacharts.json
        renderer = make_pseudocolor_renderer(layer, metacharts_path)
        layer.setRenderer(renderer)

        # Temporal properties
        t_start = timestamp_to_qdatetime(ts)
        t_end   = t_start.addSecs(step_hours * 3600)
        tp = layer.temporalProperties()
        tp.setMode(Qgis.RasterTemporalMode.FixedTemporalRange)
        tp.setFixedTemporalRange(QgsDateTimeRange(t_start, t_end))
        tp.setIsActive(True)

        project.addMapLayer(layer, addToLegend=False)
        group.addLayer(layer)

    # ── Load GeoJSON ──────────────────────────────────────────────────────
    vec_layer = QgsVectorLayer(geojson, stem, "ogr")
    if vec_layer.isValid():
        project.addMapLayer(vec_layer)
    else:
        print(f"  [WARN] Could not load GeoJSON: {geojson}")

    # ── Save project ──────────────────────────────────────────────────────
    os.makedirs(projects_dir, exist_ok=True)
    ok = project.write(project_path)
    if ok:
        print(f"  ✓  {stem}.qgz  ({len(tif_files)} layers)")
    else:
        print(f"  ✗  Failed to save: {project_path}")

    return ok


# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("WaComM QGIS Batch Loader")
print("=" * 60)
print(f"Samples dir   : {SAMPLES_DIR}")
print(f"GeoTIFF root  : {GEOTIFF_ROOT}")
print(f"GeoJSON dir   : {GEOJSON_DIR}")
print(f"Projects dir  : {PROJECTS_DIR}")
print()

csvs = find_sample_csvs(SAMPLES_DIR)
if not csvs:
    print(f"No sample CSVs found in: {SAMPLES_DIR}")
else:
    print(f"Found {len(csvs)} samples\n")
    n_ok = n_err = 0
    for i, csv_path in enumerate(csvs, start=1):
        stem = stem_from_csv(csv_path)
        print(f"[{i}/{len(csvs)}] {stem}")
        ok = create_project_for_sample(
            csv_path      = csv_path,
            geotiff_root  = GEOTIFF_ROOT,
            geojson_dir   = GEOJSON_DIR,
            projects_dir  = PROJECTS_DIR,
            metacharts_path = METACHARTS_PATH,
            step_hours    = STEP_HOURS,
        )
        if ok:
            n_ok += 1
        else:
            n_err += 1

    print(f"\n{'='*60}")
    print(f"Projects created successfully : {n_ok}")
    print(f"Errors                        : {n_err}")
    print(f"Output directory              : {os.path.abspath(PROJECTS_DIR)}")
