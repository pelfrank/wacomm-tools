# wacomm-tools

A Python toolset for extracting, visualising, and preparing bacterial concentration data from WaComM++ (Water quality Community Model) simulations, aimed at building Machine Learning training datasets for coastal water quality assessment.

The project integrates with analytical results on *Mytilus galloprovincialis* (mussels), and with NetCDF files produced by the WaComM++ ocean model managed by CMMMA (Center for Monitoring and Modeling Marine and Atmospheric research and applications) — University of Naples Parthenope.

---

## Repository structure

```
wacomm-tools/
├── config.json            # Central configuration (edit parameters here)
├── config.py              # Reads config.json, imported by all scripts
├── wacomm_profile.py      # Extract concentration profiles and matrices
├── wacomm_plot.py         # Visualisation: profiles, matrices, time series, dataset plots
├── wacomm_dataset.py      # Build ML dataset CSV files from IZS results + WaComM
├── metacharts.json        # Colour scale and concentration levels (app)
├── util/                  # Interpolation package (from ccmmma-postpro)
│   ├── Distributor.py
│   ├── Interpolator.py
│   ├── Wacomm.py
│   └── ROMS.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Prerequisites

- Python **3.11** or higher
- WaComM++ history files (NetCDF) already available on the local machine at the path set in `config.json` (`paths.history_root`). The expected naming convention is:

```
{history_root}/yyyy/mm/dd/wcm3_d03_yyyymmddZhh00.nc
```
---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/pelfrank/wacomm-tools.git
cd wacomm-tools
```

**2. Create and activate a virtual environment (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install --upgrade pip       # Recommended
pip install -r requirements.txt
```

**4. Configure paths in `config.json`**

Open `config.json` and update at least `paths.history_root` with the local path to your WaComM NetCDF files:

```json
{
    "paths": {
        "history_root": "/path/to/your/history/files",
        "cache_dir": "./cache",
        "metacharts": "./metacharts.json"
    }
}
```

---

## Configuration (`config.json`)

All tuneable parameters are centralised in `config.json`. No Python code needs to be modified to adapt the toolset to a new environment.

| Section | Key | Description | Default |
|---|---|---|---|
| `paths` | `history_root` | Root directory of the NetCDF history files | `/storage/ccmmma/...` |
| `paths` | `cache_dir` | Directory for the `.npz` cache | `./cache` |
| `paths` | `metacharts` | Path to the colour-scale file | `./metacharts.json` |
| `model` | `n_hours` | Number of hours in the timeseries before sampling | `72` |
| `model` | `fill_value` | Fill value for land/missing cells | `1e37` |
| `plot` | `default_max_depth` | Maximum depth (m) shown in plots | `50.0` |
| `plot` | `y_max_concentration` | Y-axis maximum for concentration plots. Set to `null` to use automatic scaling | `46000` |
| `dataset` | `bacteria` | Filter IZS analytes by name | `["Escherichia coli"]` |
| `dataset` | `species` | Filter IZS matrices by species | `["MYTILUS..."]` |
| `dataset` | `target_bins` | E. coli classification thresholds (CFU/100g) | `[78, 230, 4600]` |
| `dataset` | `sampling_hour_local` | Local sampling hour used for UTC conversion | `10` |
| `dataset` | `timezone` | Local timezone for UTC conversion | `"Europe/Rome"` |

---

## Usage

### `wacomm_profile.py` — Extract profiles and matrices

Extracts the vertical concentration profile (136 Copernicus depth levels) or the temporal matrix (136 levels × 72 hours) for a given geographic point and timestamp.

```bash
# Vertical profile at a single timestamp
python wacomm_profile.py profile <latitude> <longitude> <timestamp>

# Matrix for the preceding 72 hours
python wacomm_profile.py matrix <latitude> <longitude> <timestamp>

# Disable cache
python wacomm_profile.py matrix 40.85 14.27 20230523Z0800 --no-cache
```

Timestamps use the format `yyyymmddZhh00` (e.g. `20230523Z0800`).

---

### `wacomm_plot.py` — Visualisation

Generates plots from data extracted by `wacomm_profile.py`, or from CSV files produced by `wacomm_dataset.py`.

```bash
# Vertical concentration profile
python wacomm_plot.py profile <lat> <lon> <t>  [output.png] [--print] [--max-depth N] [--no-cache]

# Concentration matrix heatmap (depth × time)
python wacomm_plot.py matrix <lat> <lon> <t0>  [output.png] [--print] [--max-depth N] [--no-cache]

# One line per depth level
python wacomm_plot.py matrix-lines <lat> <lon> <t0>  [output.png] [--print] [--max-depth N] [--no-cache]

# Total water-column concentration time series
python wacomm_plot.py totals <lat> <lon> <t0>  [output.png] [--print] [--no-cache]

# Dataset plots — entire folder (all samples at once)
python wacomm_plot.py dataset <dataset_dir>  [output_dir]  [--max-depth N]

# Dataset plots — single sample
python wacomm_plot.py dataset <sample_csv> <matrix_csv>  [output_dir]  [--max-depth N]
```

**Options:**

| Flag | Description |
|---|---|
| `output.png` | Save the plot to file instead of displaying it on screen |
| `--print` | Also print numerical data to the terminal |
| `--max-depth N` | Limit the depth axis to N metres (default: from `config.json`) |
| `--no-cache` | Do not read from or write to the on-disk cache |

**Examples:**
```bash
# Plot all samples produced by wacomm_dataset.py in one command
python wacomm_plot.py dataset ./dataset_2023/

# Plot a single sample
python wacomm_plot.py dataset ./dataset_2023/1043A-10060-B_20230125Z0900.csv \
                               ./dataset_2023/1043A-10060-B_20230125Z0900_matrix.csv

# Plot with automatic Y-axis (no fixed maximum)
# → set "y_max_concentration": null in config.json, then:
python wacomm_plot.py totals 40.76558 14.37735 20230523Z0800 output.png
```

---

### `wacomm_dataset.py` — Build the ML dataset

Creates training CSV files for an ML dataset by combining analytical results with WaComM simulations.
**This script produces only CSV files; use `wacomm_plot.py dataset` to generate the plots.**

```bash
python wacomm_dataset.py <analytical_results_file> <farms_geojson_file> [--output-dir DIR] [--max-depth N] [--no-cache]
```

**Arguments:**

| Argument | Description |
|---|---|
| `izs_file` | XLS or CSV file with IZS analytical results |
| `banchi_geojson` | GeoJSON file of the mussel farming zones |
| `--output-dir` | Output directory (default: `./dataset/`) |
| `--max-depth N` | Maximum depth (m) for the matrix CSV (default: from `config.json`) |
| `--no-cache` | Do not use the history-file cache |

**Example:**
```bash
# Step 1 — generate CSV files
python wacomm_dataset.py esiti_2023.xls banchi.geojson --output-dir ./dataset_2023/

# Step 2 — generate plots from CSV files
python wacomm_plot.py dataset ./dataset_2023/
```

**Output per sample** (e.g. `scheda=1043A-10060-B`, `t0=20230125Z0900`):

| File | Contents |
|---|---|
| `1043A-10060-B_20230125Z0900.csv` | 9 metadata columns + 72 features (hourly column sums, 72×1) + `target` label |
| `1043A-10060-B_20230125Z0900_matrix.csv` | d × 72 concentration matrix, where d = Copernicus levels within `max_depth` metres |
| `1043A-10060-B_20230125Z0900_plot.png` | 72-hour WaComM time series + IZS value at t₀ *(generated by wacomm_plot.py)* |
| `1043A-10060-B_20230125Z0900_matrix_plot.png` | Matrix heatmap *(generated by wacomm_plot.py)* |

**Sample CSV structure (72×1):**

The 72 features are named `h_-71`, `h_-70`, ..., `h_+00`, where `h_+00` is the sampling hour (t₀) and `h_-71` is 71 hours before. Each feature is the sum of concentrations across the entire water column at that point and hour.

**Matrix CSV structure (72×d):**

Rows = Copernicus depth levels with depth ≤ `max_depth` (e.g. 14 rows for `max_depth=50`). Columns = 72 hours. Unlike the 72×1 CSV, each row represents a single depth level, allowing the model to distinguish the contribution of each depth.

The `target` label is the bacterial contamination class:

| Class | E. coli range (CFU/100g) | Meaning |
|---|---|---|
| 0 | ≤ 78 | Zone A — excellent quality |
| 1 | 79 – 230 | Zone B — good quality |
| 2 | 231 – 4600 | Zone C — sufficient quality |
| 3 | > 4600 | Harvesting prohibited |

---

## Cache

Scripts use a `.npz` (NumPy compressed) cache to avoid re-reading and re-computing history files on every run. The cache is stored in `cache/` (configurable). Files follow the naming convention:

```
{timestamp}_{p|m}_{n_hours}h.npz
```

where `p` = profile, `m` = matrix. Use `--no-cache` to force recomputation.

---

## Full pipeline

```
WaComM history files (.nc)
        │
        ▼
wacomm_profile.py  ──────────►  wacomm_plot.py  (profile / matrix / totals)
(data extraction)
        │
        ▼
wacomm_dataset.py              wacomm_plot.py dataset
(CSV only)          ─────────► (plots from CSV)
        │                              │
        ├──► {scheda}_{t0}.csv         ├──► {scheda}_{t0}_plot.png
        └──► {scheda}_{t0}_matrix.csv  └──► {scheda}_{t0}_matrix_plot.png
```

---

## Dependencies and credits

- **WaComM++** — Lagrangian particulate transport model for coastal waters, developed by CMMMA, University of Naples Parthenope
- **ccmmma-postpro** — interpolation package (`util/`) provided by CMMMA; `Distributor.py` and `Interpolator.py` implement the remapping from sigma coordinates to Copernicus depth levels
- **talco** — CMMMA web application for ML training data cleaning; `wacomm_dataset.py` replicates the filtering and classification logic from `talco/routes.py`