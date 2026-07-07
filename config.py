"""
config.py
---------
Loads config.json and exposes configuration parameters
as importable Python constants for use by other scripts.

Usage:
    from config import CFG, HISTORY_ROOT, CACHE_DIR, ...
"""

import json
import os

# Look for config.json in the same directory as this file
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load and return the contents of config.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Load on import ───────────────────────────────────────────────────────────
CFG = load_config()

# Paths
_base = os.path.dirname(os.path.abspath(__file__))

def _resolve(p: str) -> str:
    """Resolve a relative path relative to the directory of config.json."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_base, p))

HISTORY_ROOT    = _resolve(CFG["paths"]["history_root"])
CACHE_DIR       = _resolve(CFG["paths"]["cache_dir"])
METACHARTS_PATH = _resolve(CFG["paths"]["metacharts"])

# Model parameters
N_HOURS    = int(CFG["model"]["n_hours"])
FILL_VALUE = float(CFG["model"]["fill_value"])

# Plot parameters
DEFAULT_MAX_DEPTH = float(CFG["plot"]["default_max_depth"])
# PLOT_Y_MAX can be set to null in config.json to disable the Y-axis upper limit
_y_max = CFG["plot"]["y_max_concentration"]
PLOT_Y_MAX = float(_y_max) if _y_max is not None else None

# Dataset parameters
BACTERIA              = CFG["dataset"]["bacteria"]
SPECIES               = CFG["dataset"]["species"]
TARGET_BINS           = CFG["dataset"]["target_bins"]
SAMPLING_HOUR_LOCAL   = int(CFG["dataset"]["sampling_hour_local"])
TIMEZONE              = CFG["dataset"]["timezone"]