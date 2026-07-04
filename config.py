"""
config.py
---------
Carica config.json e rende disponibili i parametri di configurazione
come costanti Python importabili dagli altri script.

Utilizzo:
    from config import CFG, HISTORY_ROOT, CACHE_DIR, ...
"""

import json
import os

# Cerca config.json nella stessa directory di questo file
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Carica e restituisce il contenuto di config.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Caricamento all'import ───────────────────────────────────────────────────
CFG = load_config()

# Percorsi
_base = os.path.dirname(os.path.abspath(__file__))

def _resolve(p: str) -> str:
    """Risolve un percorso relativo rispetto alla directory di config.json."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_base, p))

HISTORY_ROOT    = _resolve(CFG["paths"]["history_root"])
CACHE_DIR       = _resolve(CFG["paths"]["cache_dir"])
METACHARTS_PATH = _resolve(CFG["paths"]["metacharts"])

# Parametri modello
N_HOURS    = int(CFG["model"]["n_hours"])
FILL_VALUE = float(CFG["model"]["fill_value"])

# Parametri plot
DEFAULT_MAX_DEPTH = float(CFG["plot"]["default_max_depth"])
PLOT_Y_MAX        = float(CFG["plot"]["y_max_concentration"])

# Parametri dataset
BACTERIA              = CFG["dataset"]["bacteria"]
SPECIES               = CFG["dataset"]["species"]
TARGET_BINS           = CFG["dataset"]["target_bins"]
SAMPLING_HOUR_LOCAL   = int(CFG["dataset"]["sampling_hour_local"])
TIMEZONE              = CFG["dataset"]["timezone"]