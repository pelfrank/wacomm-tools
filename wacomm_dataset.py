"""
wacomm_dataset.py
-----------------
Crea campioni per il dataset di addestramento ML integrando:
  - file XLS/CSV dei risultati IZS (E.coli su Mytilus)
  - file GeoJSON dei banchi molluschi
  - file history WaComM++ (tramite wacomm_profile.py)

Per ogni campionamento valido produce:
  - un file CSV  con 72 feature (column_sums orarie) + label + metadata
                 ({scheda}_{t0}.csv)
  - un file PNG  con la serie temporale + valore IZS come ultimo punto
                 ({scheda}_{t0}_plot.png)
  - un file CSV  con la matrice 136×72 delle concentrazioni per livello
                 ({scheda}_{t0}_matrix.csv)
  - un file PNG  con la heatmap della matrice (plot_matrix di wacomm_plot)
                 ({scheda}_{t0}_matrix_plot.png)

Utilizzo da riga di comando:
    python wacomm_dataset.py <izs_file> <banchi_geojson> [--output-dir DIR]
                             [--no-cache] [--no-plot]

    izs_file      : file XLS o CSV dei risultati IZS
    banchi_geojson: file GeoJSON delle zone di allevamento
    --output-dir  : cartella di output (default: ./dataset/)
    --no-cache    : disabilita la cache dei file history
    --no-plot     : salta la generazione dei grafici

Esempio:
    python wacomm_dataset.py esiti_2023.xls banchi.geojson --output-dir ./out/
"""

import sys
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, time, timedelta

matplotlib.use("Agg")

# Rende importabile wacomm_profile dallo stesso percorso di questo script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wacomm_profile import (
    get_concentration_matrix,
    CACHE_DIR,
)

# Importa plot_matrix e colormap da wacomm_plot se disponibili
try:
    from wacomm_plot import load_concentration_colormap, plot_matrix as _wplot_matrix
    _HAS_WACOMM_PLOT = True
except ImportError:
    _HAS_WACOMM_PLOT = False

from config import (
    N_HOURS,
    PLOT_Y_MAX,
    BACTERIA,
    SPECIES,
    TARGET_BINS,
    SAMPLING_HOUR_LOCAL,
    TIMEZONE,
)

# ── Costanti derivate dalla configurazione ───────────────────────────────────

ROME_TZ      = pytz.timezone(TIMEZONE)
TARGET_LABELS = [0, 1, 2, 3]


# ── Parsing del file XLS/CSV dei risultati IZS ──────────────────────────────

def _extract_outcome(value) -> int | None:
    """
    Estrae il valore numerico dall'esito IZS.
    Es.: '290' -> 290, '830 >' -> 830, 'NEGATIVO' -> None
    Replica la logica di extract_outcome() in routes.py di talco.
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value and value[0].isdigit():
            return int(re.sub(r"\D", "", value))
    return None


def load_izs(filepath: str) -> pd.DataFrame:
    """
    Carica e filtra il file XLS/CSV dei risultati IZS.

    Applica i filtri di talco:
      - solo righe con PARAMETRO/ANALITA in BACTERIA
      - solo righe con MATRICE in SPECIES
      - solo esiti numerici

    Aggrega per NUMERO SCHEDA prendendo il max dell'esito
    (stesso comportamento di uploadMeasurements in routes.py).

    Restituisce un DataFrame con colonne:
        scheda, year, date, sito, lat, lon, outcome, target, t0
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato file non supportato: {filepath}")

    # Filtro batterio + specie
    df = df[
        df["PARAMETRO/ANALITA"].isin(BACTERIA) &
        df["MATRICE"].isin(SPECIES)
    ].copy()

    # Estrazione esito numerico
    df["outcome"] = df["ESITO"].apply(_extract_outcome)
    df = df[df["outcome"].notna()].copy()

    if df.empty:
        raise ValueError(
            "Nessuna riga valida trovata dopo il filtro E.coli / Mytilus."
        )

    # Aggregazione per campionamento (max outcome su duplicati)
    df_agg = df.groupby("NUMERO SCHEDA").agg(
        year    = ("ANNO ACCETTAZIONE", "first"),
        date    = ("DATA PRELIEVO",     "first"),
        sito    = ("SITO",              "first"),
        lat     = ("LATITUDINE",        "first"),
        lon     = ("LONGITUDINE",       "first"),
        outcome = ("outcome",           "max"),
    ).reset_index().rename(columns={"NUMERO SCHEDA": "scheda"})

    # Normalizza nome sito
    df_agg["sito"] = df_agg["sito"].str.strip().str.upper()

    # Data UTC con offset +10h (ora 10:00 locale = ora solare) come in talco
    def _to_utc(row) -> datetime:
        d = row["date"]
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime()
        dt_local = datetime.combine(d.date(), time(SAMPLING_HOUR_LOCAL, 0, 0))
        dt_local = ROME_TZ.localize(dt_local)
        return dt_local.astimezone(pytz.utc)

    df_agg["date_utc"] = df_agg.apply(_to_utc, axis=1)

    # Timestamp t0 nel formato atteso da wacomm_profile
    df_agg["t0"] = df_agg["date_utc"].apply(
        lambda d: d.strftime("%Y%m%dZ%H00")
    )

    # Classe target: 5 bordi → 4 intervalli → 4 label
    df_agg["target"] = pd.cut(
        df_agg["outcome"],
        bins  = [-1] + TARGET_BINS + [float("inf")],
        labels= TARGET_LABELS,
        right = True,
    ).astype(int)

    return df_agg


# ── Parsing del file GeoJSON dei banchi ─────────────────────────────────────

def load_banchi(filepath: str) -> dict:
    """
    Carica il file GeoJSON dei banchi e restituisce un dizionario
        { nome_banco_upper: feature }

    Il match con il CSV avviene per nome (campo DENOMINAZI del GeoJSON
    vs campo SITO del CSV), come fa talco in uploadFarms / uploadMeasurements.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        gj = json.load(f)

    bank_map = {}
    for feat in gj["features"]:
        name = feat["properties"]["DENOMINAZI"].upper().strip()
        bank_map[name] = feat

    return bank_map


# ── Creazione campioni ───────────────────────────────────────────────────────

def build_sample(row: pd.Series, use_cache: bool = True) -> dict | None:
    """
    Per un singolo campionamento IZS, calcola la timeseries di 72 ore
    chiamando get_concentration_matrix() e restituisce un dizionario
    con features + label + metadata.

    La feature i-esima (i=0 = ora più vecchia, i=71 = t0) è il valore
    di column_sums[i], cioè la somma delle concentrazioni su tutta la
    colonna d'acqua al punto (lat, lon) all'ora corrispondente.

    Restituisce None se il calcolo della timeseries fallisce del tutto.
    """
    try:
        result = get_concentration_matrix(
            p       = float(row["lat"]),
            lam     = float(row["lon"]),
            t0      = str(row["t0"]),
            n_hours = N_HOURS,
            use_cache = use_cache,
        )
    except Exception as e:
        print(f"  [WARN] Impossibile calcolare timeseries per {row['scheda']}: {e}",
              file=sys.stderr)
        return None

    column_sums = result["column_sums"]   # array (72,)
    timestamps  = result["timestamps"]

    # Costruisce il campione: feature da h-71 a h0 + label
    sample = {
        "scheda"    : row["scheda"],
        "year"      : int(row["year"]),
        "date_utc"  : str(row["date_utc"]),
        "t0"        : row["t0"],
        "sito"      : row["sito"],
        "lat"       : float(row["lat"]),
        "lon"       : float(row["lon"]),
        "outcome"   : int(row["outcome"]),
        "target"    : int(row["target"]),
    }
    # Feature: colonne h_-71 … h_0
    for i, ts in enumerate(timestamps):
        hrel = i - (N_HOURS - 1)          # da -(N_HOURS-1) a 0
        val  = column_sums[i]
        sample[f"h_{hrel:+03d}"] = float(val) if not np.isnan(val) else None

    sample["_timestamps"] = timestamps
    sample["_column_sums"] = column_sums
    sample["_matrix"]     = result["matrix"]       # (136, 72)
    sample["_depths"]     = result["depths"]       # lista 136 profondità in metri
    sample["_missing"]    = result.get("missing_timestamps", [])

    return sample


# ── Salvataggio CSV del campione ─────────────────────────────────────────────

def save_sample_csv(sample: dict, output_dir: str) -> str:
    """
    Salva il campione come CSV nella cartella output_dir.
    Nome file: {scheda}_{t0}.csv  (con '/' nel numero scheda sostituito da '_')

    Restituisce il percorso del file salvato.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_scheda = sample["scheda"].replace("/", "_").replace("\\", "_")
    filename = f"{safe_scheda}_{sample['t0']}.csv"
    filepath = os.path.join(output_dir, filename)

    # Riga del CSV: metadata + 72 feature + label
    row = {
        "scheda"  : sample["scheda"],
        "year"    : sample["year"],
        "date_utc": sample["date_utc"],
        "t0"      : sample["t0"],
        "sito"    : sample["sito"],
        "lat"     : sample["lat"],
        "lon"     : sample["lon"],
        "outcome" : sample["outcome"],
        "target"  : sample["target"],
    }
    for key, val in sample.items():
        if key.startswith("h_"):
            row[key] = val

    pd.DataFrame([row]).to_csv(filepath, index=False)
    return filepath


# ── Plot della serie temporale + valore IZS ─────────────────────────────────

def save_sample_plot(sample: dict, csv_path: str) -> str:
    """
    Genera il grafico della serie temporale (column_sums 72h) con il valore
    IZS aggiunto come punto finale colorato diversamente (arancione/rosso),
    analogo al grafico 'totals' ma con la label al posto dell'ultima feature.

    Il file viene salvato nella stessa cartella del CSV con suffisso '_plot.png'.

    Restituisce il percorso del file salvato.
    """
    plot_path = csv_path.replace(".csv", "_plot.png")

    timestamps  = sample["_timestamps"]     # (72,)
    column_sums = np.array(sample["_column_sums"], dtype=float)
    outcome     = sample["outcome"]
    n_hours     = len(timestamps)

    x = np.arange(-(n_hours - 1), 1)        # [-71 … 0]

    # Carica colormap dell'app se disponibile, altrimenti verde semplice
    if _HAS_WACOMM_PLOT:
        try:
            cmap_app, norm_app, unit_app, label_app = load_concentration_colormap()
        except Exception:
            cmap_app = norm_app = None
    else:
        cmap_app = norm_app = None

    fig, ax = plt.subplots(figsize=(15, 5))

    # ── Parte verde: ore da -71 a -1 (features WaComM) ─────────────────────
    x_feat = x[:-1]                          # [-71 … -1]
    y_feat = column_sums[:-1]

    ax.fill_between(x_feat, 0, y_feat,
                    where=~np.isnan(y_feat),
                    color="#3a9b3a", alpha=0.85, linewidth=0, zorder=1)
    ax.plot(x_feat, y_feat,
            color="#2e7d32", linewidth=1.3, zorder=2)

    # ── Punto finale: valore IZS (outcome) come label ───────────────────────
    # Lo plottiamo nella posizione x=0 (t0), usando il valore reale IZS
    # invece della concentrazione WaComM, così il campione è visivo
    last_y = column_sums[-1]   # ultimo valore WaComM (bridge visivo)
    if np.isnan(last_y):
        last_y = 0.0

    ax.fill_between([x_feat[-1], 0], 0,
                    [y_feat[-1] if not np.isnan(y_feat[-1]) else 0, outcome],
                    color="#e53935", alpha=0.95, linewidth=0, zorder=3)
    ax.plot([x_feat[-1], 0],
            [y_feat[-1] if not np.isnan(y_feat[-1]) else 0, outcome],
            color="#c62828", linewidth=1.5, zorder=4)

    # Marker del valore IZS
    target = sample["target"]
    target_colors = {0: "#4caf50", 1: "#ff9800", 2: "#f44336", 3: "#7b1fa2"}
    ax.scatter([0], [outcome],
               color=target_colors.get(target, "#c62828"),
               s=80, zorder=6, edgecolors="black", linewidths=0.7,
               label=f"IZS outcome={outcome} UFC/100g (classe {target})")

    # ── Asse X: ore relative a t0 ───────────────────────────────────────────
    step = 6
    tick_positions = list(x[::step])
    if 0 not in tick_positions:
        tick_positions.append(0)

    tick_labels = []
    for rel_h in tick_positions:
        col = int(rel_h + (n_hours - 1))
        stamp = timestamps[col]
        hh = stamp[9:11]
        prev_col = col - step
        if col == 0 or (prev_col >= 0 and timestamps[col][:8] != timestamps[prev_col][:8]):
            tick_labels.append(f"{stamp[6:8]}/{stamp[4:6]}\n{hh}:00 ({int(rel_h):+d}h)")
        else:
            tick_labels.append(f"{hh}:00\n({int(rel_h):+d}h)")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(x[0], 0)
    ax.set_xlabel("Ore rispetto al campionamento t₀ (UTC)", fontsize=11)

    ax.set_ylim(bottom=0, top=PLOT_Y_MAX)
    ax.set_ylabel("Concentrazione totale [#]", fontsize=10)

    # ── Asse Y destro: stessa scala, etichetta MPN/100g (E.coli) ────────────
    # MPN/100g e UFC/100g sono numericamente equivalenti per E.coli su
    # molluschi, quindi l'asse destro condivide gli stessi limiti/tick
    # dell'asse sinistro, cambia solo l'etichetta.
    ax_right = ax.twinx()
    ax_right.set_ylim(0, PLOT_Y_MAX)
    ax_right.set_ylabel("E. coli [MPN/100g]", fontsize=10)

    ax.set_title(
        f"Campione dataset ML  —  {sample['sito']}\n"
        f"lat={sample['lat']:.4f}°N   lon={sample['lon']:.4f}°E   "
        f"t₀={sample['t0']}   scheda={sample['scheda']}",
        fontsize=11
    )

    ax.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)

    # Annotazione file mancanti
    if sample["_missing"]:
        ax.text(0.01, 0.95,
                f"Ore mancanti: {len(sample['_missing'])} (buchi nella serie)",
                transform=ax.transAxes, fontsize=8, color="red", va="top")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ── Salvataggio CSV e plot della matrice ────────────────────────────────────

def save_matrix_csv(sample: dict, output_dir: str) -> str:
    """
    Salva la matrice delle concentrazioni (136 livelli × 72 ore) come CSV.

    Righe   = livelli Copernicus (dalla superficie al fondo)
    Colonne = timestamps delle 72 ore (h_-71 … h_+00)
    Header  = timestamp di ogni ora; prima colonna = profondità in metri

    Nome file: {scheda}_{t0}_matrix.csv

    Restituisce il percorso del file salvato.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_scheda = sample["scheda"].replace("/", "_").replace("\\", "_")
    filename    = f"{safe_scheda}_{sample['t0']}_matrix.csv"
    filepath    = os.path.join(output_dir, filename)

    matrix     = sample["_matrix"]          # (136, 72) — salvato da build_sample
    depths     = sample["_depths"]          # (136,) profondità Copernicus in metri
    timestamps = sample["_timestamps"]      # (72,) stringhe yyyymmddZhh00

    # Costruisce il DataFrame: indice = profondità, colonne = timestamp
    col_labels = [
        f"h_{i-(N_HOURS-1):+03d}_{ts}"
        for i, ts in enumerate(timestamps)
    ]
    df_mat = pd.DataFrame(
        matrix,
        index  = [f"{d:.2f}m" for d in depths],
        columns= col_labels,
    )
    df_mat.index.name = "depth_m"
    df_mat.to_csv(filepath)
    return filepath


def save_matrix_plot(sample: dict, matrix_csv_path: str,
                     max_depth: float = 50.0) -> str:
    """
    Genera la heatmap della matrice delle concentrazioni riusando
    plot_matrix() di wacomm_plot.py, se disponibile.

    Nome file: {scheda}_{t0}_matrix_plot.png
               (stesso prefisso del CSV matrice, con '_plot.png' finale)

    Restituisce il percorso del file salvato.
    """
    plot_path = matrix_csv_path.replace("_matrix.csv", "_matrix_plot.png")

    # Prepara il dict nel formato atteso da plot_matrix di wacomm_plot
    result = {
        "matrix"             : sample["_matrix"],
        "depths"             : sample["_depths"],
        "timestamps"         : sample["_timestamps"],
        "lat_found"          : sample["lat"],
        "lon_found"          : sample["lon"],
        "lat_idx"            : 0,   # non usato dal plot, solo per metadati
        "lon_idx"            : 0,
        "missing_timestamps" : sample["_missing"],
    }

    if _HAS_WACOMM_PLOT:
        _wplot_matrix(
            result, sample["lat"], sample["lon"], sample["t0"],
            save_path=plot_path, max_depth=max_depth,
        )
    else:
        # Fallback minimale se wacomm_plot non è disponibile
        import matplotlib.pyplot as plt
        mat  = np.array(sample["_matrix"])
        deps = np.array(sample["_depths"])
        in_r = deps <= max_depth
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.pcolormesh(mat[in_r, :], cmap="YlOrRd", shading="auto")
        ax.set_title(f"Matrice concentrazione  —  {sample['sito']}  t₀={sample['t0']}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return plot_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea campioni dataset ML da risultati IZS + WaComM history"
    )
    parser.add_argument("izs_file",
                        help="File XLS/CSV dei risultati IZS")
    parser.add_argument("banchi_geojson",
                        help="File GeoJSON dei banchi molluschi")
    parser.add_argument("--output-dir", default="./dataset",
                        help="Cartella di output (default: ./dataset/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disabilita la cache dei file history")
    parser.add_argument("--no-plot", action="store_true",
                        help="Salta la generazione dei grafici")
    args = parser.parse_args()

    use_cache  = not args.no_cache
    output_dir = args.output_dir

    # ── 1. Carica e filtra dati IZS ─────────────────────────────────────────
    print(f"Carico file IZS: {args.izs_file}")
    df_izs = load_izs(args.izs_file)
    print(f"  Campioni validi dopo filtro: {len(df_izs)}")

    # ── 2. Carica banchi GeoJSON ─────────────────────────────────────────────
    print(f"Carico banchi: {args.banchi_geojson}")
    bank_map = load_banchi(args.banchi_geojson)
    print(f"  Banchi caricati: {len(bank_map)}")

    # ── 3. Filtra campioni per match sito↔banco (per nome) ──────────────────
    df_izs = df_izs[df_izs["sito"].isin(bank_map)].copy()
    print(f"  Campioni con sito nel GeoJSON: {len(df_izs)}")

    if df_izs.empty:
        print("Nessun campione da elaborare. Uscita.")
        sys.exit(0)

    # ── 4. Per ogni campione: timeseries → CSV + plot + matrice ─────────────
    os.makedirs(output_dir, exist_ok=True)
    n_ok = 0
    n_err = 0

    for idx, row in df_izs.iterrows():
        scheda = row["scheda"]
        print(f"\n[{idx+1}/{len(df_izs)}] {scheda}  sito={row['sito']}  "
              f"t0={row['t0']}  outcome={int(row['outcome'])}  "
              f"target={row['target']}")

        # Calcola timeseries WaComM
        sample = build_sample(row, use_cache=use_cache)
        if sample is None:
            n_err += 1
            continue

        # Salva CSV campione (feature + label)
        csv_path = save_sample_csv(sample, output_dir)
        print(f"  CSV campione  → {csv_path}")

        # Salva CSV matrice (136 livelli × 72 ore)
        matrix_csv_path = save_matrix_csv(sample, output_dir)
        print(f"  CSV matrice   → {matrix_csv_path}")

        # Salva plot e plot matrice
        if not args.no_plot:
            try:
                plot_path = save_sample_plot(sample, csv_path)
                print(f"  Plot campione → {plot_path}")
            except Exception as e:
                print(f"  [WARN] Errore nel plot campione: {e}", file=sys.stderr)
            try:
                matrix_plot_path = save_matrix_plot(sample, matrix_csv_path)
                print(f"  Plot matrice  → {matrix_plot_path}")
            except Exception as e:
                print(f"  [WARN] Errore nel plot matrice: {e}", file=sys.stderr)

        n_ok += 1

    # ── 5. Riepilogo ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Campioni elaborati con successo : {n_ok}")
    print(f"Campioni con errori             : {n_err}")
    print(f"Output nella cartella           : {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()