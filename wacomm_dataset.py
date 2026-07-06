"""
wacomm_dataset.py
-----------------
Crea campioni per il dataset di addestramento ML integrando:
  - file XLS/CSV dei risultati IZS (E.coli su Mytilus)
  - file GeoJSON dei banchi molluschi
  - file history WaComM++ (tramite wacomm_profile.py)

Per ogni campionamento valido produce due file CSV:
  - {scheda}_{t0}.csv         — 72 feature (column_sums orarie) + label
  - {scheda}_{t0}_matrix.csv  — matrice (d × 72) dove d = livelli Copernicus
                                entro max_depth metri

Per la generazione dei grafici usare wacomm_plot.py.

Utilizzo da riga di comando:
    python wacomm_dataset.py <izs_file> <banchi_geojson>
                             [--output-dir DIR] [--max-depth N] [--no-cache]

    izs_file      : file XLS o CSV dei risultati IZS
    banchi_geojson: file GeoJSON delle zone di allevamento
    --output-dir  : cartella di output (default: ./dataset/)
    --max-depth N : profondità massima per la matrice (default: da config.json)
    --no-cache    : disabilita la cache dei file history

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
from datetime import datetime, time

# Rende importabile wacomm_profile dallo stesso percorso di questo script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wacomm_profile import get_concentration_matrix

from config import (
    N_HOURS,
    DEFAULT_MAX_DEPTH,
    BACTERIA,
    SPECIES,
    TARGET_BINS,
    SAMPLING_HOUR_LOCAL,
    TIMEZONE,
)

# ── Costanti derivate dalla configurazione ───────────────────────────────────

ROME_TZ       = pytz.timezone(TIMEZONE)
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
    (stesso comportamento di uploadMeasurements in routes.py di talco).

    Restituisce un DataFrame con colonne:
        scheda, year, date, sito, lat, lon, outcome, target, t0
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato file non supportato: {filepath}")

    df = df[
        df["PARAMETRO/ANALITA"].isin(BACTERIA) &
        df["MATRICE"].isin(SPECIES)
    ].copy()

    df["outcome"] = df["ESITO"].apply(_extract_outcome)
    df = df[df["outcome"].notna()].copy()

    if df.empty:
        raise ValueError(
            "Nessuna riga valida trovata dopo il filtro E.coli / Mytilus."
        )

    df_agg = df.groupby("NUMERO SCHEDA").agg(
        year    = ("ANNO ACCETTAZIONE", "first"),
        date    = ("DATA PRELIEVO",     "first"),
        sito    = ("SITO",              "first"),
        lat     = ("LATITUDINE",        "first"),
        lon     = ("LONGITUDINE",       "first"),
        outcome = ("outcome",           "max"),
    ).reset_index().rename(columns={"NUMERO SCHEDA": "scheda"})

    df_agg["sito"] = df_agg["sito"].str.strip().str.upper()

    def _to_utc(row) -> datetime:
        d = row["date"]
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime()
        dt_local = datetime.combine(d.date(), time(SAMPLING_HOUR_LOCAL, 0, 0))
        dt_local = ROME_TZ.localize(dt_local)
        return dt_local.astimezone(pytz.utc)

    df_agg["date_utc"] = df_agg.apply(_to_utc, axis=1)
    df_agg["t0"] = df_agg["date_utc"].apply(
        lambda d: d.strftime("%Y%m%dZ%H00")
    )

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
    vs campo SITO del CSV), come fa talco.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        gj = json.load(f)

    return {
        feat["properties"]["DENOMINAZI"].upper().strip(): feat
        for feat in gj["features"]
    }


# ── Creazione campioni ───────────────────────────────────────────────────────

def build_sample(row: pd.Series, use_cache: bool = True) -> dict | None:
    """
    Per un singolo campionamento IZS, calcola la timeseries di N_HOURS ore
    chiamando get_concentration_matrix() e restituisce un dizionario con
    features + label + metadata + dati grezzi per il plotting.

    Restituisce None se il calcolo fallisce del tutto.

    Il dict contiene:
      - metadata: scheda, year, date_utc, t0, sito, lat, lon, outcome, target
      - feature 72×1: h_-71 … h_+00  (column_sums, somma intera colonna)
      - dati grezzi (prefisso '_', usati da wacomm_plot.py):
          _timestamps, _column_sums, _matrix, _depths, _missing
    """
    try:
        result = get_concentration_matrix(
            p         = float(row["lat"]),
            lam       = float(row["lon"]),
            t0        = str(row["t0"]),
            n_hours   = N_HOURS,
            use_cache = use_cache,
        )
    except Exception as e:
        print(f"  [WARN] Impossibile calcolare timeseries per {row['scheda']}: {e}",
              file=sys.stderr)
        return None

    column_sums = result["column_sums"]
    timestamps  = result["timestamps"]

    sample = {
        "scheda"   : row["scheda"],
        "year"     : int(row["year"]),
        "date_utc" : str(row["date_utc"]),
        "t0"       : row["t0"],
        "sito"     : row["sito"],
        "lat"      : float(row["lat"]),
        "lon"      : float(row["lon"]),
        "outcome"  : int(row["outcome"]),
        "target"   : int(row["target"]),
    }

    # Feature 72×1: somma intera colonna per ogni ora
    for i, ts in enumerate(timestamps):
        hrel = i - (N_HOURS - 1)
        val  = column_sums[i]
        sample[f"h_{hrel:+03d}"] = float(val) if not np.isnan(val) else None

    # Dati grezzi per il plotting (non finiscono nei CSV)
    sample["_timestamps"]  = timestamps
    sample["_column_sums"] = column_sums
    sample["_matrix"]      = result["matrix"]
    sample["_depths"]      = result["depths"]
    sample["_missing"]     = result.get("missing_timestamps", [])

    return sample


# ── Salvataggio CSV campione (72×1 → MPN) ────────────────────────────────────

def save_sample_csv(sample: dict, output_dir: str) -> str:
    """
    Salva il campione 72×1 come CSV.

    Colonne: 9 metadata + 72 feature (h_-71 … h_+00) + label target
    Nome file: {scheda}_{t0}.csv

    Le feature rappresentano la concentrazione totale nella colonna d'acqua
    (somma su tutti i livelli Copernicus) per ciascuna delle 72 ore.

    Restituisce il percorso del file salvato.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_scheda = sample["scheda"].replace("/", "_").replace("\\", "_")
    filepath    = os.path.join(output_dir, f"{safe_scheda}_{sample['t0']}.csv")

    row = {k: v for k, v in sample.items() if not k.startswith("_")}
    pd.DataFrame([row]).to_csv(filepath, index=False)
    return filepath


# ── Salvataggio CSV matrice (72×d → MPN) ────────────────────────────────────

def save_matrix_csv(sample: dict, output_dir: str,
                    max_depth: float = DEFAULT_MAX_DEPTH) -> str:
    """
    Salva la matrice delle concentrazioni per livello come CSV.

    Righe   = livelli Copernicus con profondità ≤ max_depth (d righe)
    Colonne = 72 ore (h_-71 … h_+00)

    A differenza del CSV campione (72×1, somma dell'intera colonna),
    qui ogni riga rappresenta un singolo livello di profondità, permettendo
    al modello ML di distinguere il contributo di ciascun livello.

    Nome file: {scheda}_{t0}_matrix.csv

    Restituisce il percorso del file salvato.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_scheda = sample["scheda"].replace("/", "_").replace("\\", "_")
    filepath    = os.path.join(output_dir,
                               f"{safe_scheda}_{sample['t0']}_matrix.csv")

    matrix     = np.array(sample["_matrix"])    # (136, 72)
    depths     = np.array(sample["_depths"])    # (136,) in metri
    timestamps = sample["_timestamps"]          # (72,)

    # Filtra ai soli livelli entro max_depth
    in_range = depths <= max_depth
    matrix_r = matrix[in_range, :]
    depths_r = depths[in_range]

    col_labels = [
        f"h_{i-(N_HOURS-1):+03d}_{ts}"
        for i, ts in enumerate(timestamps)
    ]
    df_mat = pd.DataFrame(
        matrix_r,
        index  = [f"{d:.2f}m" for d in depths_r],
        columns= col_labels,
    )
    df_mat.index.name = "depth_m"
    df_mat.to_csv(filepath)
    return filepath


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea campioni dataset ML (CSV) da risultati IZS + WaComM history"
    )
    parser.add_argument("izs_file",       help="File XLS/CSV dei risultati IZS")
    parser.add_argument("banchi_geojson", help="File GeoJSON dei banchi molluschi")
    parser.add_argument("--output-dir",  default="./dataset",
                        help="Cartella di output (default: ./dataset/)")
    parser.add_argument("--max-depth",   type=float, default=DEFAULT_MAX_DEPTH,
                        help=f"Profondità massima per la matrice in metri "
                             f"(default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Disabilita la cache dei file history")
    args = parser.parse_args()

    use_cache  = not args.no_cache
    output_dir = args.output_dir
    max_depth  = args.max_depth

    # 1. Carica IZS
    print(f"Carico file IZS: {args.izs_file}")
    df_izs = load_izs(args.izs_file)
    print(f"  Campioni validi: {len(df_izs)}")

    # 2. Carica banchi
    print(f"Carico banchi: {args.banchi_geojson}")
    bank_map = load_banchi(args.banchi_geojson)
    print(f"  Banchi caricati: {len(bank_map)}")

    # 3. Filtra per match sito↔banco
    df_izs = df_izs[df_izs["sito"].isin(bank_map)].copy()
    print(f"  Campioni con sito nel GeoJSON: {len(df_izs)}")

    if df_izs.empty:
        print("Nessun campione da elaborare. Uscita.")
        sys.exit(0)

    # 4. Per ogni campione: genera i due CSV
    os.makedirs(output_dir, exist_ok=True)
    n_ok = 0
    n_err = 0

    for idx, row in df_izs.iterrows():
        print(f"\n[{idx+1}/{len(df_izs)}] {row['scheda']}  "
              f"sito={row['sito']}  t0={row['t0']}  "
              f"outcome={int(row['outcome'])}  target={row['target']}")

        sample = build_sample(row, use_cache=use_cache)
        if sample is None:
            n_err += 1
            continue

        csv_72x1 = save_sample_csv(sample, output_dir)
        print(f"  72×1 CSV   → {csv_72x1}")

        csv_72xd = save_matrix_csv(sample, output_dir, max_depth=max_depth)
        d = (np.array(sample["_depths"]) <= max_depth).sum()
        print(f"  72×{d} CSV  → {csv_72xd}")

        n_ok += 1

    # 5. Riepilogo
    print(f"\n{'='*60}")
    print(f"Campioni elaborati con successo : {n_ok}")
    print(f"Campioni con errori             : {n_err}")
    print(f"Output nella cartella           : {os.path.abspath(output_dir)}")
    print(f"Per generare i grafici usare    : wacomm_plot.py dataset ...")


if __name__ == "__main__":
    main()