"""
wacomm_profile.py
-----------------
Funzioni per estrarre concentrazioni di particelle dai file history WaComM++.

Punto 1 — profilo verticale in un singolo istante:
    python wacomm_profile.py profile <p> <lambda> <t>

Punto 2 — matrice (136 livelli Copernicus × 72 ore) centrata su t_0:
    python wacomm_profile.py matrix <p> <lambda> <t0>

Argomenti comuni:
    p       : latitudine  (gradi nord, float, es. 40.85)
    lambda  : longitudine (gradi est,  float, es. 14.27)
    t / t0  : timestamp   nel formato yyyymmddZhh00 (es. 20260601Z0600)

I file history vengono cercati in:
    /storage/ccmmma/prometeo/data/opendap/wcm3/d03/history/yyyy/mm/dd/
    wcm3_d03_yyyymmddZhh00.nc

Cache su disco
--------------
get_concentration_profile() e get_concentration_matrix() salvano/leggono
automaticamente i risultati in ./cache/ (file .npz), per evitare di
rileggere e ricalcolare tutto a ogni esecuzione. Il file 'matrix' contiene
già 'column_sums' (il vettore usato da 'totals'), quindi non serve una
cache separata per i totali.

Nome file: {timestamp}_{p|m}_{n_hours}h.npz
    p = profile, m = matrix   (n_hours è sempre 1 per 'profile')

Per disabilitare la cache: passare use_cache=False alle funzioni, oppure
--no-cache da riga di comando.
"""

import sys
import os
import re
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset

# Rende importabile il package util/ di ccmmma-postpro.
# Modifica POSTPRO_UTIL_DIR se il percorso è diverso nel tuo ambiente.
POSTPRO_UTIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "util")
if os.path.dirname(POSTPRO_UTIL_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(POSTPRO_UTIL_DIR))

from util.Distributor import Distrib3D
from util.Interpolator import depths as COPERNICUS_DEPTHS   # 136 livelli in metri
from config import HISTORY_ROOT, CACHE_DIR, FILL_VALUE


# ── Funzioni di supporto ─────────────────────────────────────────────────────

def build_history_path(t: str) -> str:
    """
    Costruisce il percorso completo del file history dato il timestamp t.

    Parametri
    ----------
    t : str
        Timestamp nel formato yyyymmddZhh00 (es. '20260601Z0600').

    Restituisce
    -----------
    str
        Percorso assoluto del file NetCDF history.

    Eccezioni
    ---------
    ValueError
        Se il formato del timestamp non è valido.
    """
    pattern = r"^(\d{4})(\d{2})(\d{2})Z(\d{2})00$"
    m = re.match(pattern, t)
    if not m:
        raise ValueError(
            f"Timestamp non valido: '{t}'. "
            "Formato atteso: yyyymmddZhh00 (es. 20260601Z0600)"
        )
    yyyy, mm, dd, hh = m.group(1), m.group(2), m.group(3), m.group(4)
    filename = f"wcm3_d03_{t}.nc"
    return f"{HISTORY_ROOT}/{yyyy}/{mm}/{dd}/{filename}"


def find_nearest_rho_point(lat_rho: np.ndarray, lon_rho: np.ndarray,
                            p: float, lam: float) -> tuple[int, int]:
    """
    Trova gli indici (j, i) del punto RHO della griglia curvilinea
    più vicino al punto geografico (p, lam), usando la distanza euclidea
    su lat/lon (approssimazione valida per domini piccoli/medi).

    Parametri
    ----------
    lat_rho : np.ndarray, shape (eta_rho, eta_xi)
        Latitudini dei punti RHO.
    lon_rho : np.ndarray, shape (eta_rho, eta_xi)
        Longitudini dei punti RHO.
    p   : float
        Latitudine del punto target (gradi nord).
    lam : float
        Longitudine del punto target (gradi est).

    Restituisce
    -----------
    (j, i) : tuple[int, int]
        Indici del punto RHO più vicino (j = asse eta_rho, i = asse eta_xi).
    """
    dist2 = (lat_rho - p) ** 2 + (lon_rho - lam) ** 2
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(j), int(i)


# ── Cache su disco ────────────────────────────────────────────────────────────

def _cache_path(t: str, kind: str, n_hours: int = 1,
                cache_dir: str = CACHE_DIR) -> str:
    """
    Costruisce il percorso del file di cache.

    Convenzione nome file: {timestamp}_{kind}_{n_hours}h.npz
        kind = 'p' per profile, 'm' per matrix

    Parametri
    ----------
    t         : str — timestamp (t per profile, t0 per matrix)
    kind      : str — 'p' oppure 'm'
    n_hours   : int — numero di ore (1 per profile, 72 di default per matrix)
    cache_dir : str — cartella di cache

    Restituisce
    -----------
    str — percorso completo del file .npz
    """
    filename = f"{t}_{kind}_{n_hours}h.npz"
    return os.path.join(cache_dir, filename)


def _save_cache(path: str, result: dict) -> None:
    """
    Salva un dict di risultati in un file .npz, convertendo le liste
    (es. 'depths', 'timestamps') in array numpy compatibili.
    Le chiavi con valori scalari (float/int) vengono salvate come array 0-d.

    Nota: la chiave 'file' (presente nel risultato di get_concentration_profile)
    viene rinominata in '_file' solo all'interno del file .npz, perché
    collide con il parametro posizionale 'file' di np.savez_compressed.
    _load_cache() la ripristina correttamente.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {}
    for key, value in result.items():
        npz_key = "_file" if key == "file" else key
        arrays[npz_key] = np.asarray(value)
    try:
        np.savez_compressed(path, **arrays)
    except OSError as e:
        # La cache è solo un'ottimizzazione: se il salvataggio fallisce
        # (es. disco pieno, permessi) non deve bloccare l'esecuzione.
        print(f"Attenzione: impossibile salvare la cache in {path}: {e}",
              file=sys.stderr)


def _load_cache(path: str) -> dict:
    """
    Carica un dict di risultati da un file .npz precedentemente salvato
    con _save_cache(). Riconverte i campi noti al loro tipo originale
    (liste di stringhe, scalari float/int).

    Eccezioni
    ---------
    FileNotFoundError — se il file di cache non esiste
    """
    with np.load(path, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}

    # Ripristina la chiave 'file' rinominata in '_file' durante il salvataggio
    if "_file" in result:
        result["file"] = result.pop("_file")

    # Ripristina i tipi scalari/lista dai contenitori 0-d numpy
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
    Estrae il profilo verticale di concentrazione sui 136 livelli di profondità
    fisica Copernicus (in metri) per il punto geografico (p, lam) all'istante t.

    La pipeline è identica a quella di postpro-wcm3.py:
      1. Legge dal file history: lat_rho, lon_rho, s_rho, mask_rho, h, conc
      2. Costruisce la griglia regolare di destinazione (linspace su lat/lon)
      3. Applica Distrib3D.distrib() che:
           a) rimappa orizzontalmente livello per livello (nearest-neighbor)
              dalla griglia curvilinea sorgente alla griglia regolare
           b) redistribuisce verticalmente in modo conservativo
              dai livelli sigma s_rho ai 136 livelli fisici Copernicus (metri)
      4. Trova il punto più vicino a (p, lam) sulla griglia regolare risultante
      5. Restituisce il profilo verticale (136,) in quel punto

    Parametri
    ----------
    p         : float — latitudine del punto di interesse (gradi nord)
    lam       : float — longitudine del punto di interesse (gradi est)
    t         : str   — timestamp nel formato yyyymmddZhh00 (es. '20260601Z0600')
    use_cache : bool  — se True (default), legge/scrive il risultato in cache
    cache_dir : str   — cartella di cache (default ./cache/)

    Restituisce
    -----------
    dict con le chiavi:
        - 'conc'      : np.ndarray (136,) — concentrazione per livello Copernicus
                        NaN sui livelli oltre la batimetria locale o in terra
        - 'depths'    : list[float] (136,) — profondità Copernicus in metri
        - 'lat_found' : float — latitudine del punto sulla griglia regolare
        - 'lon_found' : float — longitudine del punto sulla griglia regolare
        - 'lat_idx'   : int   — indice latitudine sulla griglia regolare
        - 'lon_idx'   : int   — indice longitudine sulla griglia regolare
        - 'file'      : str   — percorso del file history letto

    Eccezioni
    ---------
    FileNotFoundError  — file history non trovato
    ValueError         — timestamp non valido
    """
    cache_file = _cache_path(t, "p", n_hours=1, cache_dir=cache_dir)
    if use_cache and os.path.exists(cache_file):
        return _load_cache(cache_file)

    filepath = build_history_path(t)

    try:
        nc = Dataset(filepath, "r")
    except FileNotFoundError:
        raise FileNotFoundError(f"File history non trovato: {filepath}")

    try:
        # 1. Legge le variabili dal file history
        lat_rho  = nc.variables["lat_rho"][:]   # (eta_rho, eta_xi)
        lon_rho  = nc.variables["lon_rho"][:]   # (eta_rho, eta_xi)
        s_rho    = nc.variables["s_rho"][:]     # (30,)
        mask_rho = nc.variables["mask_rho"][:]  # (eta_rho, eta_xi)
        h        = nc.variables["h"][:]         # (eta_rho, eta_xi)
        conc_4d  = nc.variables["conc"][:]      # (1, 30, eta_rho, eta_xi)
    finally:
        nc.close()

    # 2. Costruisce la griglia regolare di destinazione
    #    (esattamente come postpro-wcm3.py)
    dst_lon = np.linspace(lon_rho.min(), lon_rho.max(), lon_rho.shape[1])
    dst_lat = np.linspace(lat_rho.min(), lat_rho.max(), lat_rho.shape[0])

    # 3. Applica Distrib3D: rimappatura orizzontale + redistribuzione verticale
    #    conservativa sigma → livelli Copernicus (136 profondità in metri)
    distributor = Distrib3D(lon_rho, lat_rho, dst_lon, dst_lat,
                            s_rho, mask_rho, h)
    conc_dist = distributor.distrib(conc_4d)
    # conc_dist: (1, 136, len(dst_lat), len(dst_lon))

    # 4. Trova il punto più vicino a (p, lam) sulla griglia regolare 1D
    lat_idx = int(np.argmin(np.abs(dst_lat - p)))
    lon_idx = int(np.argmin(np.abs(dst_lon - lam)))

    # 5. Estrae il profilo verticale (136,) in quel punto
    profile = np.array(conc_dist[0, :, lat_idx, lon_idx], dtype=np.float64)
    # Converte fill value in NaN
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

    if use_cache:
        _save_cache(cache_file, result)

    return result


def shift_timestamp(t: str, hours: int) -> str:
    """
    Sposta un timestamp di `hours` ore (può essere negativo).

    Parametri
    ----------
    t     : str  — timestamp nel formato yyyymmddZhh00
    hours : int  — ore da aggiungere (negativo = indietro nel tempo)

    Restituisce
    -----------
    str — nuovo timestamp nel formato yyyymmddZhh00
    """
    pattern = r"^(\d{4})(\d{2})(\d{2})Z(\d{2})00$"
    m = re.match(pattern, t)
    if not m:
        raise ValueError(
            f"Timestamp non valido: '{t}'. "
            "Formato atteso: yyyymmddZhh00 (es. 20260601Z0600)"
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
    Costruisce la matrice delle concentrazioni (n_livelli × n_hours) per le
    `n_hours` ore precedenti (incluso) l'istante t0.

    L'elemento [k, col] della matrice è la concentrazione al livello Copernicus k
    nell'ora t0 - (n_hours - 1 - col), cioè:
        colonna 0         → t0 - (n_hours-1) h  (ora più vecchia)
        colonna n_hours-1 → t0                  (ora più recente)

    Per ogni ora applica la stessa pipeline di postpro-wcm3.py:
      1. Legge il file history (lat_rho, lon_rho, s_rho, mask_rho, h, conc)
      2. Costruisce la griglia regolare di destinazione
      3. Applica Distrib3D (rimappatura orizzontale + redistribuzione verticale
         conservativa sigma → 136 livelli Copernicus in metri)
      4. Estrae il profilo nel punto più vicino a (p, lam) sulla griglia regolare

    I file mancanti su disco producono una colonna di NaN senza bloccare
    l'esecuzione; vengono tracciati in 'missing_timestamps'.

    Parametri
    ----------
    p         : float — latitudine del punto di interesse (gradi nord)
    lam       : float — longitudine del punto di interesse (gradi est)
    t0        : str   — timestamp finale (formato yyyymmddZhh00)
    n_hours   : int   — numero di ore da coprire, default 72
    use_cache : bool  — se True (default), legge/scrive il risultato in cache
                        (include anche 'column_sums', quindi 'totals' riusa
                        la stessa cache senza ricalcolare nulla)
    cache_dir : str   — cartella di cache (default ./cache/)

    Restituisce
    -----------
    dict con le chiavi:
        - 'matrix'             : np.ndarray (136, n_hours), float64
                                 NaN dove il file non era disponibile o sotto
                                 la batimetria locale
        - 'column_sums'        : np.ndarray (n_hours,), float64 — per ogni ora,
                                 somma di matrix[:, col] su tutti i 136 livelli
                                 (concentrazione totale nella colonna d'acqua).
                                 NaN solo se l'intera colonna era NaN (file mancante).
        - 'depths'             : list[float] (136,) — profondità Copernicus in metri
        - 'timestamps'         : list[str] (n_hours,) — lista timestamp,
                                 colonna 0 = più vecchio, -1 = t0
        - 'lat_found'          : float — latitudine punto sulla griglia regolare
        - 'lon_found'          : float — longitudine punto sulla griglia regolare
        - 'lat_idx'            : int
        - 'lon_idx'            : int
        - 'missing_timestamps' : list[str] — ore per cui il file mancava

    Eccezioni
    ---------
    RuntimeError
        Se nessun file è disponibile nel range temporale richiesto.
    """
    cache_file = _cache_path(t0, "m", n_hours=n_hours, cache_dir=cache_dir)
    if use_cache and os.path.exists(cache_file):
        return _load_cache(cache_file)

    # Costruisce la lista ordinata dei timestamp (dal più vecchio al più recente)
    timestamps = [
        shift_timestamp(t0, -(n_hours - 1 - col))
        for col in range(n_hours)
    ]

    n_levels = len(COPERNICUS_DEPTHS)
    matrix   = np.full((n_levels, n_hours), np.nan, dtype=np.float64)
    missing  = []

    # Variabili per il punto trovato (valorizzate al primo file disponibile)
    lat_found, lon_found, lat_idx, lon_idx = None, None, None, None

    # Per ogni ora: applica la pipeline completa Interpolator + Distributor
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

            # Griglia regolare di destinazione (identica a postpro-wcm3.py)
            dst_lon = np.linspace(lon_rho.min(), lon_rho.max(), lon_rho.shape[1])
            dst_lat = np.linspace(lat_rho.min(), lat_rho.max(), lat_rho.shape[0])

            # Pipeline: rimappatura orizzontale + redistribuzione verticale
            # sigma → 136 livelli Copernicus in metri
            distributor = Distrib3D(lon_rho, lat_rho, dst_lon, dst_lat,
                                    s_rho, mask_rho, h)
            conc_dist = distributor.distrib(conc_4d)
            # conc_dist: (1, 136, len(dst_lat), len(dst_lon))

            # Trova il punto sulla griglia regolare 1D (solo al primo file)
            if lat_idx is None:
                lat_idx   = int(np.argmin(np.abs(dst_lat - p)))
                lon_idx   = int(np.argmin(np.abs(dst_lon - lam)))
                lat_found = float(dst_lat[lat_idx])
                lon_found = float(dst_lon[lon_idx])

            # Estrae il profilo verticale (136,) e converte fill value in NaN
            profile = np.array(conc_dist[0, :, lat_idx, lon_idx], dtype=np.float64)
            profile[profile >= FILL_VALUE * 0.9] = np.nan
            matrix[:, col] = profile

        except (FileNotFoundError, OSError):
            missing.append(ts)

    if lat_idx is None:
        raise RuntimeError(
            f"Nessun file history disponibile nel range "
            f"{timestamps[0]} — {timestamps[-1]}"
        )

    # Vettore (n_hours,): somma delle concentrazioni su tutti i livelli,
    # per ciascuna ora. column_sums[col] = sum(matrix[:, col]), ignorando i NaN.
    # Se per un'ora TUTTI i livelli sono NaN (es. file mancante), la somma
    # risultante è NaN (non 0), per distinguere "nessun dato" da "zero particelle".
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


# ── Main (interfaccia da riga di comando) ────────────────────────────────────

def _print_profile(result: dict, p: float, lam: float, t: str) -> None:
    """Stampa il profilo a schermo in modo leggibile."""
    print(f"\nFile letto       : {result['file']}")
    print(f"Punto richiesto  : lat={p:.4f}°N  lon={lam:.4f}°E")
    print(f"Punto trovato    : lat={result['lat_found']:.4f}°N  "
          f"lon={result['lon_found']:.4f}°E  "
          f"(lat_idx={result['lat_idx']}, lon_idx={result['lon_idx']})")
    print(f"\n{'Livello':>8}  {'Profondità (m)':>16}  {'conc':>12}")
    print("-" * 44)
    for k, (depth_m, c) in enumerate(zip(result["depths"], result["conc"])):
        c_str = f"{c:12.4f}" if not np.isnan(c) else "      (terra)"
        print(f"{k:>8d}  {depth_m:>16.4f}  {c_str}")
    print()


def _print_matrix_summary(result: dict, p: float, lam: float) -> None:
    """Stampa il riepilogo e la matrice completa (136 righe × 72 colonne)."""
    mat    = result["matrix"]
    ts     = result["timestamps"]
    depths = result["depths"]

    # ── Intestazione ────────────────────────────────────────────────────────
    print(f"\nPunto richiesto  : lat={p:.4f}°N  lon={lam:.4f}°E")
    print(f"Punto trovato    : lat={result['lat_found']:.4f}°N  "
          f"lon={result['lon_found']:.4f}°E  "
          f"(lat_idx={result['lat_idx']}, lon_idx={result['lon_idx']})")
    print(f"Range temporale  : {ts[0]}  →  {ts[-1]}")
    print(f"Shape matrice    : {mat.shape[0]} livelli × {mat.shape[1]} ore")
    print(f"min={np.nanmin(mat):.2f}  max={np.nanmax(mat):.2f}  "
          f"NaN={int(np.isnan(mat).sum())}")

    if result["missing_timestamps"]:
        print(f"File mancanti ({len(result['missing_timestamps'])}):",
              ", ".join(result["missing_timestamps"]))

    # ── Intestazione colonne (timestamp) ────────────────────────────────────
    print()
    col_w = 5   # larghezza colonna valori
    lbl_w = 12  # larghezza colonna etichetta livello (es. "136.4m")

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

    # ── Righe della matrice (un livello Copernicus per riga) ────────────────
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

    # ── Vettore delle somme per colonna (concentrazione totale per ora) ─────
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
        "Utilizzo:\n"
        "  Profilo singolo : python wacomm_profile.py profile <p> <lambda> <t>  [--no-cache]\n"
        "  Matrice 72h     : python wacomm_profile.py matrix  <p> <lambda> <t0> [--no-cache]\n"
        "\n"
        "  p / lambda : latitudine e longitudine (es. 40.85  14.27)\n"
        "  t / t0     : timestamp nel formato yyyymmddZhh00 (es. 20260601Z0600)\n"
        "  --no-cache : ignora ed esclude la cache su disco (./cache/)\n"
    )

    raw_args  = sys.argv[1:]
    use_cache = "--no-cache" not in raw_args
    args      = [a for a in raw_args if a != "--no-cache"]

    if len(args) != 4:
        print(USAGE)
        sys.exit(1)

    subcommand = args[0]
    if subcommand not in ("profile", "matrix"):
        print(f"Sottocomando non riconosciuto: '{subcommand}'\n")
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
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)