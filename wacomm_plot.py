"""
wacomm_plot.py
--------------
Visualizzazione dei risultati di wacomm_profile.py.

Utilizzo da riga di comando:
    python wacomm_plot.py profile      <p> <lambda> <t>   [output.png] [--print] [--max-depth N]
    python wacomm_plot.py matrix       <p> <lambda> <t0>  [output.png] [--print] [--max-depth N]
    python wacomm_plot.py matrix-lines <p> <lambda> <t0>  [output.png] [--print] [--max-depth N]
    python wacomm_plot.py totals       <p> <lambda> <t0>  [output.png] [--print]

Stessi argomenti di wacomm_profile.py.
--print          : stampa anche i dati numerici a schermo (come fa wacomm_profile.py),
                    oltre a generare il grafico.
--max-depth N    : profondità massima (in metri) mostrata sull'asse Y, default 50.
                    L'asse Y è lineare (non logaritmico).
--no-cache       : ignora ed esclude la cache su disco (./cache/)

La scala colori della concentrazione usa gli stessi livelli/colori dell'app
(file metacharts.json, campi 'clevels' e 'ccolors').
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Rende importabile wacomm_profile dallo stesso percorso di questo script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wacomm_profile import (
    get_concentration_profile,
    get_concentration_matrix,
    COPERNICUS_DEPTHS,
    _print_profile,
    _print_matrix_summary,
)

from config import METACHARTS_PATH, DEFAULT_MAX_DEPTH


# ── Scala colori della concentrazione (condivisa con l'app) ────────────────

def load_concentration_colormap(path: str = METACHARTS_PATH):
    """
    Carica la scala colori discreta della concentrazione da metacharts.json,
    usando gli stessi 'clevels' (soglie) e 'ccolors' (colori RGBA) dell'app.

    I clevels rappresentano i bordi superiori di ciascun intervallo di colore
    (36 livelli → 36 colori → 37 bordi, includendo lo 0 come bordo inferiore
    del primo intervallo).

    Parametri
    ----------
    path : str — percorso del file metacharts.json

    Restituisce
    -----------
    (cmap, norm, unit, label) :
        cmap  : matplotlib.colors.ListedColormap — colori discreti
        norm  : matplotlib.colors.BoundaryNorm   — bordi degli intervalli
        unit  : str — unità di misura (es. "#")
        label : str — etichetta della legenda (es. "Number of Particles")

    Eccezioni
    ---------
    FileNotFoundError — se il file non esiste
    KeyError          — se mancano i campi 'clevels' o 'ccolors'
    """
    with open(path, "r") as f:
        meta = json.load(f)["meta-chart"]

    clevels = meta["clevels"]
    ccolors = meta["ccolors"]   # RGBA 0-255

    if len(clevels) != len(ccolors):
        raise ValueError(
            f"clevels ({len(clevels)}) e ccolors ({len(ccolors)}) "
            "devono avere la stessa lunghezza"
        )

    # Colori normalizzati 0-1 per ListedColormap
    colors_norm = [[c / 255.0 for c in rgba] for rgba in ccolors]
    cmap = mcolors.ListedColormap(colors_norm)

    # Bordi degli intervalli: 0 come bordo inferiore, poi i clevels.
    # N livelli → N+1 bordi (l'ultimo clevel è già il bordo superiore
    # dell'ultimo colore, quindi i bordi sono [0] + clevels).
    boundaries = [0] + list(clevels)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    unit  = meta.get("unit_bars", "")
    label = meta.get("title_bars", "Concentrazione")

    return cmap, norm, unit, label


# ── Punto 1: profilo verticale ───────────────────────────────────────────────

def plot_profile(result: dict, p: float, lam: float, t: str,
                 save_path: str = None,
                 max_depth: float = DEFAULT_MAX_DEPTH) -> None:
    """
    Grafico del profilo verticale di concentrazione sui livelli Copernicus
    entro max_depth metri.

    Asse X : concentrazione [#particelle]  (zeri esclusi / trasparenti)
    Asse Y : profondità in metri, scala LINEARE, crescente verso il basso,
             limitata a [0, max_depth]
    Colore dei marker : stessa scala discreta usata dall'app (metacharts.json)

    Parametri
    ----------
    result    : dict restituito da get_concentration_profile()
    p, lam    : coordinate del punto richiesto
    t         : timestamp
    save_path : se fornito, salva su file invece di mostrare a schermo
    max_depth : profondità massima asse Y, in metri (default 50)
    """
    depths = np.array(result["depths"])            # (136,) in metri
    conc   = np.array(result["conc"], dtype=float) # (136,)

    # Limita ai livelli entro max_depth
    in_range = depths <= max_depth
    depths_r = depths[in_range]
    conc_r   = conc[in_range]

    # Zeri e NaN → non plottati (trasparenti)
    valid = (~np.isnan(conc_r)) & (conc_r > 0)

    cmap, norm, unit, cbar_label = load_concentration_colormap()

    fig, ax = plt.subplots(figsize=(6, 8))

    # Linea di base sottile a collegare i punti (solo dove ci sono dati)
    ax.plot(conc_r[valid], depths_r[valid],
            color="lightgray", linewidth=1, zorder=1)

    # Marker colorati secondo la stessa scala dell'app
    sc = ax.scatter(conc_r[valid], depths_r[valid],
                    c=conc_r[valid], cmap=cmap, norm=norm,
                    s=40, edgecolors="black", linewidths=0.4, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"{cbar_label} [{unit}]", fontsize=9)

    # Asse Y: LINEARE, crescente verso il basso, limitata a max_depth
    ax.set_ylim(max_depth, 0)
    ax.set_ylabel("Profondità [m] ↓", fontsize=11)
    ax.set_xlabel("Concentrazione [#]", fontsize=11)
    ax.set_title(
        f"Profilo verticale di concentrazione (0–{max_depth:.0f} m)\n"
        f"lat={result['lat_found']:.4f}°N   lon={result['lon_found']:.4f}°E\n"
        f"t = {t}",
        fontsize=11
    )
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotazione livelli validi / totali (nel range mostrato)
    ax.text(0.98, 0.02,
            f"{valid.sum()}/{in_range.sum()} livelli validi (conc > 0) "
            f"entro {max_depth:.0f}m",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Punto 2: matrice profondità × tempo ─────────────────────────────────────

def plot_matrix(result: dict, p: float, lam: float, t0: str,
                save_path: str = None,
                max_depth: float = DEFAULT_MAX_DEPTH) -> None:
    """
    Heatmap della matrice concentrazione, limitata ai livelli entro max_depth.

    Asse X : ore relative a t0, da -71 (sinistra) a 0 (destra = t0)
    Asse Y : profondità in metri, scala LINEARE, crescente verso il basso,
             limitata a [0, max_depth]
    Colore : stessa scala discreta usata dall'app (metacharts.json);
             0 = trasparente

    Parametri
    ----------
    result    : dict restituito da get_concentration_matrix()
    p, lam    : coordinate del punto richiesto
    t0        : timestamp finale
    save_path : se fornito, salva su file invece di mostrare a schermo
    max_depth : profondità massima asse Y, in metri (default 50)
    """
    mat_full   = result["matrix"]               # (136, 72)
    timestamps = result["timestamps"]           # lista 72 stringhe yyyymmddZhh00
    depths     = np.array(result["depths"])     # (136,) in metri
    n_hours    = mat_full.shape[1]               # 72

    # Limita ai livelli entro max_depth
    in_range = depths <= max_depth
    depths_r = depths[in_range]
    mat      = mat_full[in_range, :]

    # ── Asse X: ore relative, da -(n_hours-1) a 0 ───────────────────────────
    x_centers = np.arange(-(n_hours - 1), 1)           # [-71, -70, ..., 0]
    x_edges   = np.arange(-(n_hours - 1) - 0.5, 0.6)  # 73 bordi

    # ── Asse Y: bordi dei bin di profondità (solo livelli entro max_depth) ──
    d = depths_r
    if len(d) >= 2:
        d_edges = np.concatenate([
            [max(0.0, d[0] - (d[1] - d[0]) / 2)],
            (d[:-1] + d[1:]) / 2,
            [d[-1] + (d[-1] - d[-2]) / 2]
        ])
    else:
        d_edges = np.array([0.0, max_depth])

    # ── Concentrazione 0 → NaN (trasparente) ────────────────────────────────
    mat_plot = mat.copy()
    mat_plot[mat_plot <= 0] = np.nan

    # ── Colormap discreta condivisa con l'app ────────────────────────────────
    cmap, norm, unit, cbar_label = load_concentration_colormap()

    # ── Figura ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))

    im = ax.pcolormesh(
        x_edges, d_edges, mat_plot,
        cmap=cmap, norm=norm, shading="flat"
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(f"{cbar_label} [{unit}]", fontsize=10)

    # ── Asse Y: LINEARE, crescente verso il basso, limitata a max_depth ─────
    ax.set_ylim(max_depth, 0)
    ax.set_ylabel("Profondità [m] ↓", fontsize=11)

    # ── Asse X: valori negativi ogni 6 ore + etichetta data al cambio giorno ─
    step = 6
    tick_positions = x_centers[::step]          # es. [-71, -65, ..., -5, 0 (se step divide)]
    # Assicura che 0 sia sempre presente come ultima etichetta
    if 0 not in tick_positions:
        tick_positions = np.append(tick_positions, 0)

    tick_labels = []
    for rel_h in tick_positions:
        col = int(rel_h + (n_hours - 1))        # indice nella lista timestamps
        stamp = timestamps[col]
        hh    = stamp[9:11]
        # Mostra giorno/mese solo alla prima etichetta o al cambio di giorno
        prev_col = col - step
        if col == 0 or (prev_col >= 0 and timestamps[col][:8] != timestamps[prev_col][:8]):
            tick_labels.append(f"{stamp[6:8]}/{stamp[4:6]}\n{hh}:00 ({int(rel_h):+d}h)")
        else:
            tick_labels.append(f"{hh}:00\n({int(rel_h):+d}h)")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Ore rispetto a t₀ (UTC)", fontsize=11)

    ax.set_title(
        f"Concentrazione (0–{max_depth:.0f} m)  —  lat={result['lat_found']:.4f}°N   "
        f"lon={result['lon_found']:.4f}°E\n"
        f"t₀ = {t0}  |  range: {timestamps[0]}  →  {timestamps[-1]}",
        fontsize=11
    )
    ax.grid(True, which="major", linestyle="--", alpha=0.3, color="gray")

    # Annotazione file mancanti
    if result["missing_timestamps"]:
        ax.text(0.01, 0.02,
                f"File mancanti: {len(result['missing_timestamps'])}",
                transform=ax.transAxes, fontsize=8, color="red")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Punto 2bis: tutte le profondità come linee sovrapposte ──────────────────

def plot_matrix_lines(result: dict, p: float, lam: float, t0: str,
                      save_path: str = None,
                      max_depth: float = DEFAULT_MAX_DEPTH) -> None:
    """
    Grafico a linee della concentrazione nel tempo, una linea per ciascun
    livello di profondità Copernicus entro max_depth (tutte sullo stesso asse).

    Asse X : ore relative a t0, da -71 (sinistra) a 0 (destra = t0)
    Asse Y : concentrazione [#particelle]
    Colore delle linee : scala continua in base alla profondità
                         (chiaro = superficie, scuro = più profondo)

    Parametri
    ----------
    result    : dict restituito da get_concentration_matrix()
    p, lam    : coordinate del punto richiesto
    t0        : timestamp finale
    save_path : se fornito, salva su file invece di mostrare a schermo
    max_depth : profondità massima dei livelli mostrati, in metri (default 50)
    """
    mat_full   = result["matrix"]               # (136, n_hours)
    timestamps = result["timestamps"]
    depths     = np.array(result["depths"])     # (136,) in metri
    n_hours    = mat_full.shape[1]

    # Limita ai livelli entro max_depth
    in_range = depths <= max_depth
    depths_r = depths[in_range]                 # es. (14,) per max_depth=50
    mat      = mat_full[in_range, :]             # (n_livelli_r, n_hours)
    n_levels_r = mat.shape[0]

    x = np.arange(-(n_hours - 1), 1)             # [-71, -70, ..., 0]

    # ── Colormap continua basata sulla profondità ────────────────────────────
    # Chiaro (superficie) → scuro (profondo). 'Blues' va da quasi bianco a blu
    # scuro; normalizziamo sull'intervallo dei livelli mostrati (non su tutta
    # la scala 0-136) così la differenza tra le linee è ben visibile anche
    # con poche righe selezionate.
    cmap = plt.get_cmap("Blues")
    if n_levels_r > 1:
        depth_norm = mcolors.Normalize(vmin=depths_r.min(), vmax=depths_r.max())
    else:
        depth_norm = mcolors.Normalize(vmin=0, vmax=max(depths_r[0], 1))
    # Schiariamo il range inferiore: a vmin pieno il colore sarebbe troppo
    # chiaro/invisibile su sfondo bianco, quindi mappiamo su [0.25, 0.95]
    def color_for_depth(d):
        t = depth_norm(d)
        return cmap(0.25 + 0.70 * t)

    fig, ax = plt.subplots(figsize=(14, 7))

    for k in range(n_levels_r):
        y = mat[k, :]
        color = color_for_depth(depths_r[k])
        ax.plot(x, y, color=color, linewidth=1.3,
               label=f"{depths_r[k]:.1f} m")

    # ── Asse X: stessa convenzione di plot_matrix / plot_column_sums ────────
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
    ax.set_xlabel("Ore rispetto a t₀ (UTC)", fontsize=11)

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Concentrazione [#]", fontsize=11)
    ax.set_title(
        f"Concentrazione per livello di profondità (0–{max_depth:.0f} m)\n"
        f"lat={result['lat_found']:.4f}°N   lon={result['lon_found']:.4f}°E\n"
        f"t₀ = {t0}",
        fontsize=11
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legenda: una voce per livello, ordinata per profondità crescente ────
    # Con max_depth=50 sono circa 14 voci: gestibile in una colonna a lato.
    ax.legend(title="Profondità", loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, title_fontsize=9, frameon=False)

    # Annotazione file mancanti
    if result["missing_timestamps"]:
        ax.text(0.01, 0.97,
                f"File mancanti: {len(result['missing_timestamps'])} "
                "(interruzioni nelle serie)",
                transform=ax.transAxes, fontsize=8, color="red", va="top")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Punto 3: serie temporale della concentrazione totale ────────────────────

def plot_column_sums(result: dict, p: float, lam: float, t0: str,
                     save_path: str = None) -> None:
    """
    Area chart della concentrazione totale nella colonna d'acqua, ora per ora.

    Usa il vettore 'column_sums' già calcolato da get_concentration_matrix()
    (somma di tutti i 136 livelli per ciascuna ora).

    Asse X : ore relative a t0, da -71 (sinistra) a 0 (destra = t0)
    Asse Y : concentrazione totale [#particelle] (somma su tutta la colonna)
    L'ultimo punto (t0, l'ora di campionamento) è evidenziato in rosso,
    mentre il resto della serie è disegnato come area verde riempita,
    in stile simile a un grafico a funzione continua (non un istogramma).

    Parametri
    ----------
    result    : dict restituito da get_concentration_matrix()
                (deve contenere la chiave 'column_sums')
    p, lam    : coordinate del punto richiesto
    t0        : timestamp finale (ora di campionamento)
    save_path : se fornito, salva su file invece di mostrare a schermo
    """
    sums       = np.array(result["column_sums"], dtype=float)  # (72,)
    timestamps = result["timestamps"]                          # (72,)
    n_hours    = len(sums)

    x = np.arange(-(n_hours - 1), 1)   # [-71, -70, ..., 0]

    # I NaN (ore con file mancante) non vengono disegnati: la linea/area
    # si interrompe in quei punti. Sostituiamo momentaneamente con 0 solo
    # per il riempimento, mantenendo NaN nella linea per renderla visibile
    # come "buco".
    y = sums.copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Area verde riempita per tutti i punti tranne l'ultimo ───────────────
    # (l'ultimo, t0, viene evidenziato separatamente in rosso)
    x_main, y_main = x[:-1], y[:-1]
    x_last, y_last = x[-1],  y[-1]

    ax.fill_between(x_main, 0, y_main, color="#3a9b3a", alpha=0.85,
                    linewidth=0, zorder=1)
    ax.plot(x_main, y_main, color="#2e7d32", linewidth=1.3, zorder=2)

    # ── Ultimo punto (t0, ora di campionamento) in rosso ────────────────────
    # Disegnato come piccola area/barra rossa che collega l'ultimo valore
    # verde al punto finale, per restare fedele al riferimento fornito.
    if not np.isnan(y_last):
        ax.fill_between([x_main[-1], x_last], 0,
                        [y_main[-1] if not np.isnan(y_main[-1]) else 0, y_last],
                        color="#e53935", alpha=0.95, linewidth=0, zorder=3)
        ax.plot([x_main[-1], x_last],
               [y_main[-1] if not np.isnan(y_main[-1]) else 0, y_last],
               color="#c62828", linewidth=1.3, zorder=4)
        ax.scatter([x_last], [y_last], color="#c62828", s=25, zorder=5)

    # ── Asse X: stessa convenzione di plot_matrix (ore relative a t0) ───────
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
    ax.set_xlabel("Ore rispetto a t₀ (UTC)", fontsize=11)

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Concentrazione totale [#]", fontsize=11)
    ax.set_title(
        f"Concentrazione totale nella colonna d'acqua\n"
        f"lat={result['lat_found']:.4f}°N   lon={result['lon_found']:.4f}°E\n"
        f"t₀ = {t0}  (barra rossa)",
        fontsize=11
    )
    ax.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotazione file mancanti
    if result["missing_timestamps"]:
        ax.text(0.01, 0.95,
                f"File mancanti: {len(result['missing_timestamps'])} "
                "(interruzioni nella serie)",
                transform=ax.transAxes, fontsize=8, color="red", va="top")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Utilità ──────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str = None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    USAGE = (
        "Utilizzo:\n"
        "  python wacomm_plot.py profile      <p> <lambda> <t>   [output.png] [--print] [--max-depth N] [--no-cache]\n"
        "  python wacomm_plot.py matrix       <p> <lambda> <t0>  [output.png] [--print] [--max-depth N] [--no-cache]\n"
        "  python wacomm_plot.py matrix-lines <p> <lambda> <t0>  [output.png] [--print] [--max-depth N] [--no-cache]\n"
        "  python wacomm_plot.py totals       <p> <lambda> <t0>  [output.png] [--print] [--no-cache]\n"
        "\n"
        "  p / lambda    : latitudine e longitudine (es. 40.85  14.27)\n"
        "  t / t0        : timestamp yyyymmddZhh00  (es. 20230523Z0800)\n"
        "  output.png    : opzionale — se assente mostra il grafico a schermo\n"
        "  --print       : opzionale — stampa anche i dati numerici a schermo\n"
        "  --max-depth N : opzionale — profondità massima asse Y (matrix) o dei livelli\n"
        "                  mostrati come linee (matrix-lines), in metri (default 50)\n"
        "  --no-cache    : opzionale — ignora ed esclude la cache su disco (./cache/)\n"
    )

    raw_args = sys.argv[1:]

    # Estrae --print
    do_print = "--print" in raw_args
    if do_print:
        raw_args = [a for a in raw_args if a != "--print"]

    # Estrae --no-cache
    use_cache = "--no-cache" not in raw_args
    if not use_cache:
        raw_args = [a for a in raw_args if a != "--no-cache"]

    # Estrae --max-depth N (consuma due token consecutivi)
    max_depth = DEFAULT_MAX_DEPTH
    if "--max-depth" in raw_args:
        idx = raw_args.index("--max-depth")
        try:
            max_depth = float(raw_args[idx + 1])
        except (IndexError, ValueError):
            print("Errore: --max-depth richiede un valore numerico dopo di sé.\n")
            print(USAGE)
            sys.exit(1)
        del raw_args[idx:idx + 2]

    args = raw_args

    if len(args) not in (4, 5):
        print(USAGE)
        sys.exit(1)

    subcommand = args[0]
    if subcommand not in ("profile", "matrix", "matrix-lines", "totals"):
        print(f"Sottocomando non riconosciuto: '{subcommand}'\n")
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
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)