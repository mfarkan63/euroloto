"""
euroloto — Analyse probabiliste et statistique du Loto et EuroMillions
======================================================================

Usage rapide :
    import euroloto
    euroloto.init()                              # charge config.yaml depuis le dossier courant
    euroloto.info()
    euroloto.frequency('loto')
    euroloto.prediction([26, 27], kind='loto')
    fig = euroloto.plot_frequency('loto')
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from euroloto import _state as _s
from euroloto._config import GAMES


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init(config_path: Union[str, Path] = 'config.yaml') -> None:
    """
    Load data from the paths declared in config.yaml.

    config.yaml format:
        loto:
          file: resultat-loto.xlsm
          sheet: Feuil1
        euro:
          file: resultat-euro.xlsm
          sheet: Feuil1

    The YAML can contain an optional `data_dir` key (default: directory of config.yaml).
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML requis : pip install pyyaml")

    cfg_path = Path(config_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {cfg_path}")

    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg.get('data_dir', cfg_path.parent))

    from euroloto._loader import load as _load
    from euroloto._config import GAMES as _GAMES

    _s.data_dir = data_dir
    _s.dfs.clear()
    _s.configs.clear()
    _s.invalidate()

    for kind in ('loto', 'euro'):
        if kind not in cfg:
            raise ValueError(f"Clé '{kind}' manquante dans {cfg_path}")
        entry = cfg[kind]
        game_cfg = dict(_GAMES[kind])  # copy so we can augment
        game_cfg['file'] = entry['file']
        game_cfg['sheet'] = entry.get('sheet', 'Feuil1')
        _s.configs[kind] = game_cfg
        _s.dfs[kind] = _load(kind, data_dir, entry['file'], entry.get('sheet', 'Feuil1'))

    print(f"euroloto initialisé depuis {cfg_path}")
    info()


def info() -> None:
    """Print a short summary of loaded data."""
    if not _s.dfs:
        print("Aucune donnée chargée. Appelez euroloto.init() d'abord.")
        return
    for kind, df in _s.dfs.items():
        cfg = _s.configs[kind]
        date_col = cfg['date_col']
        d_min = df[date_col].min().date()
        d_max = df[date_col].max().date()
        print(f"  {cfg['name']:16s} {len(df):5d} tirages  ({d_min} → {d_max})")


# ---------------------------------------------------------------------------
# Raw data access
# ---------------------------------------------------------------------------

def draws(kind: str) -> pd.DataFrame:
    """Return the raw draw DataFrame for 'loto' or 'euro'."""
    df, _ = _s.require(kind)
    return df.copy()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def frequency(kind: str) -> pd.Series:
    """Frequency of each main ball."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import frequency as _freq
    return _freq(df, cfg['main_cols'])


def hot_cold(kind: str, n_recent: int = 50) -> pd.DataFrame:
    """Hot/cold analysis comparing recent vs overall frequency."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import hot_cold as _hc
    return _hc(df, cfg['main_cols'], n_recent)


def cooccurrence(kind: str, top_n: int = 20, min_cooc: int = 0) -> pd.DataFrame:
    """Pairwise co-occurrence matrix (top_n most frequent numbers)."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import frequency as _freq, cooccurrence_matrix as _cooc
    top_nums = _freq(df, cfg['main_cols']).nlargest(top_n).index.tolist()
    matrix = _cooc(df, cfg['main_cols'])
    sub = matrix.loc[top_nums, top_nums].copy().astype(float)
    if min_cooc > 0:
        import numpy as np
        sub[sub < min_cooc] = np.nan
    return sub


def top_pairs(kind: str, n: int = 20) -> pd.DataFrame:
    """Top n most co-occurring number pairs."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import top_pairs as _tp
    return _tp(df, cfg['main_cols'], n)


def overdue(kind: str) -> pd.Series:
    """Draws since last appearance for each main ball, sorted descending."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import last_seen as _ls
    return _ls(df, cfg['main_cols']).sort_values(ascending=False)


def sum_stats(kind: str) -> pd.Series:
    """Descriptive statistics on the sum of drawn numbers per draw."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import sum_statistics as _ss
    return _ss(df, cfg['main_cols'])


def even_odd(kind: str) -> pd.DataFrame:
    """Distribution of even/odd counts per draw."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import even_odd_distribution as _eo
    return _eo(df, cfg['main_cols'])


def chi2_test(kind: str) -> dict:
    """Chi-squared uniformity test for main balls."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import frequency as _freq, chi2_uniformity_test as _chi2
    freq = _freq(df, cfg['main_cols'])
    return _chi2(freq, cfg['main_range'])


def ks_test(kind: str) -> dict:
    """Kolmogorov-Smirnov uniformity test for main balls."""
    df, cfg = _s.require(kind)
    from euroloto._analyzer import ks_uniformity_test as _ks
    return _ks(df, cfg['main_cols'], cfg['main_range'])


def companions(
    fixed: List[int],
    kind: str = 'loto',
    n_top: int = 15,
) -> pd.DataFrame:
    """
    Find the most frequent companion numbers for a given set of fixed numbers.

    kind='all' crosses both Loto and EuroMillions (common range 1–49).
    Returns a DataFrame with columns: frequence, pct[, loto, euro].
    """
    from euroloto._analyzer import companions as _comp

    fixed = sorted(fixed)

    if kind == 'all':
        return _companions_all(fixed, n_top)

    df, cfg = _s.require(kind)
    _, comp_freq = _comp(df, cfg['main_cols'], fixed, n_top=n_top)
    return comp_freq


def _companions_all(fixed: List[int], n_top: int) -> pd.DataFrame:
    """Cross-game companions analysis for numbers in range 1–49."""
    from euroloto._analyzer import companions as _comp

    df_loto, cfg_loto = _s.require('loto')
    df_euro, cfg_euro = _s.require('euro')

    _, comp_loto = _comp(df_loto, cfg_loto['main_cols'], fixed, n_top=999)
    _, comp_euro = _comp(df_euro, cfg_euro['main_cols'], fixed, n_top=999)

    freq_loto = comp_loto['frequence'].rename('loto') if not comp_loto.empty else pd.Series(dtype=int, name='loto')
    freq_euro = comp_euro['frequence'].rename('euro') if not comp_euro.empty else pd.Series(dtype=int, name='euro')

    df_loto_f, _ = _comp(df_loto, cfg_loto['main_cols'], fixed, n_top=999)
    df_euro_f, _ = _comp(df_euro, cfg_euro['main_cols'], fixed, n_top=999)
    n_total = len(df_loto_f) + len(df_euro_f)

    combined = (
        pd.concat([freq_loto, freq_euro], axis=1)
        .fillna(0).astype(int)
    )
    combined['total'] = combined['loto'] + combined['euro']
    combined['pct_%'] = (combined['total'] / n_total * 100).round(1) if n_total > 0 else 0.0
    return combined.sort_values('total', ascending=False).head(n_top)[['total', 'pct_%', 'loto', 'euro']]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def prediction(
    fixed: Optional[List[int]] = None,
    kind: str = 'loto',
    n: int = 10,
    alpha: float = 0.6,
    seed: Optional[int] = None,
) -> Union[List[dict], dict]:
    """
    Generate candidate combinations.

    fixed: optional list of numbers that must appear in every combination.
    kind:  'loto' | 'euro' | 'all' (returns {'loto': [...], 'euro': [...]}).
    alpha: 1.0 = pure frequency, 0.0 = pure recency (retard).

    Each combination is a dict: {'main': [...], 'bonus': [...], 'score': float}.
    """
    if kind == 'all':
        return {
            'loto': _predict_one('loto', fixed, n, alpha, seed),
            'euro': _predict_one('euro', fixed, n, alpha, seed),
        }
    return _predict_one(kind, fixed, n, alpha, seed)


def _predict_one(
    kind: str, fixed: Optional[List[int]], n: int, alpha: float, seed: Optional[int]
) -> List[dict]:
    from euroloto._analyzer import companions as _comp

    pred = _s.predictor(kind)

    if fixed:
        df, cfg = _s.require(kind)
        _, comp_freq = _comp(df, cfg['main_cols'], sorted(fixed), n_top=999)
        return pred.generate_with_fixed(sorted(fixed), comp_freq, n=n, alpha=alpha, seed=seed)

    return pred.generate_combinations(n=n, alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# Deep companions — cascaded conditional co-occurrence
# ---------------------------------------------------------------------------

def deep_companions(
    fixed: List[int],
    kind: str = 'euro',
    n_top: int = 5,
    k_seeds: int = 1,
) -> dict:
    """
    Iterative companion search: from a fixed pair to a full 5-number combination.

    k_seeds: number of most recent seed draws to use for the FILTRÉE filter.
             k_seeds=1 (default) = your manual procedure.
             k_seeds=3..10 = more robust signal, less noise.
             The union of companions across K seed draws is used as the filter set.

    At each step, the function compares two methods side by side:
    - GLOBAL   : all draws containing the current fixed set
    - FILTRÉE  : at step 1 only, draws that also share ≥1 number with the most
                 recent seed draw (the draw that gave the initial companions).
                 From step 2 onward both methods use unconstrained draws.

    kind='all' crosses Loto + EuroMillions (seed drawn from EuroMillions).

    Returns a dict with keys: seed_draw, seed_date, seed_companions,
    steps (list), combo_global (List[int]), combo_filtered (List[int]).
    """
    from euroloto._analyzer import companions as _comp, companions_with_seed as _comp_seed

    fixed = sorted(fixed)

    # --- Resolve databases and seed ---
    if kind == 'all':
        df_seed, cfg_seed = _s.require('euro')   # seed always from euro
        df_loto, cfg_loto = _s.require('loto')
        df_euro, cfg_euro = df_seed, cfg_seed
    else:
        df_seed, cfg_seed = _s.require(kind)

    cols_seed = cfg_seed['main_cols']
    date_col  = cfg_seed['date_col']

    # Find most recent draw containing all fixed numbers
    seed_mask = pd.Series([True] * len(df_seed), index=df_seed.index)
    for num in fixed:
        seed_mask &= (df_seed[cols_seed] == num).any(axis=1)
    seed_draws = df_seed[seed_mask].sort_values(date_col)

    if seed_draws.empty:
        print(f"Aucun tirage contenant {fixed} dans {cfg_seed['name']}.")
        return {}

    # Use the K most recent seed draws — union of their companions as the filter set
    k = max(1, k_seeds)
    recent_seeds = seed_draws.tail(k)
    seed_companions_set: set = set()
    for _, row in recent_seeds.iterrows():
        for n in row[cols_seed].astype(int).tolist():
            if n not in fixed:
                seed_companions_set.add(n)
    seed_companions = sorted(seed_companions_set)

    # For display: show each seed draw
    seed_row       = recent_seeds.iloc[-1]
    seed_date      = seed_row[date_col].date()
    seed_draw_full = sorted(seed_row[cols_seed].astype(int).tolist())

    game_label = 'Loto + EuroMillions' if kind == 'all' else cfg_seed['name']
    print(f"\n{'='*65}")
    print(f"  deep_companions {fixed} — {game_label}  (k_seeds={k})")
    print(f"{'='*65}")
    print(f"\nTirages graines ({cfg_seed['name']}) :")
    for _, row in recent_seeds.iterrows():
        d = row[date_col].date()
        draw = sorted(row[cols_seed].astype(int).tolist())
        print(f"  {d}  →  {draw}")
    print(f"  Compagnons graine (union) : {seed_companions}")

    # --- Helper: get companions for current path ---
    def _get_companions(path, min_one_of=None):
        if kind == 'all':
            # combine loto + euro companions
            _, cL = _comp(df_loto, cfg_loto['main_cols'], path, n_top=999)
            _, cE = _comp(df_euro, cfg_euro['main_cols'], path, n_top=999)
            if min_one_of:
                _, cL = _comp_seed(df_loto, cfg_loto['main_cols'], path, min_one_of, n_top=999)
                _, cE = _comp_seed(df_euro, cfg_euro['main_cols'], path, min_one_of, n_top=999)
            fL = cL['frequence'].rename('loto')  if not cL.empty else pd.Series(dtype=int, name='loto')
            fE = cE['frequence'].rename('euro')  if not cE.empty else pd.Series(dtype=int, name='euro')
            combined = pd.concat([fL, fE], axis=1).fillna(0).astype(int)
            combined['total'] = combined['loto'] + combined['euro']
            # filter out fixed numbers
            combined = combined[~combined.index.isin(path)]
            return combined.sort_values('total', ascending=False).head(n_top)
        else:
            df_g, cfg_g = _s.require(kind)
            if min_one_of:
                filt, comp = _comp_seed(df_g, cfg_g['main_cols'], path, min_one_of, n_top=n_top)
            else:
                filt, comp = _comp(df_g, cfg_g['main_cols'], path, n_top=n_top)
            return comp

    # --- Cascade ---
    n_main        = len(cols_seed)
    path_global   = list(fixed)
    path_filtered = list(fixed)
    steps_result  = []

    for step_idx in range(len(fixed), n_main):
        step_num = step_idx - len(fixed) + 1
        position = step_idx + 1

        print(f"\n{'─'*65}")
        ordinal = {1:'1er', 2:'2ème', 3:'3ème'}.get(position, f'{position}ème')
        print(f"ÉTAPE {step_num} — {ordinal} numéro")

        # GLOBAL: unconstrained companions for current best paths
        comp_g = _get_companions(path_global)

        # FILTRÉE: only at first step, apply seed filter
        remaining_seeds = [s for s in seed_companions if s not in path_filtered]
        if remaining_seeds and step_idx == len(fixed):
            comp_f = _get_companions(path_filtered, min_one_of=remaining_seeds)
            filter_label = f"avec ≥1 de {{{', '.join(map(str, remaining_seeds))}}}"
        else:
            comp_f = _get_companions(path_filtered)
            filter_label = "idem global (filtre épuisé)"

        # --- Display side by side ---
        freq_col = 'total' if kind == 'all' else 'frequence'

        g_top = list(comp_g.index[:n_top]) if not comp_g.empty else []
        f_top = list(comp_f.index[:n_top]) if not comp_f.empty else []
        max_rows = max(len(g_top), len(f_top), 1)

        col_w = 26
        print(f"\n  {'GLOBAL':<{col_w}}  FILTRÉE ({filter_label})")
        print(f"  {'fixés: ' + str(path_global):<{col_w}}  fixés: {path_filtered}")
        print(f"  {'─'*24}  {'─'*24}")
        header = f"  {'num':>4}  {'freq':>5}  {'pct':>6}    {'num':>4}  {'freq':>5}  {'pct':>6}"
        print(header)
        print(f"  {'─'*24}  {'─'*24}")

        for i in range(max_rows):
            # global side
            if i < len(g_top):
                num_g = g_top[i]
                frq_g = int(comp_g.loc[num_g, freq_col])
                pct_g = '' if kind == 'all' else (
                    f"{comp_g.loc[num_g, 'pct']:.1f}%" if 'pct' in comp_g.columns else ''
                )
                g_str = f"  {num_g:>4}  {frq_g:>5}  {pct_g:>6}"
            else:
                g_str = f"  {'':>4}  {'':>5}  {'':>6}"

            # filtered side
            if i < len(f_top):
                num_f = f_top[i]
                frq_f = int(comp_f.loc[num_f, freq_col])
                pct_f = f"{comp_f.loc[num_f, 'pct']:.1f}%" if 'pct' in comp_f.columns else ''
                if kind == 'all':
                    pct_f = ''
                f_str = f"  {num_f:>4}  {frq_f:>5}  {pct_f:>6}"
            else:
                f_str = f"  {'':>4}  {'':>5}  {'':>6}"

            marker = '  ←' if (g_top and f_top and i == 0 and g_top[0] != f_top[0]) else ''
            print(f"{g_str}  {f_str}{marker}")

        # Advance both paths with their top candidate
        best_g = g_top[0] if g_top else None
        best_f = f_top[0] if f_top else None
        if best_g:
            path_global   = sorted(path_global   + [best_g])
        if best_f:
            path_filtered = sorted(path_filtered + [best_f])

        steps_result.append({
            'step': step_num,
            'position': position,
            'global':   comp_g,
            'filtered': comp_f,
        })

    print(f"\n{'='*65}")
    print(f"Combinaison finale GLOBALE   : {path_global}")
    print(f"Combinaison finale FILTRÉE   : {path_filtered}")
    if path_global == path_filtered:
        print("  → Les deux méthodes convergent vers la même combinaison.")

    return {
        'seed_draw':        seed_draw_full,
        'seed_date':        str(seed_date),
        'seed_companions':  seed_companions,
        'steps':            steps_result,
        'combo_global':     path_global,
        'combo_filtered':   path_filtered,
    }


def plot_deep_companions(
    fixed: List[int],
    kind: str = 'euro',
    n_top: int = 15,
    min_affinity: int = 1,
    metric: str = 'count',
) -> List:
    """
    Cascade of affinity heatmaps: one figure per step (pair → triplet → quadruplet).

    At each step, the heatmap shows the conditional co-occurrence matrix among
    the top companion candidates in draws containing the current fixed set.
    The top candidate is auto-selected (by frequency) to advance to the next step.

    metric='count' : raw co-occurrence count (default).
    metric='lift'  : lift = P(x∩y|fixed) / (P(x|fixed)×P(y|fixed))
                     Values > 1 = genuine affinity beyond base rates.
    min_affinity   : pairs below this value are masked (white cells).
                     For lift, threshold is automatically set to 1.0.

    Returns a list of matplotlib Figures (one per cascade step).
    """
    from euroloto._analyzer import companions as _comp
    from euroloto._plots import affinity_heatmap

    fixed = sorted(fixed)
    figures = []

    def _get_filtered(path):
        if kind == 'all':
            df_l, cfg_l = _s.require('loto')
            df_e, cfg_e = _s.require('euro')
            mask_l = pd.Series([True] * len(df_l), index=df_l.index)
            mask_e = pd.Series([True] * len(df_e), index=df_e.index)
            for n in path:
                mask_l &= (df_l[cfg_l['main_cols']] == n).any(axis=1)
                mask_e &= (df_e[cfg_e['main_cols']] == n).any(axis=1)
            # Combine on common columns
            COMMON = ['date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
            fl = df_l[mask_l][COMMON].copy()
            fe = df_e[mask_e][COMMON].copy()
            return pd.concat([fl, fe], ignore_index=True), COMMON[1:], cfg_l
        else:
            df, cfg = _s.require(kind)
            mask = pd.Series([True] * len(df), index=df.index)
            for n in path:
                mask &= (df[cfg['main_cols']] == n).any(axis=1)
            return df[mask].copy(), cfg['main_cols'], cfg

    current_fixed = list(fixed)
    n_main = len(_s.require('loto' if kind == 'all' else kind)[1]['main_cols'])

    for step_idx in range(len(fixed), n_main):
        step_num = step_idx - len(fixed) + 1
        ordinal = {1: '1er', 2: '2ème', 3: '3ème'}.get(step_num, f'{step_num}ème')
        step_label = f'Étape {step_num} — {ordinal} numéro  |  '

        filtered_df, cols, cfg = _get_filtered(current_fixed)

        fig = affinity_heatmap(
            filtered_df, cols, current_fixed, cfg,
            n_top=n_top, min_affinity=min_affinity,
            metric=metric, step_label=step_label,
        )
        figures.append(fig)

        # Advance with top companion by frequency
        _, comp = _comp(filtered_df, cols, current_fixed, n_top=1)
        if comp.empty:
            break
        best = comp.index[0]
        current_fixed = sorted(current_fixed + [best])

    return figures


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_frequency(kind: str, label: str = 'main'):
    """Bar chart of main (label='main') or bonus (label='bonus') frequencies."""
    import matplotlib.pyplot as plt
    df, cfg = _s.require(kind)
    cols = cfg['main_cols'] if label == 'main' else cfg['bonus_cols']
    from euroloto._plots import frequency_bar
    return frequency_bar(df, cols, cfg, label)


def plot_hot_cold(kind: str, n_recent: int = 50):
    """Hot/cold deviation bar chart."""
    df, cfg = _s.require(kind)
    from euroloto._plots import hot_cold_chart
    return hot_cold_chart(df, cfg['main_cols'], cfg, n_recent)


def plot_cooccurrence(kind: str, top_n: int = 20, min_cooc: int = 0):
    """Co-occurrence heatmap."""
    df, cfg = _s.require(kind)
    from euroloto._plots import cooccurrence_heatmap
    return cooccurrence_heatmap(df, cfg['main_cols'], cfg, top_n, min_cooc)


def plot_gaps(kind: str):
    """Boxplot of inter-draw gaps per number."""
    df, cfg = _s.require(kind)
    from euroloto._plots import gap_boxplot
    return gap_boxplot(df, cfg['main_cols'], cfg)


def plot_sum(kind: str):
    """Histogram of draw sums with normal fit overlay."""
    df, cfg = _s.require(kind)
    from euroloto._plots import sum_histogram
    return sum_histogram(df, cfg['main_cols'], cfg)


def plot_temporal(kind: str, top_n: int = 10):
    """Normalized yearly frequency trend for the top_n most frequent numbers."""
    df, cfg = _s.require(kind)
    from euroloto._plots import temporal_trend
    return temporal_trend(df, cfg['main_cols'], cfg, top_n)


def plot_overdue(kind: str):
    """Horizontal bar chart of the 20 most overdue numbers."""
    df, cfg = _s.require(kind)
    from euroloto._plots import retard_bar
    return retard_bar(df, cfg['main_cols'], cfg)


def plot_companions(fixed: List[int], kind: str = 'loto', n_top: int = 15):
    """
    Bar chart of companion frequencies.
    kind='all' produces a stacked bar crossing both games.
    """
    from euroloto._analyzer import companions as _comp

    fixed = sorted(fixed)

    if kind == 'all':
        df_loto, cfg_loto = _s.require('loto')
        df_euro, cfg_euro = _s.require('euro')
        filtered_loto, comp_loto_full = _comp(df_loto, cfg_loto['main_cols'], fixed, n_top=999)
        filtered_euro, comp_euro_full = _comp(df_euro, cfg_euro['main_cols'], fixed, n_top=999)
        comp_combined = _companions_all(fixed, n_top)
        comp_loto_top = comp_loto_full.reindex(comp_combined.index).fillna(0).astype({'frequence': int, 'pct': float})
        comp_euro_top = comp_euro_full.reindex(comp_combined.index).fillna(0).astype({'frequence': int, 'pct': float})
        from euroloto._plots import companions_bar_all
        return companions_bar_all(filtered_loto, filtered_euro, comp_loto_top, comp_euro_top, comp_combined, fixed)

    df, cfg = _s.require(kind)
    filtered, comp_freq = _comp(df, cfg['main_cols'], fixed, n_top=n_top)
    if comp_freq.empty:
        raise ValueError(f"Aucun tirage contient simultanément {fixed}.")
    from euroloto._plots import companions_bar
    return companions_bar(filtered, comp_freq, fixed, cfg)
