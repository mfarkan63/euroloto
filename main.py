#!/usr/bin/env python3
"""
Loto / EuroMillions — Analyse probabiliste et statistique
=========================================================
Usage examples:
  python main.py loto --analyze
  python main.py euro --predict 10 --alpha 0.6
  python main.py loto --visualize
  python main.py euro                          # tout faire (défaut)
"""
import argparse
import sys
import io
from pathlib import Path
from typing import List

# Force UTF-8 output on Windows to handle special characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import GAMES
from src.data_loader import load, load_all, ALL_CONFIG
from src import analyzer, models, visualizer


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_analyze(game_key: str):
    config = GAMES[game_key]
    df = load(config)
    date_min = df[config['date_col']].min().date()
    date_max = df[config['date_col']].max().date()

    print(f"\n{'='*60}")
    print(f"  {config['name']} — {len(df)} tirages  ({date_min} → {date_max})")
    print(f"{'='*60}\n")

    main_cols = config['main_cols']
    bonus_cols = config['bonus_cols']

    # --- Fréquences ---
    freq = analyzer.frequency(df, main_cols)
    print("Top 10 boules les plus fréquentes :")
    print(freq.nlargest(10).to_string())
    print()
    print("Top 5 boules les plus rares :")
    print(freq.nsmallest(5).to_string())
    print()

    freq_b = analyzer.frequency(df, bonus_cols)
    bonus_label = 'Numéro chance' if game_key == 'loto' else 'Étoiles'
    print(f"Fréquence {bonus_label} :")
    print(freq_b.sort_index().to_string())
    print()

    # --- Tests statistiques ---
    chi2 = analyzer.chi2_uniformity_test(freq, config['main_range'])
    print(f"Test χ² (uniformité boules) : stat={chi2['statistic']:.2f}  p={chi2['pvalue']:.4f}"
          f"  → {'Uniforme ✓' if chi2['is_uniform'] else 'Non uniforme ✗'}")

    ks = analyzer.ks_uniformity_test(df, main_cols, config['main_range'])
    print(f"Test KS (uniformité boules) : stat={ks['statistic']:.5f}  p={ks['pvalue']:.4f}"
          f"  → {'Uniforme ✓' if ks['is_uniform'] else 'Non uniforme ✗'}")
    print()

    # --- Somme ---
    ss = analyzer.sum_statistics(df, main_cols)
    print("Statistiques de la somme des boules :")
    print(ss.to_string())
    print()

    # --- Chaud / Froid ---
    hc = analyzer.hot_cold(df, main_cols)
    print("5 numéros les plus CHAUDS (50 derniers tirages) :")
    print(hc[hc['statut'] == 'chaud'].head(5)[['total_pct', 'recent_pct', 'delta']].to_string())
    print()
    print("5 numéros les plus FROIDS (50 derniers tirages) :")
    print(hc[hc['statut'] == 'froid'].tail(5)[['total_pct', 'recent_pct', 'delta']].to_string())
    print()

    # --- Pair / Impair ---
    eo = analyzer.even_odd_distribution(df, main_cols)
    print("Distribution pair / impair par tirage :")
    print(eo.to_string(index=False))
    print()

    # --- Haut / Bas ---
    hl = analyzer.high_low_distribution(df, main_cols, config['main_range'])
    print("Distribution haut / bas par tirage :")
    print(hl.to_string(index=False))
    print()

    # --- Paires les plus fréquentes ---
    pairs = analyzer.top_pairs(df, main_cols, n=10)
    print("Top 10 paires les plus co-occurentes :")
    print(pairs.to_string(index=False))
    print()

    # --- Sauvegarde rapport CSV ---
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    freq.rename('frequence').to_csv(reports_dir / f'{game_key}_frequence_boules.csv')
    freq_b.rename('frequence').to_csv(reports_dir / f'{game_key}_frequence_bonus.csv')
    hc.to_csv(reports_dir / f'{game_key}_chaud_froid.csv')
    pairs.to_csv(reports_dir / f'{game_key}_top_paires.csv', index=False)
    print(f"Rapports CSV sauvegardés dans reports/")


def cmd_predict(game_key: str, n: int, alpha: float, seed: int | None = None):
    config = GAMES[game_key]
    df = load(config)
    predictor = models.LotoPredictor(df, config)

    print(f"\n{'='*60}")
    print(f"  {config['name']} — {n} combinaisons candidates")
    print(f"  alpha={alpha}  (1.0=fréquence pure · 0.0=retard pur)")
    print(f"{'='*60}")
    print()

    bonus_label = 'Chance' if game_key == 'loto' else 'Étoiles'
    combos = predictor.generate_combinations(n=n, alpha=alpha, seed=seed)

    for i, c in enumerate(combos, 1):
        main_str = '  '.join(f'{x:02d}' for x in c['main'])
        bonus_str = '  '.join(f'{x:02d}' for x in c['bonus'])
        print(f"  #{i:02d}  [ {main_str} ]  +  {bonus_label}: [ {bonus_str} ]   score: {c['score']:.2f}")

    print()
    print("Top 10 numéros les plus fréquents :")
    print(predictor.top_numbers(10).to_string())
    print()
    print("Top 10 numéros les plus en retard :")
    print(predictor.overdue_numbers(10).to_string())


def cmd_visualize(game_key: str, min_cooc: int = 0):
    config = GAMES[game_key]
    df = load(config)
    print(f"\nGénération des graphiques pour {config['name']}...")

    tasks = [
        (visualizer.frequency_bar, (df, config['main_cols'], config, game_key), {'label': 'main'}),
        (visualizer.frequency_bar, (df, config['bonus_cols'], config, game_key), {'label': 'bonus'}),
        (visualizer.cooccurrence_heatmap, (df, config['main_cols'], config, game_key), {'min_cooc': min_cooc}),
        (visualizer.gap_boxplot, (df, config['main_cols'], config, game_key), {}),
        (visualizer.sum_histogram, (df, config['main_cols'], config, game_key), {}),
        (visualizer.hot_cold_chart, (df, config['main_cols'], config, game_key), {'n_recent': 50}),
        (visualizer.temporal_trend, (df, config['main_cols'], config, game_key), {}),
        (visualizer.retard_bar, (df, config['main_cols'], config, game_key), {}),
    ]

    for fn, args, kwargs in tasks:
        path = fn(*args, **kwargs)
        print(f"  ✓  {path.name}")

    print(f"\nTous les graphiques sont dans charts/")


def _generate_combinations(df, config, fixed, comp_freq, alpha, seed, game_key):
    """Generate 10 complete combinations with fixed numbers boosted by companion frequencies."""
    import numpy as np
    predictor = models.LotoPredictor(df, config)
    boosted_freq = predictor.main_freq.copy()
    for num, row in comp_freq.iterrows():
        if num in boosted_freq.index:
            boosted_freq[num] += row['frequence'] * 2

    rng = np.random.default_rng(seed)
    b_min, b_max = config['bonus_range']
    n_bonus = len(config['bonus_cols'])
    bonus_w = predictor._combined_weights(predictor.bonus_freq, predictor.bonus_last, alpha)
    bonus_nums = np.arange(b_min, b_max + 1)
    n_to_complete = len(config['main_cols']) - len(fixed)

    available = [n for n in predictor.main_nums if n not in fixed]
    avail_w = boosted_freq.reindex(available, fill_value=0).values.astype(float)
    if avail_w.sum() > 0:
        avail_w = avail_w / avail_w.sum()

    candidates, seen = [], set()
    attempts = 0
    while len(candidates) < 10 and attempts < 500:
        attempts += 1
        complement = sorted(rng.choice(available, size=n_to_complete, replace=False, p=avail_w).tolist())
        main = sorted(fixed + complement)
        key = tuple(main)
        if key in seen:
            continue
        seen.add(key)
        bonus = sorted(rng.choice(bonus_nums, size=n_bonus, replace=False, p=bonus_w).tolist())
        candidates.append({'main': main, 'bonus': bonus, 'score': predictor.score(main, bonus)})

    candidates.sort(key=lambda x: x['score'], reverse=True)
    bonus_label = 'Chance' if game_key == 'loto' else 'Étoiles'
    fixed_set = set(fixed)
    print(f"Combinaisons complètes (numéros fixes + {n_to_complete} meilleurs compléments) :")
    for i, c in enumerate(candidates, 1):
        main_str = '  '.join(f'*{x:02d}*' if x in fixed_set else f' {x:02d} ' for x in c['main'])
        bonus_str = '  '.join(f'{x:02d}' for x in c['bonus'])
        print(f"  #{i:02d}  [ {main_str} ]  +  {bonus_label}: [ {bonus_str} ]   score: {c['score']:.2f}")
    print("  (* = numéros fixés)")


def cmd_companions(game_key: str, fixed: List[int], n_top: int = 15, alpha: float = 0.6, seed=None):
    fixed = sorted(fixed)

    if game_key == 'all':
        _cmd_companions_all(fixed, n_top)
        return

    config = GAMES[game_key]
    df = load(config)
    main_cols = config['main_cols']

    print(f"\n{'='*60}")
    print(f"  {config['name']} — Compléments pour {fixed}")
    print(f"{'='*60}\n")

    filtered, comp_freq = analyzer.companions(df, main_cols, fixed, n_top=n_top)

    if len(filtered) == 0:
        print(f"  Aucun tirage historique ne contient simultanément {fixed}.")
        return

    print(f"  {len(filtered)} tirages historiques contiennent {fixed}\n")
    print(f"Top {n_top} compléments les plus fréquents :")
    print(comp_freq.to_string())
    print()

    _generate_combinations(df, config, fixed, comp_freq, alpha, seed, game_key)
    path = visualizer.companions_chart(filtered, comp_freq, fixed, config, game_key)
    print(f"\n  Graphique sauvegardé : {path.name}")


def _cmd_companions_all(fixed: List[int], n_top: int = 15):
    """Companions mode 'all': crosses both Loto and EuroMillions databases."""
    import pandas as pd
    from src.config import LOTO, EURO

    print(f"\n{'='*60}")
    print(f"  Loto + EuroMillions (croisé) — Compléments pour {fixed}")
    print(f"  Plage commune : 1–49  |  numéros bonus exclus")
    print(f"{'='*60}\n")

    df_loto = load(LOTO)
    df_euro = load(EURO)

    # Use all companions (no n_top limit) for accurate combined totals
    filtered_loto, comp_loto_full = analyzer.companions(df_loto, LOTO['main_cols'], fixed, n_top=999)
    filtered_euro, comp_euro_full = analyzer.companions(df_euro, EURO['main_cols'], fixed, n_top=999)

    n_loto = len(filtered_loto)
    n_euro = len(filtered_euro)
    n_total = n_loto + n_euro

    if n_total == 0:
        print(f"  Aucun tirage (dans aucun des deux jeux) ne contient simultanément {fixed}.")
        return

    print(f"  Tirages contenant {fixed} :")
    print(f"    Loto         : {n_loto:4d} / {len(df_loto):,}   ({n_loto/len(df_loto)*100:.2f}%)")
    print(f"    EuroMillions : {n_euro:4d} / {len(df_euro):,}   ({n_euro/len(df_euro)*100:.2f}%)")
    print(f"    TOTAL        : {n_total:4d}")
    print()

    # Build combined table by summing both sources directly (avoids 50-filter artifacts)
    freq_loto = comp_loto_full['frequence'].rename('loto') if not comp_loto_full.empty else pd.Series(dtype=int, name='loto')
    freq_euro = comp_euro_full['frequence'].rename('euro') if not comp_euro_full.empty else pd.Series(dtype=int, name='euro')

    combined = (
        pd.concat([freq_loto, freq_euro], axis=1)
        .fillna(0)
        .astype(int)
    )
    combined['total'] = combined['loto'] + combined['euro']
    combined['pct_%'] = (combined['total'] / n_total * 100).round(1)
    combined = combined.sort_values('total', ascending=False).head(n_top)
    combined = combined[['total', 'pct_%', 'loto', 'euro']]

    # Trimmed per-source views aligned on top combined numbers
    comp_loto_top = comp_loto_full.reindex(combined.index).fillna(0).astype({'frequence': int, 'pct': float})
    comp_euro_top = comp_euro_full.reindex(combined.index).fillna(0).astype({'frequence': int, 'pct': float})

    print(f"Top {n_top} compléments (combiné) :")
    print(combined.to_string())
    print()

    path = visualizer.companions_chart_all(
        filtered_loto, filtered_euro,
        comp_loto_top, comp_euro_top, combined,
        fixed, n_top,
    )
    print(f"  Graphique sauvegardé : {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyse probabiliste et statistique du Loto / EuroMillions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('game', choices=['loto', 'euro', 'all'],
                        help="Jeu à analyser. 'all' croise les deux bases (uniquement avec --companions)")
    parser.add_argument('--analyze', action='store_true', help='Analyse statistique complète')
    parser.add_argument('--predict', type=int, metavar='N', help='Générer N combinaisons candidates')
    parser.add_argument('--visualize', action='store_true', help='Générer tous les graphiques')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Poids fréquence vs retard pour la prédiction (0–1, défaut: 0.6)')
    parser.add_argument('--seed', type=int, default=None, help='Graine aléatoire (pour reproductibilité)')
    parser.add_argument('--companions', type=int, nargs='+', metavar='N',
                        help='Trouver les meilleurs compléments pour 1 ou 2 numéros fixés (ex: --companions 26 27)')
    parser.add_argument('--min-cooc', type=int, default=0, dest='min_cooc',
                        help='Seuil minimum de co-occurrences pour le graphique heatmap (défaut: 0 = tout afficher)')

    args = parser.parse_args()

    # 'all' est réservé à --companions
    if args.game == 'all' and not args.companions:
        parser.error("Le mode 'all' n'est disponible qu'avec --companions.")

    # Default: do everything (sauf si --companions seul ou mode all)
    if args.game != 'all' and not any([args.analyze, args.predict, args.visualize, args.companions]):
        args.analyze = True
        args.predict = 10
        args.visualize = True

    if args.analyze and args.game != 'all':
        cmd_analyze(args.game)
    if args.predict and args.game != 'all':
        cmd_predict(args.game, args.predict, args.alpha, args.seed)
    if args.visualize and args.game != 'all':
        cmd_visualize(args.game, min_cooc=args.min_cooc)
    if args.companions:
        cmd_companions(args.game, args.companions, alpha=args.alpha, seed=args.seed)


if __name__ == '__main__':
    main()
