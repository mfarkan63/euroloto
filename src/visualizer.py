"""
Visualization functions — each returns the saved Path.
All charts are saved to charts/<game>_<name>.png.
"""
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import analyzer

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.0)
CHART_DIR = Path(__file__).parent.parent / 'charts'


def _save(fig: plt.Figure, name: str, game: str) -> Path:
    CHART_DIR.mkdir(exist_ok=True)
    path = CHART_DIR / f'{game}_{name}.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path


def frequency_bar(
    df: pd.DataFrame, cols: List[str], config: dict, game: str, label: str = 'main'
) -> Path:
    """Bar chart of number frequencies with hot/cold color coding."""
    n_min, n_max = config['main_range'] if label == 'main' else config['bonus_range']
    freq = analyzer.frequency(df, cols).reindex(range(n_min, n_max + 1), fill_value=0)
    mean = freq.mean()
    threshold = 0.10

    colors = [
        '#e74c3c' if v > mean * (1 + threshold)
        else '#3498db' if v < mean * (1 - threshold)
        else '#95a5a6'
        for v in freq
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(freq.index, freq.values, color=colors, edgecolor='white', linewidth=0.4)
    ax.axhline(mean, color='#f39c12', linestyle='--', linewidth=1.8, zorder=5)

    ax.set_xlabel('Numéro', fontsize=12)
    ax.set_ylabel('Nombre d\'apparitions', fontsize=12)
    title_label = 'Boules principales' if label == 'main' else 'Numéros bonus'
    ax.set_title(
        f'{config["name"]} — Fréquence des {title_label} ({len(df)} tirages)',
        fontsize=14, fontweight='bold',
    )

    legend = [
        mpatches.Patch(color='#e74c3c', label=f'Chaud (> moy +{threshold*100:.0f}%)'),
        mpatches.Patch(color='#3498db', label=f'Froid (< moy -{threshold*100:.0f}%)'),
        mpatches.Patch(color='#95a5a6', label='Neutre'),
        plt.Line2D([0], [0], color='#f39c12', linestyle='--', linewidth=2,
                   label=f'Moyenne : {mean:.1f}'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=9)
    return _save(fig, f'frequency_{label}', game)


def cooccurrence_heatmap(
    df: pd.DataFrame,
    cols: List[str],
    config: dict,
    game: str,
    top_n: int = 20,
    min_cooc: int = 0,
) -> Path:
    """
    Heatmap of pairwise co-occurrence for the top_n most frequent numbers.
    min_cooc: pairs with fewer co-occurrences are masked (shown in white).
              Numbers with no pair above the threshold are removed from the heatmap.
    """
    freq = analyzer.frequency(df, cols)
    top_nums = freq.nlargest(top_n).index.tolist()
    matrix = analyzer.cooccurrence_matrix(df, cols)
    sub = matrix.loc[top_nums, top_nums].copy().astype(float)

    if min_cooc > 0:
        sub[sub < min_cooc] = np.nan
        # Drop rows/columns where every pair is masked
        has_valid = sub.notna().any(axis=1)
        sub = sub.loc[has_valid, has_valid]

    if sub.empty:
        raise ValueError(f"Aucune paire avec min_cooc >= {min_cooc}. Réduisez le seuil.")

    # Always use float format since matrix may contain NaN (float)
    fmt = '.0f'
    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.6), max(8, len(sub) * 0.5)))
    sns.heatmap(
        sub, annot=True, fmt=fmt, cmap='YlOrRd', ax=ax,
        linewidths=0.4, cbar_kws={'label': 'Co-occurrences'},
        annot_kws={'size': 8}, mask=sub.isna(),
    )
    threshold_str = f' (seuil ≥ {min_cooc})' if min_cooc > 0 else f' (top {top_n} numéros)'
    ax.set_title(
        f'{config["name"]} — Co-occurrences{threshold_str}',
        fontsize=13, fontweight='bold',
    )
    suffix = f'cooccurrence_min{min_cooc}' if min_cooc > 0 else 'cooccurrence'
    return _save(fig, suffix, game)


def gap_boxplot(df: pd.DataFrame, cols: List[str], config: dict, game: str) -> Path:
    """Boxplot of inter-draw gaps for each number."""
    gaps = analyzer.gap_analysis(df, cols)
    data = {k: v for k, v in sorted(gaps.items()) if len(v) >= 5}
    labels = list(data.keys())
    values = list(data.values())

    n_min, n_max = config['main_range']
    n_drawn = len(cols)
    expected_gap = (n_max - n_min + 1) / n_drawn

    fig, ax = plt.subplots(figsize=(20, 6))
    bp = ax.boxplot(
        values, labels=labels, patch_artist=True,
        boxprops=dict(facecolor='#3498db', alpha=0.6),
        medianprops=dict(color='#e74c3c', linewidth=2),
        whiskerprops=dict(linewidth=1),
        flierprops=dict(marker='o', markersize=2, alpha=0.4),
    )
    ax.axhline(
        expected_gap, color='#f39c12', linestyle='--', linewidth=1.8,
        label=f'Écart théorique : {expected_gap:.1f} tirages',
    )
    ax.set_xlabel('Numéro', fontsize=12)
    ax.set_ylabel('Tirages entre deux apparitions', fontsize=12)
    ax.set_title(
        f'{config["name"]} — Distribution des écarts entre apparitions',
        fontsize=14, fontweight='bold',
    )
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.legend(fontsize=10)
    return _save(fig, 'gap_boxplot', game)


def sum_histogram(df: pd.DataFrame, cols: List[str], config: dict, game: str) -> Path:
    """Histogram of draw sums with normal fit overlay."""
    from scipy import stats as scipy_stats

    sums = df[cols].sum(axis=1)
    mu, sigma = sums.mean(), sums.std()

    fig, ax = plt.subplots(figsize=(12, 5))
    n, bins, _ = ax.hist(
        sums, bins=45, density=True, color='#3498db', edgecolor='white',
        alpha=0.75, label='Tirages observés',
    )
    x = np.linspace(sums.min(), sums.max(), 300)
    ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma), color='#e74c3c', linewidth=2,
            label=f'Normale N({mu:.0f}, {sigma:.1f})')
    ax.axvline(mu, color='#f39c12', linestyle='--', linewidth=1.8, label=f'Moyenne : {mu:.1f}')

    ax.set_xlabel('Somme des boules', fontsize=12)
    ax.set_ylabel('Densité', fontsize=12)
    ax.set_title(
        f'{config["name"]} — Distribution de la somme des boules tirées',
        fontsize=14, fontweight='bold',
    )
    ax.legend(fontsize=10)
    return _save(fig, 'sum_distribution', game)


def hot_cold_chart(
    df: pd.DataFrame, cols: List[str], config: dict, game: str, n_recent: int = 50
) -> Path:
    """Bar chart showing hot/cold deviation for last n_recent draws."""
    hc = analyzer.hot_cold(df, cols, n_recent)
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in hc['delta']]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(hc.index.astype(str), hc['delta'], color=colors, edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Numéro', fontsize=12)
    ax.set_ylabel('Δ% (récent − global)', fontsize=12)
    ax.set_title(
        f'{config["name"]} — Numéros chauds / froids (derniers {n_recent} tirages)',
        fontsize=14, fontweight='bold',
    )
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    legend = [
        mpatches.Patch(color='#e74c3c', label='Chaud (surreprésenté récemment)'),
        mpatches.Patch(color='#3498db', label='Froid (sous-représenté récemment)'),
    ]
    ax.legend(handles=legend, fontsize=10)
    return _save(fig, f'hot_cold_{n_recent}', game)


def temporal_trend(
    df: pd.DataFrame, cols: List[str], config: dict, game: str, top_n: int = 10
) -> Path:
    """
    Normalized yearly frequency chart (appearances per 100 draws).
    Normalization removes the effect of draw-frequency changes (e.g. EuroMillions
    going from 1 to 2 draws/week in 2011).
    Structural change years defined in config['structural_changes'] are annotated.
    """
    date_col = config['date_col']

    # Count total draws per year for normalization
    draws_per_year = (
        df.groupby(df[date_col].dt.year).size().rename('n_draws')
    )

    raw = analyzer.temporal_frequency(df, cols, date_col, 'Y')

    # Normalize: occurrences / total_draws_that_year * 100
    norm = raw.copy().astype(float)
    for period_str in norm.index:
        year = int(period_str[:4])
        if year in draws_per_year.index and draws_per_year[year] > 0:
            norm.loc[period_str] = raw.loc[period_str] / draws_per_year[year] * 100

    top_nums = analyzer.frequency(df, cols).nlargest(top_n).index
    years_numeric = [int(p[:4]) for p in norm.index]

    fig, ax = plt.subplots(figsize=(14, 6))

    for num in top_nums:
        if num in norm.columns:
            ax.plot(
                years_numeric, norm[num].values,
                marker='o', markersize=4, label=str(num), alpha=0.85,
            )

    # Annotate structural changes
    for change_year, label in config.get('structural_changes', {}).items():
        ax.axvline(change_year - 0.5, color='#e74c3c', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.text(change_year - 0.3, ax.get_ylim()[1] * 0.97, label,
                color='#e74c3c', fontsize=8, va='top')

    # Expected frequency per draw = n_cols / (n_max - n_min + 1) * 100
    n_min, n_max = config['main_range']
    expected_pct = len(cols) / (n_max - n_min + 1) * 100
    ax.axhline(expected_pct, color='#95a5a6', linestyle=':', linewidth=1.2,
               label=f'Attendu théorique : {expected_pct:.2f}%')

    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Fréquence normalisée (% par tirage)', fontsize=12)
    ax.set_title(
        f'{config["name"]} — Tendance temporelle des {top_n} numéros les plus fréquents\n'
        f'(normalisée par nombre de tirages annuels)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    fig.tight_layout()
    return _save(fig, 'temporal_trend', game)


def companions_chart(
    filtered_df: pd.DataFrame,
    comp_freq: pd.DataFrame,
    fixed: List[int],
    config: dict,
    game: str,
) -> Path:
    """Bar chart of companion number frequencies for a given fixed pair."""
    if comp_freq.empty:
        raise ValueError("Aucun tirage trouvé pour cette combinaison fixe.")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c' if i < 3 else '#3498db' for i in range(len(comp_freq))]
    ax.bar(comp_freq.index.astype(str), comp_freq['frequence'], color=colors,
           edgecolor='white', linewidth=0.4)

    for i, (num, row) in enumerate(comp_freq.iterrows()):
        ax.text(i, row['frequence'] + 0.2, f"{row['pct']:.0f}%",
                ha='center', va='bottom', fontsize=8)

    fixed_str = ' + '.join(str(n) for n in sorted(fixed))
    ax.set_xlabel('Numéro complémentaire', fontsize=12)
    ax.set_ylabel(f'Apparitions avec [{fixed_str}]', fontsize=12)
    ax.set_title(
        f'{config["name"]} — Compléments de [{fixed_str}]\n'
        f'({len(filtered_df)} tirages historiques contiennent cette paire)',
        fontsize=13, fontweight='bold',
    )
    legend = [
        mpatches.Patch(color='#e74c3c', label='Top 3'),
        mpatches.Patch(color='#3498db', label='Autres'),
    ]
    ax.legend(handles=legend, fontsize=10)
    suffix = 'companions_' + '_'.join(str(n) for n in sorted(fixed))
    return _save(fig, suffix, game)


def companions_chart_all(
    filtered_loto: pd.DataFrame,
    filtered_euro: pd.DataFrame,
    comp_loto: pd.DataFrame,
    comp_euro: pd.DataFrame,
    comp_all: pd.DataFrame,
    fixed: List[int],
    n_top: int = 15,
) -> Path:
    """
    Stacked bar chart of companion frequencies from both Loto and EuroMillions.
    Each bar shows the loto contribution (bottom) and euro contribution (top).
    """
    top_nums = comp_all.index.tolist()[:n_top]

    # comp_loto / comp_euro aligned on top_nums (frequence column)
    loto_col = 'frequence' if 'frequence' in comp_loto.columns else 'loto'
    euro_col = 'frequence' if 'frequence' in comp_euro.columns else 'euro'
    loto_vals = comp_loto[loto_col].reindex(top_nums, fill_value=0) if not comp_loto.empty else pd.Series(0, index=top_nums)
    euro_vals = comp_euro[euro_col].reindex(top_nums, fill_value=0) if not comp_euro.empty else pd.Series(0, index=top_nums)

    x = np.arange(len(top_nums))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(12, len(top_nums) * 0.8), 6))
    ax.bar(x, loto_vals.values, width, color='#3498db', edgecolor='white',
           linewidth=0.4, label=f'Loto ({len(filtered_loto)} tirages)')
    ax.bar(x, euro_vals.values, width, bottom=loto_vals.values,
           color='#e74c3c', edgecolor='white', linewidth=0.4,
           label=f'EuroMillions ({len(filtered_euro)} tirages)')

    pct_col = 'pct_%' if 'pct_%' in comp_all.columns else 'pct'
    for i, num in enumerate(top_nums):
        total = int(loto_vals.iloc[i] + euro_vals.iloc[i])
        pct = float(comp_all.loc[num, pct_col]) if num in comp_all.index else 0
        ax.text(i, total + 0.15, f'{total}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in top_nums])
    ax.set_xlabel('Numéro complémentaire', fontsize=12)
    fixed_str = ' + '.join(str(n) for n in sorted(fixed))
    n_total = len(filtered_loto) + len(filtered_euro)
    ax.set_ylabel(f'Apparitions combinées avec [{fixed_str}]', fontsize=12)
    ax.set_title(
        f'Loto + EuroMillions — Compléments de [{fixed_str}]\n'
        f'({n_total} tirages croisés : {len(filtered_loto)} Loto + {len(filtered_euro)} EuroMillions)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=10)
    suffix = 'companions_all_' + '_'.join(str(n) for n in sorted(fixed))
    return _save(fig, suffix, 'all')


def retard_bar(df: pd.DataFrame, cols: List[str], config: dict, game: str) -> Path:
    """Horizontal bar chart of draws since last appearance for each number."""
    last = analyzer.last_seen(df, cols).sort_values(ascending=False)
    n_min, n_max = config['main_range']
    full = last.reindex(range(n_min, n_max + 1), fill_value=len(df))

    # Show only top 20 most overdue
    top = full.nlargest(20)
    expected = (n_max - n_min + 1) / len(cols)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if v > expected * 2 else '#f39c12' if v > expected else '#3498db'
              for v in top.values]
    ax.barh(top.index.astype(str), top.values, color=colors, edgecolor='white')
    ax.axvline(expected, color='#2ecc71', linestyle='--', linewidth=1.8,
               label=f'Retard moyen théorique : {expected:.0f}')
    ax.set_xlabel('Nombre de tirages depuis la dernière apparition', fontsize=11)
    ax.set_ylabel('Numéro', fontsize=11)
    ax.set_title(
        f'{config["name"]} — Top 20 numéros les plus en retard',
        fontsize=14, fontweight='bold',
    )
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    return _save(fig, 'retard', game)
