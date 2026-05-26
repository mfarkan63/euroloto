"""
Statistical analysis functions.
All are pure (no side effects) — take DataFrames and return pandas/numpy objects.
"""
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def frequency(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Count total appearances of each number across all draws."""
    return pd.Series(df[cols].values.flatten()).value_counts().sort_index()


def hot_cold(df: pd.DataFrame, cols: List[str], n_recent: int = 50) -> pd.DataFrame:
    """Compare frequency in last n_recent draws vs overall average."""
    total_freq = frequency(df, cols)
    recent_freq = frequency(df.tail(n_recent), cols)

    result = pd.DataFrame({
        'total_count': total_freq,
        'recent_count': recent_freq.reindex(total_freq.index, fill_value=0),
        'total_pct': total_freq / len(df) * 100,
        'recent_pct': recent_freq.reindex(total_freq.index, fill_value=0) / n_recent * 100,
    })
    result['delta'] = result['recent_pct'] - result['total_pct']
    result['statut'] = result['delta'].apply(lambda x: 'chaud' if x > 0 else 'froid')
    return result.sort_values('delta', ascending=False)


def cooccurrence_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Pairwise co-occurrence frequency as a symmetric DataFrame."""
    values = df[cols].values.flatten().astype(int)
    n_min, n_max = int(values.min()), int(values.max())
    index = list(range(n_min, n_max + 1))
    idx_map = {v: i for i, v in enumerate(index)}
    matrix = np.zeros((len(index), len(index)), dtype=np.int32)

    for row in df[cols].values:
        nums = sorted(int(x) for x in row)
        for a, b in combinations(nums, 2):
            i, j = idx_map[a], idx_map[b]
            matrix[i, j] += 1
            matrix[j, i] += 1

    return pd.DataFrame(matrix, index=index, columns=index)


def gap_analysis(df: pd.DataFrame, cols: List[str]) -> Dict[int, List[int]]:
    """For each number, list of draw-counts between consecutive appearances."""
    all_nums = sorted(set(int(x) for x in df[cols].values.flatten()))
    gaps: Dict[int, List[int]] = {}

    for num in all_nums:
        mask = (df[cols] == num).any(axis=1).values
        indices = np.where(mask)[0]
        gaps[num] = list(np.diff(indices).astype(int)) if len(indices) > 1 else []

    return gaps


def last_seen(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Number of draws since each number last appeared (0 = appeared in last draw)."""
    n_draws = len(df)
    all_nums = sorted(set(int(x) for x in df[cols].values.flatten()))
    last: Dict[int, int] = {n: -1 for n in all_nums}

    for i, row in enumerate(df[cols].values):
        for num in row.astype(int):
            last[num] = i

    return pd.Series({n: n_draws - 1 - last[n] for n in all_nums})


def chi2_uniformity_test(freq: pd.Series, n_range: Tuple[int, int]) -> dict:
    """Chi-squared test for uniform distribution over n_range."""
    n_min, n_max = n_range
    full = freq.reindex(range(n_min, n_max + 1), fill_value=0)
    expected = np.full(len(full), full.sum() / len(full))
    stat, pvalue = stats.chisquare(full.values, f_exp=expected)
    return {
        'statistic': round(stat, 4),
        'pvalue': round(pvalue, 6),
        'is_uniform': pvalue > 0.05,
        'ddl': len(full) - 1,
    }


def ks_uniformity_test(df: pd.DataFrame, cols: List[str], n_range: Tuple[int, int]) -> dict:
    """Kolmogorov-Smirnov test against uniform distribution."""
    values = df[cols].values.flatten().astype(float)
    n_min, n_max = n_range
    normalized = (values - n_min) / (n_max - n_min)
    stat, pvalue = stats.kstest(normalized, 'uniform')
    return {
        'statistic': round(stat, 6),
        'pvalue': round(pvalue, 6),
        'is_uniform': pvalue > 0.05,
    }


def sum_statistics(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Descriptive statistics on the sum of drawn numbers per draw."""
    sums = df[cols].sum(axis=1)
    return pd.Series({
        'mean': round(sums.mean(), 2),
        'std': round(sums.std(), 2),
        'min': int(sums.min()),
        'q25': sums.quantile(0.25),
        'median': sums.median(),
        'q75': sums.quantile(0.75),
        'max': int(sums.max()),
        'skewness': round(float(stats.skew(sums)), 4),
        'kurtosis': round(float(stats.kurtosis(sums)), 4),
    }, name='sum_stats')


def even_odd_distribution(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Distribution of even/odd counts per draw."""
    n = len(cols)
    even_counts = df[cols].apply(lambda row: int((row % 2 == 0).sum()), axis=1)
    counts = even_counts.value_counts().sort_index()
    result = pd.DataFrame({
        'n_pair': counts.index,
        'n_impair': n - counts.index,
        'nb_tirages': counts.values,
        'pct': (counts.values / counts.sum() * 100).round(2),
    })
    return result.reset_index(drop=True)


def high_low_distribution(df: pd.DataFrame, cols: List[str], n_range: Tuple[int, int]) -> pd.DataFrame:
    """Distribution of high/low number counts per draw (split at midpoint)."""
    n_min, n_max = n_range
    mid = (n_min + n_max) / 2
    high_counts = df[cols].apply(lambda row: int((row > mid).sum()), axis=1)
    n = len(cols)
    counts = high_counts.value_counts().sort_index()
    result = pd.DataFrame({
        'n_haut': counts.index,
        'n_bas': n - counts.index,
        'nb_tirages': counts.values,
        'pct': (counts.values / counts.sum() * 100).round(2),
    })
    return result.reset_index(drop=True)


def temporal_frequency(
    df: pd.DataFrame, cols: List[str], date_col: str, freq: str = 'Y'
) -> pd.DataFrame:
    """Frequency of each number grouped by time period (default: year)."""
    df2 = df.copy()
    df2['_period'] = df2[date_col].dt.to_period(freq).astype(str)
    rows = []
    for period, grp in df2.groupby('_period'):
        f = frequency(grp, cols)
        rows.append({'period': period, **f.to_dict()})
    return pd.DataFrame(rows).set_index('period').fillna(0).astype(int)


def companions(
    df: pd.DataFrame,
    cols: List[str],
    fixed: List[int],
    n_top: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find all draws containing every number in `fixed`, then compute
    the frequency of the companion numbers (the remaining drawn numbers).

    Returns:
        filtered_df    — subset of df where all fixed numbers appear
        companion_freq — DataFrame(frequence, pct, retard) sorted by frequency
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for num in fixed:
        mask &= (df[cols] == num).any(axis=1)

    filtered = df[mask].copy()
    n_filtered = len(filtered)

    if n_filtered == 0:
        return filtered, pd.DataFrame(columns=['frequence', 'pct'])

    flat = filtered[cols].values.flatten().astype(int)
    comp_nums = [n for n in flat if n not in fixed]

    freq = pd.Series(comp_nums).value_counts()
    ls = last_seen(filtered, cols)

    # Lift = conditional probability / marginal probability
    # Values > 1 mean the number appears more often with `fixed` than by chance
    freq_total = frequency(df, cols)
    n_total = len(df)
    marginal = freq_total.reindex(freq.index, fill_value=1) / n_total
    conditional = freq / n_filtered
    lift = (conditional / marginal).round(2)

    result = pd.DataFrame({
        'frequence': freq,
        'pct': (freq / n_filtered * 100).round(1),
        'lift': lift,
        'retard': ls.reindex(freq.index, fill_value=n_filtered),
    }).head(n_top)

    return filtered, result


def companions_with_seed(
    df: pd.DataFrame,
    cols: List[str],
    fixed: List[int],
    min_one_of: List[int],
    n_top: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like companions() but requires at least one number from min_one_of
    to be present alongside fixed (the 'seed filter').

    min_one_of: companions from the most recent seed draw — draws not sharing
                any of them are excluded.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for num in fixed:
        mask &= (df[cols] == num).any(axis=1)

    remaining = [n for n in min_one_of if n not in fixed]
    if remaining:
        seed_mask = pd.Series([False] * len(df), index=df.index)
        for num in remaining:
            seed_mask |= (df[cols] == num).any(axis=1)
        mask &= seed_mask

    filtered = df[mask].copy()
    n_filtered = len(filtered)

    if n_filtered == 0:
        return filtered, pd.DataFrame(columns=['frequence', 'pct'])

    flat = filtered[cols].values.flatten().astype(int)
    comp_nums = [n for n in flat if n not in fixed]
    freq = pd.Series(comp_nums).value_counts()
    ls = last_seen(filtered, cols)

    result = pd.DataFrame({
        'frequence': freq,
        'pct': (freq / n_filtered * 100).round(1),
        'retard': ls.reindex(freq.index, fill_value=n_filtered),
    }).head(n_top)

    return filtered, result


def top_pairs(df: pd.DataFrame, cols: List[str], n: int = 20) -> pd.DataFrame:
    """Top n most co-occurring pairs."""
    matrix = cooccurrence_matrix(df, cols)
    pairs = []
    nums = matrix.index.tolist()
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            pairs.append({'num_a': a, 'num_b': b, 'co_occurrences': matrix.loc[a, b]})
    return (
        pd.DataFrame(pairs)
        .sort_values('co_occurrences', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_combinations(
    df: pd.DataFrame,
    cols: List[str],
    fixed: List[int],
    n_top: int = 10,
    n_candidates: int = 30,
) -> Tuple[pd.DataFrame, float]:
    """
    Enumerate the most promising combinations starting from `fixed` numbers.

    Algorithm
    ---------
    1. Filter draws that contain every number in `fixed`.
    2. Rank companion numbers by conditional frequency.
    3. Take the top `n_candidates` companions and enumerate all
       C(n_candidates, n_to_fill) complete combinations.
    4. Score each combination by **mean pairwise conditional lift**:
       ``score = mean over all pairs (a,b) of P(a∩b|fixed) / (P(a|fixed)×P(b|fixed))``
    5. Select the top `n_top` combinations using **greedy Jaccard diversity**
       so that the returned set has maximum inter-combination variety
       (avoids returning near-duplicate combinations).
    6. Normalize scores to probability percentages (sum = 100 % over all
       enumerated candidates).

    Parameters
    ----------
    df            : full draw DataFrame
    cols          : main ball columns
    fixed         : the 2 (or more) numbers already chosen
    n_top         : how many combinations to return
    n_candidates  : how many top companion numbers to consider (C(n_candidates, n_to_fill)
                    combinations are enumerated; 30 gives 4 060 triplets for 5-ball games)

    Returns
    -------
    combos_df : DataFrame with columns ``main, score, prob_pct, diversity_rank``
    reference : mean score over **all** enumerated candidates — the baseline to beat
    """
    from itertools import combinations as _comb

    fixed = sorted(fixed)
    n_to_fill = len(cols) - len(fixed)
    if n_to_fill <= 0:
        return pd.DataFrame(), 0.0

    # ------------------------------------------------------------------
    # 1. Filtered draws
    # ------------------------------------------------------------------
    mask = pd.Series([True] * len(df), index=df.index)
    for num in fixed:
        mask &= (df[cols] == num).any(axis=1)
    filtered = df[mask]
    n_f = len(filtered)

    if n_f < 5:
        return pd.DataFrame(), 0.0

    # ------------------------------------------------------------------
    # 2. Companion frequencies inside filtered draws
    # ------------------------------------------------------------------
    flat = filtered[cols].values.flatten().astype(int)
    comp_series = pd.Series([n for n in flat if n not in fixed]).value_counts()

    candidates = [c for c in comp_series.nlargest(n_candidates).index if c not in fixed]
    if len(candidates) < n_to_fill:
        return pd.DataFrame(), 0.0

    # Conditional probabilities P(x | fixed)
    p_cond: Dict[int, float] = {n: comp_series.get(n, 0) / n_f for n in candidates}
    for n in fixed:
        p_cond[n] = 1.0   # fixed numbers always present

    # ------------------------------------------------------------------
    # 3. Co-occurrence matrix inside filtered draws
    # ------------------------------------------------------------------
    cooc_f = cooccurrence_matrix(filtered, cols)

    def _mean_pairwise_lift(complement: tuple) -> float:
        """Mean conditional lift over all C(5,2) pairs."""
        full = fixed + list(complement)
        lifts: List[float] = []
        for a, b in _comb(full, 2):
            pa = p_cond.get(a, 0.0)
            pb = p_cond.get(b, 0.0)
            denom = pa * pb
            if denom <= 0 or a not in cooc_f.index or b not in cooc_f.columns:
                continue
            p_ab = cooc_f.loc[a, b] / n_f
            lifts.append(p_ab / denom)
        return float(np.mean(lifts)) if lifts else 0.0

    # ------------------------------------------------------------------
    # 4. Score all C(n_candidates, n_to_fill) combinations
    # ------------------------------------------------------------------
    all_combos = list(_comb(candidates, n_to_fill))
    rows = [
        {
            'main': sorted(fixed + list(combo)),
            '_key': tuple(sorted(fixed + list(combo))),
            'score': round(_mean_pairwise_lift(combo), 6),
        }
        for combo in all_combos
    ]
    combos_df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)

    # Reference = mean over ALL enumerated candidates
    reference = float(combos_df['score'].mean())

    # Normalize to probability percentages (proportional to score)
    total_score = combos_df['score'].sum()
    combos_df['prob_pct'] = (
        (combos_df['score'] / total_score * 100).round(6) if total_score > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 5. Greedy Jaccard diversity selection
    # ------------------------------------------------------------------
    def _jaccard_dist(set_a: set, set_b: set) -> float:
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return 1.0 - inter / union if union > 0 else 0.0

    selected_idx: List[int] = [0]
    selected_sets: List[set] = [set(combos_df.loc[0, 'main'])]

    for _ in range(n_top - 1):
        best_i: Optional[int] = None
        best_min_dist = -1.0
        for i in combos_df.index:
            if i in selected_idx:
                continue
            s = set(combos_df.loc[i, 'main'])
            min_dist = min(_jaccard_dist(s, ss) for ss in selected_sets)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_i = i
        if best_i is None:
            break
        selected_idx.append(best_i)
        selected_sets.append(set(combos_df.loc[best_i, 'main']))

    result = combos_df.loc[selected_idx, ['main', 'score', 'prob_pct']].copy()
    result.insert(0, 'rank', range(1, len(result) + 1))

    # Diversity metric: min Jaccard distance to nearest neighbour in the selected set
    # 0 = identical complement (clone), 1 = no common number outside fixed
    s_list = [set(r) for r in result['main']]
    min_jacs = []
    for i, s in enumerate(s_list):
        others = [ss for j, ss in enumerate(s_list) if j != i]
        min_jacs.append(round(min((_jaccard_dist(s, o) for o in others), default=1.0), 3))
    result['diversite'] = min_jacs

    result = result.reset_index(drop=True)
    return result, reference
