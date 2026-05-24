"""
Statistical analysis functions.
All are pure (no side effects) — take DataFrames and return pandas/numpy objects.
"""
from itertools import combinations
from typing import Dict, List, Tuple

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

    result = pd.DataFrame({
        'frequence': freq,
        'pct': (freq / n_filtered * 100).round(1),
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
