"""
Probabilistic predictor — Monte Carlo weighted sampling.

AVERTISSEMENT : Les tirages sont indépendants et équiprobables par construction.
Ce modèle identifie des tendances empiriques dans les données historiques.
Il ne prédit pas les tirages futurs. Jouer reste un jeu de hasard.
"""
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd

from euroloto import _analyzer as analyzer


class LotoPredictor:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self._build()

    def _build(self):
        main_cols = self.config['main_cols']
        bonus_cols = self.config['bonus_cols']
        m_min, m_max = self.config['main_range']
        b_min, b_max = self.config['bonus_range']
        n = len(self.df)

        self.main_nums = np.arange(m_min, m_max + 1)
        self.bonus_nums = np.arange(b_min, b_max + 1)

        freq = analyzer.frequency(self.df, main_cols)
        self.main_freq = freq.reindex(self.main_nums, fill_value=0)

        # Bonus: drop rows with NaN bonus (old Loto format before 2008)
        bonus_df = self.df.dropna(subset=bonus_cols)
        freq_b = analyzer.frequency(bonus_df, bonus_cols)
        self.bonus_freq = freq_b.reindex(self.bonus_nums, fill_value=0)

        self.main_last = analyzer.last_seen(self.df, main_cols).reindex(self.main_nums, fill_value=n)
        self.bonus_last = analyzer.last_seen(bonus_df, bonus_cols).reindex(self.bonus_nums, fill_value=n)

        gaps = analyzer.gap_analysis(self.df, main_cols)
        self.main_avg_gap = pd.Series(
            {num: float(np.mean(g)) if g else float(n) for num, g in gaps.items()}
        ).reindex(self.main_nums, fill_value=float(n))

        self._cooc = analyzer.cooccurrence_matrix(self.df, main_cols)

    def generate_combinations(
        self,
        n: int = 10,
        alpha: float = 0.6,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """Generate n candidate combinations via weighted Monte Carlo sampling.

        alpha: 1.0 = pure frequency weight, 0.0 = pure recency weight.
        Returns list of dicts sorted by descending score.
        """
        rng = np.random.default_rng(seed)
        n_main = len(self.config['main_cols'])
        n_bonus = len(self.config['bonus_cols'])

        main_w = self._combined_weights(self.main_freq, self.main_last, alpha)
        bonus_w = self._combined_weights(self.bonus_freq, self.bonus_last, alpha)

        results = []
        for _ in range(n):
            main = sorted(
                rng.choice(self.main_nums, size=n_main, replace=False, p=main_w).tolist()
            )
            bonus = sorted(
                rng.choice(self.bonus_nums, size=n_bonus, replace=False, p=bonus_w).tolist()
            )
            results.append({'main': main, 'bonus': bonus, 'score': self.score(main, bonus)})

        return sorted(results, key=lambda x: x['score'], reverse=True)

    def generate_with_fixed(
        self,
        fixed: List[int],
        comp_freq: pd.DataFrame,
        n: int = 10,
        alpha: float = 0.6,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """Generate n combinations that include all `fixed` numbers.

        Companion frequencies boost the sampling weights for the remaining slots.
        """
        rng = np.random.default_rng(seed)
        b_min, b_max = self.config['bonus_range']
        n_bonus = len(self.config['bonus_cols'])
        bonus_w = self._combined_weights(self.bonus_freq, self.bonus_last, alpha)
        bonus_nums = np.arange(b_min, b_max + 1)

        boosted = self.main_freq.copy()
        for num, row in comp_freq.iterrows():
            if num in boosted.index:
                boosted[num] += row['frequence'] * 2

        n_to_fill = len(self.config['main_cols']) - len(fixed)
        available = [n for n in self.main_nums if n not in fixed]
        avail_w = boosted.reindex(available, fill_value=0).values.astype(float)
        if avail_w.sum() > 0:
            avail_w = avail_w / avail_w.sum()

        candidates, seen = [], set()
        attempts = 0
        while len(candidates) < n and attempts < 500:
            attempts += 1
            complement = sorted(
                rng.choice(available, size=n_to_fill, replace=False, p=avail_w).tolist()
            )
            main = sorted(fixed + complement)
            key = tuple(main)
            if key in seen:
                continue
            seen.add(key)
            bonus = sorted(rng.choice(bonus_nums, size=n_bonus, replace=False, p=bonus_w).tolist())
            candidates.append({'main': main, 'bonus': bonus, 'score': self.score(main, bonus)})

        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def score(self, main: List[int], bonus: List[int]) -> float:
        """Score = 50% mean frequency + 30% mean co-occurrence + 20% bonus frequency."""
        freq_score = float(self.main_freq.reindex(main, fill_value=0).mean())

        cooc_score = 0.0
        for a, b in combinations(main, 2):
            if a in self._cooc.index and b in self._cooc.columns:
                cooc_score += self._cooc.loc[a, b]
        n_pairs = len(main) * (len(main) - 1) / 2
        cooc_score = cooc_score / n_pairs if n_pairs else 0.0

        bonus_score = float(self.bonus_freq.reindex(bonus, fill_value=0).mean())
        return round(freq_score * 0.5 + cooc_score * 0.3 + bonus_score * 0.2, 4)

    def top_numbers(self, n: int = 10) -> pd.DataFrame:
        return pd.concat(
            [self.main_freq.rename('frequence'), self.main_last.rename('retard')], axis=1
        ).sort_values('frequence', ascending=False).head(n)

    def overdue_numbers(self, n: int = 10) -> pd.DataFrame:
        return pd.concat(
            [self.main_freq.rename('frequence'), self.main_last.rename('retard')], axis=1
        ).sort_values('retard', ascending=False).head(n)

    def summary(self) -> pd.DataFrame:
        w = self._combined_weights(self.main_freq, self.main_last, alpha=0.6)
        return pd.DataFrame({
            'frequence': self.main_freq.values,
            'ecart_moyen': self.main_avg_gap.values.round(1),
            'retard': self.main_last.values,
            'poids': (w * 100).round(3),
        }, index=self.main_nums)

    @staticmethod
    def _normalize(s: pd.Series) -> np.ndarray:
        total = s.sum()
        if total == 0:
            return np.ones(len(s)) / len(s)
        return (s.values / total).astype(float)

    def _combined_weights(self, freq: pd.Series, last: pd.Series, alpha: float) -> np.ndarray:
        freq_w = self._normalize(freq)
        recency_w = self._normalize(last)
        combined = alpha * freq_w + (1.0 - alpha) * recency_w
        return combined / combined.sum()
