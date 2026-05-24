"""
Probabilistic predictor.

AVERTISSEMENT : Les tirages du loto sont indépendants et équiprobables par
construction. Ce modèle identifie des tendances empiriques dans les données
historiques — il ne prédit pas les tirages futurs. Jouer reste un jeu de hasard.
"""
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd

from . import analyzer


class LotoPredictor:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self._build()

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self):
        main_cols = self.config['main_cols']
        bonus_cols = self.config['bonus_cols']
        m_min, m_max = self.config['main_range']
        b_min, b_max = self.config['bonus_range']
        n = len(self.df)

        self.main_nums = np.arange(m_min, m_max + 1)
        self.bonus_nums = np.arange(b_min, b_max + 1)

        # Frequency
        freq = analyzer.frequency(self.df, main_cols)
        self.main_freq = freq.reindex(self.main_nums, fill_value=0)

        freq_b = analyzer.frequency(self.df, bonus_cols)
        self.bonus_freq = freq_b.reindex(self.bonus_nums, fill_value=0)

        # Recency (draws since last appearance — higher = more overdue)
        self.main_last = analyzer.last_seen(self.df, main_cols).reindex(self.main_nums, fill_value=n)
        self.bonus_last = analyzer.last_seen(self.df, bonus_cols).reindex(self.bonus_nums, fill_value=n)

        # Average gap
        gaps = analyzer.gap_analysis(self.df, main_cols)
        self.main_avg_gap = pd.Series(
            {num: float(np.mean(g)) if g else float(n) for num, g in gaps.items()}
        ).reindex(self.main_nums, fill_value=float(n))

        # Co-occurrence matrix (precomputed once)
        self._cooc = analyzer.cooccurrence_matrix(self.df, main_cols)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_combinations(
        self,
        n: int = 10,
        alpha: float = 0.6,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """
        Generate n candidate combinations using weighted Monte Carlo sampling.

        alpha: weight for frequency (0 = pure recency / retard, 1 = pure frequency).
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
            results.append({
                'main': main,
                'bonus': bonus,
                'score': self.score(main, bonus),
            })

        return sorted(results, key=lambda x: x['score'], reverse=True)

    def score(self, main: List[int], bonus: List[int]) -> float:
        """
        Score a combination based on:
        - Mean historical frequency of main numbers
        - Co-occurrence bonus for pairs that appeared together often
        - Mean historical frequency of bonus numbers
        """
        freq_score = float(self.main_freq.reindex(main, fill_value=0).mean())

        cooc_score = 0.0
        for a, b in combinations(main, 2):
            if a in self._cooc.index and b in self._cooc.columns:
                cooc_score += self._cooc.loc[a, b]
        n_pairs = len(main) * (len(main) - 1) / 2
        cooc_score = cooc_score / n_pairs if n_pairs else 0.0

        bonus_score = float(self.bonus_freq.reindex(bonus, fill_value=0).mean())

        return round((freq_score * 0.5 + cooc_score * 0.3 + bonus_score * 0.2), 4)

    def top_numbers(self, n: int = 10) -> pd.DataFrame:
        """Top n most frequent main numbers with recency info."""
        ls = self.main_last.rename('retard')
        freq = self.main_freq.rename('frequence')
        df = pd.concat([freq, ls], axis=1).sort_values('frequence', ascending=False).head(n)
        return df

    def overdue_numbers(self, n: int = 10) -> pd.DataFrame:
        """Top n most overdue main numbers (not appeared for longest time)."""
        ls = self.main_last.rename('retard')
        freq = self.main_freq.rename('frequence')
        df = pd.concat([freq, ls], axis=1).sort_values('retard', ascending=False).head(n)
        return df

    def summary(self) -> pd.DataFrame:
        """Full table: number | frequency | avg_gap | retard | combined_weight."""
        w = self._combined_weights(self.main_freq, self.main_last, alpha=0.6)
        return pd.DataFrame({
            'frequence': self.main_freq.values,
            'ecart_moyen': self.main_avg_gap.values.round(1),
            'retard': self.main_last.values,
            'poids': (w * 100).round(3),
        }, index=self.main_nums)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(s: pd.Series) -> np.ndarray:
        total = s.sum()
        if total == 0:
            return np.ones(len(s)) / len(s)
        return (s.values / total).astype(float)

    def _combined_weights(
        self, freq: pd.Series, last: pd.Series, alpha: float
    ) -> np.ndarray:
        freq_w = self._normalize(freq)
        recency_w = self._normalize(last)
        combined = alpha * freq_w + (1.0 - alpha) * recency_w
        return combined / combined.sum()
