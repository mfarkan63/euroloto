"""Module-level state — holds loaded DataFrames and config after init()."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

# Populated by init()
data_dir: Optional[Path] = None
dfs: dict[str, pd.DataFrame] = {}        # 'loto' | 'euro'
configs: dict[str, dict] = {}            # 'loto' | 'euro'
_predictors: dict[str, object] = {}      # cached LotoPredictor instances


def require(kind: str) -> tuple[pd.DataFrame, dict]:
    """Return (df, config) for kind, raising if init() was not called."""
    if kind not in dfs:
        raise RuntimeError(
            f"Données '{kind}' non chargées. Appelez euroloto.init() d'abord."
        )
    return dfs[kind], configs[kind]


def predictor(kind: str):
    """Return cached LotoPredictor for kind, building it on first access."""
    if kind not in _predictors:
        from euroloto._models import LotoPredictor
        df, cfg = require(kind)
        _predictors[kind] = LotoPredictor(df, cfg)
    return _predictors[kind]


def invalidate(kind: str | None = None):
    """Clear cached predictors (call after data reload)."""
    if kind is None:
        _predictors.clear()
    else:
        _predictors.pop(kind, None)
