"""XLSM data loading — reads file paths from module state (set by init())."""
from __future__ import annotations
import warnings
from pathlib import Path

import pandas as pd

from euroloto._config import GAMES


def load(kind: str, data_dir: Path, file: str, sheet: str) -> pd.DataFrame:
    """Load and clean draw data for one game."""
    config = GAMES[kind]
    path = data_dir / file

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')

    keep = [config['date_col']] + config['main_cols'] + config['bonus_cols']
    df = df[[c for c in keep if c in df.columns]].copy()

    df[config['date_col']] = pd.to_datetime(df[config['date_col']], errors='coerce')
    yr_min, yr_max = config['valid_years']
    df = df[df[config['date_col']].dt.year.between(yr_min, yr_max)].copy()

    num_cols = config['main_cols'] + config['bonus_cols']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols)
    for col in num_cols:
        df[col] = df[col].astype(int)

    m_min, m_max = config['main_range']
    for col in config['main_cols']:
        df = df[df[col].between(m_min, m_max)]

    b_min, b_max = config['bonus_range']
    for col in config['bonus_cols']:
        df = df[df[col].between(b_min, b_max)]

    return df.sort_values(config['date_col']).reset_index(drop=True)
