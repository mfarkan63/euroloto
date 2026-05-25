"""
XLSX / XLSM data loading.

Reads the standardized tirage.xlsx produced by build_tirage() / update_tirage().
Still supports the old XLSM format for backward compatibility.

Bonus columns are treated as *optional*: rows with NaN bonus values are kept
so that old Loto draws (1976-2008, no numéro chance) are available for
main-ball co-occurrence analysis.  Functions in _analyzer.py and _models.py
that need bonus values filter NaN rows themselves.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from euroloto._config import GAMES


def load(kind: str, data_dir: Path, file: str, sheet: str) -> pd.DataFrame:
    """
    Load and clean draw data for one game from an Excel file.

    Accepts both .xlsx (tirage.xlsx) and .xlsm (legacy) formats.
    Bonus columns with NaN values are preserved (old-format rows).
    """
    config = GAMES[kind]
    path = data_dir / file

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')

    # Keep only recognized columns
    keep = [config['date_col']] + config['main_cols'] + config['bonus_cols']
    df = df[[c for c in keep if c in df.columns]].copy()

    # --- Date ---
    df[config['date_col']] = pd.to_datetime(df[config['date_col']], errors='coerce')
    df = df.dropna(subset=[config['date_col']])
    yr_min, yr_max = config['valid_years']
    df = df[df[config['date_col']].dt.year.between(yr_min, yr_max)].copy()

    # --- Main balls (required, must not be NaN) ---
    for col in config['main_cols']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=config['main_cols'])
    for col in config['main_cols']:
        df[col] = df[col].astype(int)

    m_min, m_max = config['main_range']
    for col in config['main_cols']:
        df = df[df[col].between(m_min, m_max)]

    # --- Bonus columns (optional — NaN rows are kept) ---
    b_min, b_max = config['bonus_range']
    for col in config['bonus_cols']:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Accept NaN (old-format row) OR value within valid range
        valid_mask = df[col].isna() | df[col].between(b_min, b_max)
        df = df[valid_mask]
        # Downcast only non-NaN entries to int (keep NaN as float/pd.NA)
        df[col] = pd.to_numeric(df[col], downcast='integer')

    return df.sort_values(config['date_col']).reset_index(drop=True)
