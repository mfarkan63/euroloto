import warnings
import pandas as pd
from .config import DATA_DIR, LOTO, EURO

MAIN_COLS = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']

ALL_CONFIG = {
    'name': 'Loto + EuroMillions',
    'main_cols': MAIN_COLS,
    'bonus_cols': [],
    'main_range': (1, 49),   # intersection des deux plages
    'bonus_range': (1, 1),   # inutilisé
    'date_col': 'date_de_tirage',
    'valid_years': (2004, 2026),
    'structural_changes': {},
}


def load(config: dict) -> pd.DataFrame:
    """Load and clean draw data from XLSM file (Feuil1 only)."""
    path = DATA_DIR / config['file']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_excel(path, sheet_name=config['sheet'], engine='openpyxl')

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


def load_all() -> pd.DataFrame:
    """
    Combine Loto + EuroMillions on their 5 main balls only.
    Numbers 50 (EuroMillions only) are excluded to keep a common 1–49 range.
    A 'source' column indicates the origin of each draw.
    """
    def _load_main(cfg: dict, source: str) -> pd.DataFrame:
        df = load(cfg)
        out = df[['date_de_tirage'] + cfg['main_cols']].copy()
        out.columns = ['date_de_tirage'] + MAIN_COLS
        out['source'] = source
        # Filter to common range 1-49
        for col in MAIN_COLS:
            out = out[out[col].between(1, 49)]
        return out

    df_loto = _load_main(LOTO, 'loto')
    df_euro = _load_main(EURO, 'euro')
    combined = pd.concat([df_loto, df_euro], ignore_index=True)
    return combined.sort_values('date_de_tirage').reset_index(drop=True)
