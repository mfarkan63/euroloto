"""
FDJ data fetcher - downloads historical draw CSVs from the official FDJ CDN.

All ZIP file URLs are versioned by start-date (e.g. loto_201911.zip = Nov 2019 -> today).
The most-recent file always contains the latest draws; update_tirage() fetches only it.
"""
from __future__ import annotations

import io
import zipfile
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# CDN base and file lists (newest -> oldest)
# ---------------------------------------------------------------------------

FDJ_CDN = "https://cdn-media.fdj.fr/static-draws/csv"

LOTO_ZIPS: List[str] = [
    "loto/loto_201911.zip",   # Nov 2019 -> present  (5 balls + numero_chance)
    "loto/loto_201902.zip",   # Feb 2019 -> Nov 2019
    "loto/loto_201703.zip",   # Mar 2017 -> Feb 2019
    "loto/loto_200810.zip",   # Oct 2008 -> Mar 2017
    "loto/loto_197605.zip",   # May 1976 -> Oct 2008  (6 balls, old format)
]

EURO_ZIPS: List[str] = [
    "euromillions/euromillions_202002.zip",  # Feb 2020 -> present  (etoiles 1-12)
    "euromillions/euromillions_201902.zip",  # Feb 2019 -> Feb 2020
    "euromillions/euromillions_201609.zip",  # Sep 2016 -> Feb 2019
    "euromillions/euromillions_201402.zip",  # Feb 2014 -> Sep 2016
    "euromillions/euromillions_201105.zip",  # May 2011 -> Feb 2014
    "euromillions/euromillions_200402.zip",  # Feb 2004 -> May 2011  (etoiles 1-9)
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_zip(url: str, timeout: int = 30) -> bytes:
    """Download a ZIP from *url* and return raw bytes."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests requis : pip install requests")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _read_zip_csv(raw: bytes) -> pd.DataFrame:
    """Extract and parse the first CSV from a FDJ ZIP bytes object."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            raw_csv = f.read()
    from io import StringIO
    text = raw_csv.decode('latin-1')
    # index_col=False: prevent pandas from auto-detecting a row-index when data
    # has one more field than the header (some intermediate-era FDJ files).
    return pd.read_csv(StringIO(text), sep=';', dtype=str, index_col=False)


def _parse_date(series: pd.Series) -> pd.Series:
    """
    Parse FDJ date columns - handles both formats:
    - DD/MM/YYYY  (2008+ files)
    - YYYYMMDD    (pre-2008 files)
    """
    s = series.str.strip()
    result = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(s[mask], format='%Y%m%d', errors='coerce')
    return result


def _normalize_loto(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map a raw Loto CSV to the 7-column canonical schema:
    date_de_tirage, boule_1-5, numero_chance.

    Old format (1976-2008): 6 main balls + boule_complementaire, no numero_chance.
      - Filter to primary draws only (1er_ou_2eme_tirage == 1).
      - Keep boule_1..5 (ignore boule_6).
      - numero_chance -> pd.NA (retained but marked as missing).

    New format (2008+): 5 main balls + numero_chance.
    """
    # Old format guard: keep only primary draws
    if '1er_ou_2eme_tirage' in raw.columns:
        raw = raw[raw['1er_ou_2eme_tirage'].astype(str).str.strip() == '1'].copy()

    out = pd.DataFrame()
    out['date_de_tirage'] = _parse_date(raw['date_de_tirage'])
    for c in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        out[c] = pd.to_numeric(raw[c], errors='coerce')

    if 'numero_chance' in raw.columns:
        out['numero_chance'] = pd.to_numeric(raw['numero_chance'], errors='coerce')
    else:
        out['numero_chance'] = pd.NA   # old format: no bonus mapping

    return out.dropna(
        subset=['date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
    )


def _normalize_euro(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map a raw EuroMillions CSV to the 8-column canonical schema:
    date_de_tirage, boule_1-5, etoile_1, etoile_2.

    The etoile range changed from 1-9 (pre-Sep 2016) to 1-12 (post-Sep 2016).
    Both eras are stored as-is; the config bonus_range covers the full range (1-12).
    Rows with etoiles outside (1-12) are dropped by the loader.
    """
    out = pd.DataFrame()
    out['date_de_tirage'] = _parse_date(raw['date_de_tirage'])
    for c in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']:
        out[c] = pd.to_numeric(raw[c], errors='coerce')
    return out.dropna(
        subset=['date_de_tirage', 'boule_1', 'boule_2',
                'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all(kind: str, verbose: bool = True) -> pd.DataFrame:
    """
    Download **all** historical ZIP files for *kind* ('loto' or 'euro') and
    return a sorted, deduplicated DataFrame in canonical schema.

    Takes ~15-30 s (11 network requests total for both games).
    """
    zips = LOTO_ZIPS if kind == 'loto' else EURO_ZIPS
    normalizer = _normalize_loto if kind == 'loto' else _normalize_euro

    frames: list = []
    for path in zips:
        url = f"{FDJ_CDN}/{path}"
        if verbose:
            print(f"  {path:<58}", end='', flush=True)
        try:
            raw_bytes = _download_zip(url)
            df_norm = normalizer(_read_zip_csv(raw_bytes))
            frames.append(df_norm)
            if verbose:
                print(f"  OK  {len(df_norm):>5} tirages")
        except Exception as exc:
            if verbose:
                print(f"  ERR {exc}")

    if not frames:
        raise RuntimeError(f"Aucun fichier telecharge pour '{kind}'.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values('date_de_tirage')
        .drop_duplicates(subset=['date_de_tirage'])
        .reset_index(drop=True)
    )
    return combined


def fetch_latest(kind: str, verbose: bool = True) -> pd.DataFrame:
    """
    Download **only the most recent** ZIP file for *kind*.
    Used by :func:`update_tirage` for fast incremental updates.
    """
    zips = LOTO_ZIPS if kind == 'loto' else EURO_ZIPS
    normalizer = _normalize_loto if kind == 'loto' else _normalize_euro

    path = zips[0]   # newest file first
    url = f"{FDJ_CDN}/{path}"
    if verbose:
        print(f"  {path:<58}", end='', flush=True)
    raw_bytes = _download_zip(url)
    df_norm = normalizer(_read_zip_csv(raw_bytes))
    if verbose:
        print(f"  OK  {len(df_norm):>5} tirages")
    return df_norm.sort_values('date_de_tirage').reset_index(drop=True)
