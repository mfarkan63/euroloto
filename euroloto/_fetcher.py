"""
FDJ data fetcher — downloads historical draw ZIPs from the FDJ API.

FDJ migrated from the old CDN (cdn-media.fdj.fr/static-draws/csv)
to a new authenticated-free API endpoint in 2024.  The old CDN ZIPs
are frozen at July 2024; the new API always serves up-to-date data.

New base URL:
  https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/documentations/{uuid}

Each UUID maps to one period ZIP.  The most-recent ID is always first;
update_tirage() fetches only that one for fast incremental updates.
"""
from __future__ import annotations

import io
import zipfile
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# New FDJ API (2024+) — UUID-based endpoints
# ---------------------------------------------------------------------------

FDJ_API = "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/documentations"

# Human-readable labels (same index as the UUID lists below)
LOTO_LABELS: List[str] = [
    "loto Nov 2019 -> present  (5 boules + numero_chance)",
    "loto Feb 2019 -> Nov 2019",
    "loto Mar 2017 -> Feb 2019",
    "loto Oct 2008 -> Mar 2017",
    "loto May 1976 -> Oct 2008  (6 boules, ancien format)",
]

EURO_LABELS: List[str] = [
    "euro Feb 2020 -> present  (etoiles 1-12)",
    "euro Mar 2019 -> Feb 2020",
    "euro Sep 2016 -> Mar 2019",
    "euro Feb 2014 -> Sep 2016",
    "euro May 2011 -> Feb 2014",
    "euro Feb 2004 -> May 2011  (etoiles 1-9)",
]

LOTO_IDS: List[str] = [
    "1a2b3c4d-9876-4562-b3fc-2c963f66afp6",  # Nov 2019 -> present
    "1a2b3c4d-9876-4562-b3fc-2c963f66afo6",  # Feb 2019 -> Nov 2019
    "1a2b3c4d-9876-4562-b3fc-2c963f66afn6",  # Mar 2017 -> Feb 2019
    "1a2b3c4d-9876-4562-b3fc-2c963f66afm6",  # Oct 2008 -> Mar 2017
    "1a2b3c4d-9876-4562-b3fc-2c963f66afl6",  # May 1976 -> Oct 2008 (6-ball old format)
]

EURO_IDS: List[str] = [
    "1a2b3c4d-9876-4562-b3fc-2c963f66afe6",  # Feb 2020 -> present
    "1a2b3c4d-9876-4562-b3fc-2c963f66afd6",  # Mar 2019 -> Feb 2020
    "1a2b3c4d-9876-4562-b3fc-2c963f66afc6",  # Sep 2016 -> Mar 2019
    "1a2b3c4d-9876-4562-b3fc-2c963f66afb6",  # Feb 2014 -> Sep 2016
    "1a2b3c4d-9876-4562-b3fc-2c963f66afa9",  # May 2011 -> Feb 2014
    "1a2b3c4d-9876-4562-b3fc-2c963f66afa8",  # Feb 2004 -> May 2011 (etoiles 1-9)
]

# ---------------------------------------------------------------------------
# Legacy CDN (frozen at July 2024 — kept for reference only)
# ---------------------------------------------------------------------------
# FDJ_CDN = "https://cdn-media.fdj.fr/static-draws/csv"
# LOTO_ZIPS = ["loto/loto_201911.zip", ...]
# EURO_ZIPS  = ["euromillions/euromillions_202002.zip", ...]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def _download_zip(uid: str, timeout: int = 30) -> bytes:
    """Download a ZIP from the FDJ API by UUID and return raw bytes."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests requis : pip install requests")
    url = f"{FDJ_API}/{uid}"
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
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
    Download **all** historical ZIPs for *kind* ('loto' or 'euro') and
    return a sorted, deduplicated DataFrame in canonical schema.

    Takes ~15-30 s (11 network requests total for both games).
    """
    ids = LOTO_IDS if kind == 'loto' else EURO_IDS
    labels = LOTO_LABELS if kind == 'loto' else EURO_LABELS
    normalizer = _normalize_loto if kind == 'loto' else _normalize_euro

    frames: list = []
    for uid, label in zip(ids, labels):
        if verbose:
            print(f"  {label:<58}", end='', flush=True)
        try:
            raw_bytes = _download_zip(uid)
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
    Download **only the most recent** ZIP for *kind*.
    Used by :func:`update_tirage` for fast incremental updates.
    """
    ids = LOTO_IDS if kind == 'loto' else EURO_IDS
    labels = LOTO_LABELS if kind == 'loto' else EURO_LABELS
    normalizer = _normalize_loto if kind == 'loto' else _normalize_euro

    uid = ids[0]        # newest first
    label = labels[0]
    if verbose:
        print(f"  {label:<58}", end='', flush=True)
    raw_bytes = _download_zip(uid)
    df_norm = normalizer(_read_zip_csv(raw_bytes))
    if verbose:
        print(f"  OK  {len(df_norm):>5} tirages")
    return df_norm.sort_values('date_de_tirage').reset_index(drop=True)
