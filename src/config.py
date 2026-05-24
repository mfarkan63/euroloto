from pathlib import Path

DATA_DIR = Path(__file__).parent.parent

LOTO = {
    'name': 'Loto',
    'file': 'resultat-loto.xlsm',
    'sheet': 'Feuil1',
    'date_col': 'date_de_tirage',
    'main_cols': ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5'],
    'bonus_cols': ['numero_chance'],
    'main_range': (1, 49),
    'bonus_range': (1, 10),
    'valid_years': (2008, 2026),
}

EURO = {
    'name': 'EuroMillions',
    'file': 'resultat-euro.xlsm',
    'sheet': 'Feuil1',
    'date_col': 'date_de_tirage',
    'main_cols': ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5'],
    'bonus_cols': ['etoile_1', 'etoile_2'],
    'main_range': (1, 50),
    'bonus_range': (1, 12),
    'valid_years': (2004, 2026),
    # Structural changes to annotate on temporal charts
    # Key: first year of new regime, Value: label
    'structural_changes': {
        2011: '2 tirages/sem.\n(mar. + ven.)',
    },
}

GAMES = {'loto': LOTO, 'euro': EURO}
