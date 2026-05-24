"""
Hardcoded game rules (number ranges, column names, draw structure).
File paths live in config.yaml — not here.
"""

LOTO: dict = {
    'name': 'Loto',
    'date_col': 'date_de_tirage',
    'main_cols': ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5'],
    'bonus_cols': ['numero_chance'],
    'main_range': (1, 49),
    'bonus_range': (1, 10),
    'valid_years': (2008, 2026),
    'structural_changes': {},
}

EURO: dict = {
    'name': 'EuroMillions',
    'date_col': 'date_de_tirage',
    'main_cols': ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5'],
    'bonus_cols': ['etoile_1', 'etoile_2'],
    'main_range': (1, 50),
    'bonus_range': (1, 12),
    'valid_years': (2004, 2026),
    'structural_changes': {
        2011: '2 tirages/sem.\n(mar. + ven.)',
    },
}

GAMES: dict[str, dict] = {'loto': LOTO, 'euro': EURO}
