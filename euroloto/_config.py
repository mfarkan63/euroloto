"""
Hardcoded game rules (number ranges, column names, draw structure).
File paths live in config.yaml — not here.

Loto structural notes
---------------------
- May 1976 – Oct 2008  : 6 main balls (1-49) + boule_complémentaire; no numéro chance.
  Fetcher normalizes to 5 main balls (boule_6 discarded); numero_chance = NaN.
- Oct 2008 – present   : 5 main balls (1-49) + numéro chance (1-10).
The bonus_range covers the 2008+ era; old rows with NaN bonus are kept for main-ball analysis.

EuroMillions structural notes
------------------------------
- Feb 2004 – Sep 2016  : étoiles range 1-9
- Sep 2016 – present   : étoiles range 1-12
bonus_range = (1, 12) covers both eras; pre-2016 values 1-9 remain valid.
"""

LOTO: dict = {
    'name': 'Loto',
    'date_col': 'date_de_tirage',
    'main_cols': ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5'],
    'bonus_cols': ['numero_chance'],
    'main_range': (1, 49),
    'bonus_range': (1, 10),
    'valid_years': (1976, 2030),          # 1976+ includes the 6-ball era
    'structural_changes': {
        2008: 'Nouveau Loto\n(5 boules + chance)',
        2019: '3 tirages/sem.',
    },
}

EURO: dict = {
    'name': 'EuroMillions',
    'date_col': 'date_de_tirage',
    'main_cols': ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5'],
    'bonus_cols': ['etoile_1', 'etoile_2'],
    'main_range': (1, 50),
    'bonus_range': (1, 12),               # covers both 1-9 and 1-12 eras
    'valid_years': (2004, 2030),
    'structural_changes': {
        2011: '2 tirages/sem.\n(mar. + ven.)',
        2016: 'Étoiles → 1-12',
    },
}

GAMES: dict[str, dict] = {'loto': LOTO, 'euro': EURO}
