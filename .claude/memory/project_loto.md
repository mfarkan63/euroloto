---
name: project-loto
description: "Projet d'analyse statistique du Loto et EuroMillions en Python (Anaconda)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 145ec33b-9c47-445f-b63a-28adb764b8da
---

Projet Python d'analyse probabiliste/statistique du Loto et EuroMillions.

**Why:** L'utilisateur veut explorer les corrélations et tendances empiriques dans les tirages historiques officiels.

**How to apply:** Proposer des extensions ou analyses supplémentaires dans ce contexte.

## Localisation
- Répertoire : `C:\Users\farhane\Documents\Claude\Loto`
- Python : `C:\anaconda3\python.exe` (Python 3.13.9, Anaconda)
- Données : `resultat-loto.xlsm` (2796 tirages, 2008–2025) et `resultat-euro.xlsm` (1945 tirages, 2004–2025)

## Structure du projet
```
Loto/
├── euroloto/              # Package Python — API publique plate (usage notebook)
│   ├── __init__.py        # API publique : init, frequency, prediction, plot_*, etc.
│   ├── _state.py          # État module (DataFrames chargés, predictors en cache)
│   ├── _config.py         # Règles des jeux (ranges, colonnes) — pas de chemins
│   ├── _loader.py         # Chargement XLSM depuis config.yaml
│   ├── _analyzer.py       # Analyses statistiques pures
│   ├── _models.py         # LotoPredictor (Monte Carlo pondéré)
│   └── _plots.py          # Graphiques → retourne Figure (pas de fichiers)
├── src/                   # Module CLI (utilisé par main.py uniquement)
│   ├── config.py
│   ├── data_loader.py
│   ├── analyzer.py
│   ├── models.py
│   └── visualizer.py      # Graphiques → sauvegarde dans charts/
├── notebooks/
│   ├── demo.ipynb         # Notebook court pour euroloto (usage recommandé)
│   └── loto_analysis.ipynb
├── config.yaml            # Chemins vers les fichiers XLSM (éditable)
├── main.py                # CLI : --analyze / --predict N / --visualize / --alpha / --seed
└── requirements.txt
```

## Package euroloto — API notebook
```python
import euroloto
euroloto.init()                              # lit config.yaml depuis le dossier courant
euroloto.info()
euroloto.draws('loto')                       # DataFrame brut

# Analyse
euroloto.frequency('loto')
euroloto.hot_cold('loto', n_recent=50)
euroloto.cooccurrence('loto', top_n=20, min_cooc=30)
euroloto.top_pairs('loto', n=10)
euroloto.overdue('loto')
euroloto.sum_stats('loto')
euroloto.even_odd('loto')
euroloto.chi2_test('loto')
euroloto.ks_test('loto')
euroloto.companions([26,27], kind='loto')    # kind='all' croise les deux jeux

# Prédiction
euroloto.prediction(kind='loto', n=10, alpha=0.6, seed=42)
euroloto.prediction([26,27], kind='loto')    # numéros fixés
euroloto.prediction([26,27], kind='all')     # → {'loto': [...], 'euro': [...]}

# Graphiques (retourne Figure, pas de fichiers sauvegardés)
euroloto.plot_frequency('loto')              # label='bonus' pour les bonus
euroloto.plot_hot_cold('loto')
euroloto.plot_cooccurrence('loto', min_cooc=30)
euroloto.plot_gaps('loto')
euroloto.plot_sum('loto')
euroloto.plot_temporal('euro')
euroloto.plot_overdue('loto')
euroloto.plot_companions([26,27], kind='all')
```

## Commandes CLI
```bash
$env:PYTHONIOENCODING="utf-8"
python main.py loto --analyze
python main.py euro --predict 10 --alpha 0.6 --seed 42
python main.py loto --visualize --min-cooc 30    # heatmap avec seuil co-occurrences
python main.py loto --companions 26 27           # compléments (jeu unique)
python main.py euro --companions 47 48
python main.py all  --companions 26 27           # mode croisé Loto + EuroMillions
```

## Mode 'all'
- `all` est un 3ème jeu réservé à `--companions` uniquement (analyze/predict/visualize refusés)
- Charge les deux bases séparément (sans filtrer le 50), somme les fréquences par source
- Tableau = total | pct_% | loto | euro (les totaux s'additionnent exactement)
- Graphique : barplot empilé bleu (Loto) + rouge (EuroMillions) → charts/all_companions_all_X_Y.png

## Fonctionnalité companions
- Les feuilles Feuil2–Feuil10 du XLSM Loto = tirages filtrés par une paire fixe (26,27 / 19,46 / 31,34 / 13,27 / 3,33 / 26,46 / 47,48 / numéro 20)
- L'utilisateur cherche : "si je joue ces 2 numéros, quels 3 meilleurs compléments ?"
- `analyzer.companions(df, cols, [a, b])` reproduit cette logique automatiquement

## Points techniques
- EuroMillions : passage 1→2 tirages/semaine en mai 2011 (86 tirages en 2011, ~104 ensuite)
  → `temporal_trend` normalise par draws/année pour corriger ce saut (fréq. en % par tirage)
  → `structural_changes` dans config.py = annotation verticale sur le graphique
- EuroMillions 2024 : 173 tirages (anomalie data : doublons + jours aberrants), filtré par normalisation
- pandas 3.x sur Python 3.13 : utiliser `'Y'` (pas `'YE'`) pour `dt.to_period()`
- Sortie CLI : nécessite `$env:PYTHONIOENCODING="utf-8"` sous PowerShell Windows
- heatmap co-occurrence : format `.0f` obligatoire (float après masquage NaN)
