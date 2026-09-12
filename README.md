# SDSS Object Classification with XGBoost

Classifying celestial objects — **stars, galaxies, and quasars (QSOs)** — from Sloan Digital Sky Survey (SDSS) photometric and spectroscopic data using an XGBoost classifier.

## Dataset

The dataset (`digitalskysurvey.csv`) contains **10,000 observations** from the Sloan Digital Sky Survey, with 18 original columns including:

- **Photometric magnitudes**: `u`, `g`, `r`, `i`, `z` (five-band brightness measurements)
- **Sky coordinates**: `ra` (right ascension), `dec` (declination)
- **Spectroscopic info**: `redshift`, `plate`, `mjd`, `fiberid`
- **Target**: `class` — `STAR`, `GALAXY`, or `QSO`

Instrument/pipeline metadata (`objid`, `specobjid`, `run`, `rerun`, `camcol`, `field`) was dropped early, since these identify *how* the object was observed rather than describing the object itself.

## Approach

1. **EDA** — null/duplicate checks, class distribution, correlation heatmap, pairplots, and distribution plots (violin/strip/histogram) by class.
2. **Feature selection** — a correlation-based function flags features with pairwise correlation above a threshold (0.80); highly redundant bands (`g`, `i`, `z`) and `mjd` were dropped.
3. **Modeling** — `XGBClassifier` trained on the remaining features (`ra`, `dec`, `u`, `g`/`r`, `redshift`, etc.).
4. **Hyperparameter tuning** — `RandomizedSearchCV` (50 iterations, 5-fold CV, `f1_macro` scoring) over `n_estimators`, `max_depth`, `learning_rate`, and `colsample_bytree`.
5. **Evaluation** — accuracy, classification report, confusion matrix (visualized), feature importances, and 5-fold cross-validation to check result stability.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline XGBoost | ~98.8% | ~0.98 |
| Tuned (RandomizedSearchCV) | ~98.9% | ~0.98 |

5-fold cross-validation (`f1_macro`): **mean ≈ 0.979, std ≈ 0.004** — indicating the model's performance is stable across different data splits, not a result of a lucky train/test split.

`redshift` is the single most important feature by a wide margin, which aligns with astrophysics: stars sit within our own galaxy (redshift ≈ 0), while galaxies and especially quasars are far more distant, giving them measurably higher redshift.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

## Usage

1. Clone the repository
2. Place `digitalskysurvey.csv` in the project root
3. Open and run `XGBoostClassifier.ipynb` in Jupyter

## Project Structure

```
.
├── XGBoostClassifier.ipynb   # Main notebook: EDA, feature selection, modeling, tuning
├── digitalskysurvey.csv      # SDSS dataset
└── README.md
```
