# Exoplanet Classification

Classifying exoplanets as **Not Habitable**, **Likely Habitable**, or **Optimistically Habitable** using the [Habitable Worlds Catalog (HWC)](https://phl.upr.edu/hwc) from the Planetary Habitability Laboratory (PHL).

## Overview

This notebook walks through a full classification pipeline — from raw data retrieval to model evaluation and a real-world application — on exoplanet data. The target label, `P_HABITABLE`, is a 3-class variable:

| Value | Class |
|---|---|
| 0 | Not Habitable |
| 1 | Likely Habitable |
| 2 | Optimistically Habitable |

## Notebook Structure

- **A. Retrieve the Data** — Load the HWC dataset.
- **B. Explore the Data** — Exploratory data analysis: summary statistics, null-value checks, distributions, outlier detection, and scatter plots (e.g., mass vs. radius, orbital distance vs. habitable zone bounds, eccentricity comparisons, error-bar plots for measurement uncertainty).
- **C. Prepare the Dataset for Classification** — Preprocessing: median imputation for missing values, log-transform (`log1p`) for heavily skewed features, one-hot encoding for `P_DETECTION` (with rare categories grouped into "Other"), removal of redundant habitable-zone columns, and a stratified train/test split.
- **D. Train** — Training four classifiers:
  - **kNN** (tuned `k` via `f1_macro`-scored cross-validation, distance-weighted)
  - **Random Forest** (`class_weight='balanced'`)
  - **XGBoost** (multi-class log loss, sample-weighted for class imbalance)
  - **MLP** (small architecture, tuned hidden-layer size, SMOTE oversampling inside a CV pipeline for kNN/MLP)
- **E. Evaluate** — Model comparison using macro-F1, per-class recall, confusion matrices, and precision-recall curves (rather than accuracy or ROC/AUC, given class imbalance).
- **F. Application** — Classifying a new, hypothetical planet ("Coruscant b") using imputed values for unmeasured features.

## Key Results

| Model | Macro-F1 |
|---|---|
| XGBoost | 1.000 |
| Random Forest | 0.921 |
| MLP | 0.877 |
| kNN | 0.765 |

XGBoost and Random Forest handled the minority (habitable) classes best; kNN struggled most with minority-class recall.

## Data Source

- Habitable Worlds Catalog (HWC), Planetary Habitability Laboratory, University of Puerto Rico at Arecibo: https://phl.upr.edu/hwc
- Habitable zone boundary formulas: Kopparapu, R. K., et al. (2013). "Habitable Zones around Main-sequence Stars: New Estimates." *The Astrophysical Journal*, 765(2), 131. https://doi.org/10.1088/0004-637x/765/2/131

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook homework1.ipynb
```
