# Methodology

Step-by-step of the analysis, in the order the code runs
(`python src/run_all.py`). Written for a reader with a mathematics /
statistics background — no machine-learning machinery required.

## 0. Data assembly (`00_generate_sample_data.py`)

Lot-level auction results, one row per lot: artist, hammer price, sale date,
auction house, medium, dimensions, signature, provenance and exhibition
counts. The bundled file is synthetic but mirrors the shape of real London
sale results (Christie's, Sotheby's, Bonhams, 2005–2015).

## 1. Cleaning (`01_data_cleaning.py`)

- **Unsold lots** are dropped — they have no price (censored data), and are
  disclosed as a limitation rather than silently imputed.
- Prices are **deflated to constant 2005 GBP** with the UK CPI so the index
  measures real appreciation.
- Art prices are **log-normal**, so modelling happens on `log(price)`;
  coefficients then read as percentage effects.
- Log prices are **winsorized** at the 0.5/99.5 percentiles against
  data-entry errors and trophy lots.

## 2. Exploration (`02_eda.py`)

Distributions, boxplots of log price by auction house / segment / medium,
and a naive quarterly median-price path — the "before" picture that the
hedonic index will improve on.

## 3. Feature engineering (`03_feature_engineering.py`)

- Categorical variables → dummy variables with explicit baselines
  (Bonhams; Old Masters; works on paper; 2005Q1).
- **Artist fixed effects**: one dummy per artist — artist identity is the
  dominant price driver, and fixed effects absorb segment and tier
  differences automatically.
- **Time**: quarterly dummies for the index regression; year + season
  dummies for prediction (future quarters must never leak into training).
- Continuous: log artwork area (size elasticity), signature flag,
  provenance and exhibition counts.

## 4. Hedonic regression (`04_hedonic_regression.py`)

The workhorse:

```
log(price_i) = X_i β + ε_i
```

OLS with **heteroscedasticity-robust (HC1) standard errors**, since price
variance is not constant. Diagnostics: R² / adjusted R², variance inflation
factors, residuals-vs-fitted plot. A coefficient of 0.12 on `signed` means a
signature is associated with ≈12% higher price, holding everything else fixed.

## 5. Price index (`05_price_index.py`)

The quarterly time-dummy coefficients are the pure time effect — price
movement with artwork mix held constant:

```
Index_t = 100 · exp(β_t − β_base),   base quarter 2005Q1 = 100
```

This is the standard hedonic-index construction from the art-economics
literature.

## 6. Time series (`06_time_series.py`)

- **ADF test** for stationarity (difference if the unit root is not rejected).
- **Seasonal ARIMA** on the log index, order chosen by AIC grid search with
  seasonal period 4 — the London sale calendar (Feb / Jun–Jul / Nov–Dec)
  creates genuine quarterly seasonality.
- **Ljung-Box** check that no autocorrelation is left in the residuals.
- 4-quarter forecast with 95% confidence intervals.

## 7. Prediction (`07_prediction.py`)

A **chronological** train/test split — train on sales up to 2013, test on
2014–2015 (shuffling time-series data would leak the future). Compares OLS
against ridge regression (alpha by cross-validation), evaluated on the
original GBP scale with RMSE, MAE and MAPE after back-transformation.
Duan's smearing correction was tested and rejected (see the comment in
`07_prediction.py`): under heteroscedastic errors it worsened out-of-sample
performance, so the naive retransformation is used.

## 8. Segment comparison (`08_segment_comparison.py`)

- The hedonic model is fit **separately** within Old Masters,
  Impressionist & Modern, and Post-War & Contemporary, so each segment's
  price drivers can be read off directly.
- An **F-test** of segment interactions formally tests whether the drivers
  differ across segments (H₀: same slopes everywhere).

## What this does not do

No neural networks, no gradient boosting, no text mining — deliberately.
The questions (what drives prices? how do segments differ? where is the
market heading?) are answered more transparently with regression and
classical time-series tools, which is also what the underlying thesis used.
