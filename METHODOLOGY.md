# Methodology

Step-by-step account of the analysis, so every choice is inspectable.
Numbers below are the values produced by the current run.

## 0. Data (`00_fetch_data.py`)

Real transaction data: the `sales` data frame from
[mdlincoln/londonauctions](https://github.com/mdlincoln/londonauctions)
(Lincoln & Fox 2016), itself transcribed from the Getty Provenance Index's
British sales catalogues, 1780–1835. Downloaded once (~5 MB .rda) and
converted to CSV with `pyreadr`. The CSVs live in `data/` but are gitignored
— anyone can regenerate them with this script.

## 1. Cleaning (`01_data_cleaning.py`)

- **Sample**: `transaction_type == "sold"` with a valid positive
  `transaction_amt` (decimal pounds; guineas already converted). 304,788
  raw transactions → 166,897 non-sold dropped → 705 missing/zero prices
  dropped → **137,186 lots**.
- **Artist**: the `artists` field lists attributions separated by `;`
  (catalogues list the principal attribution first); the primary one is
  kept, brackets stripped (`[Anonymous]` → `Anonymous`).
- **Subject**: the catalogue `title` is classified into 8 market segments
  with a documented keyword rule (see `SUBJECT_RULES` in the script):
  religious, mythological, portrait, landscape, still_life, animal, genre,
  other. Priority order resolves multi-matches (e.g. "A Nymph asleep, in a
  Landscape" → mythological). Resulting mix: other 34%, landscape 30%,
  portrait 12%, religious 10%, genre/mythological/still_life/animal the rest.
- **Prices**: log-transformed (log-normal distribution) and winsorized at
  the 0.5/99.5 percentiles (1,369 lots) against data-entry errors. Prices
  are **nominal** pounds — see Limitations in the README.

## 2. EDA (`02_eda.py`)

Median price £2.30; mean £13.92 — heavily right-skewed, hence logs.
Christie's dominates volume (114k of 305k raw transactions); Foster and
Phillips follow. Sales are strongly seasonal: March–July is the London
season, August–October nearly dead (month dummies capture this).

## 3. Features (`03_feature_engineering.py`)

- Artists: fixed-effect dummies for the 40 most-traded artists
  (`Anonymous`, Rubens, Teniers, Rembrandt, Titian, ...); everyone else →
  `Other` baseline. 40 dummies on 137k rows is easily estimable and absorbs
  school/period differences alongside name recognition.
- Houses: dummies for the 10 largest houses (`Other` baseline).
- Subjects: dummies for all 8 segments (`other` baseline).
- Time: annual dummies 1781–1835 (1780 baseline) for the index; a numeric
  `year_trend` + month dummies for prediction (no future leakage).
- Final design matrix: **137,186 × 123 features**.

## 4. Hedonic regression (`04_hedonic_regression.py`)

`log(price_i) = X_i β + ε_i`, OLS with **HC1 heteroscedasticity-robust**
standard errors. R² = **0.275** (adj. 0.275) — modest, as expected for noisy
historical catalogue data: condition, size, medium and buyer competition
are unobserved and live in the residuals.

Headline effects (all p < 0.001 unless noted):
- Artists: Poussin +138%, Claude +128%, Canaletti +116%; Anonymous −70%.
- Houses: Coxe +95%, Phillips +63%; Southgate −75%, Wheatley −71%.
- Subjects: religious +12%, genre +11%; still life −33%, portrait −39%.
- Season: May +75% vs January (the London season effect).

Diagnostics: VIFs on house/subject dummies all < 5 (no multicollinearity;
artists span houses, years and subjects, so no nesting issue); residual plot
shows mild heteroscedasticity, hence the robust SEs.

## 5. Price index (`05_price_index.py`)

`Index_t = 100 · exp(β_t)` from the annual time dummies — the pure time
effect holding artist, house, subject and season constant. 1780 = 100 →
1835 ≈ **209** (nominal). The naive annual-median path is bumpier because it
confounds price changes with mix changes.

## 6. Time series (`06_time_series.py`)

- ADF test on the log index: statistic −4.68, **p = 0.0001** → stationary,
  no unit root.
- Grid search over ARIMA(p,d,q), p,q ∈ {0,1,2}, d ∈ {0,1} by AIC → best is
  **ARIMA(0,1,2), AIC = 15.6**. (Annual data → no seasonal component.)
- Ljung-Box p ≈ 1.0 → no leftover autocorrelation.
- 5-year forecast (1836–1840): flat at ≈ 236 index points with wide 95%
  bands — the honest answer for a 56-point annual series.

## 7. Prediction (`07_prediction.py`)

Chronological split: train on ≤ 1824 (86,525 lots), test on 1825–1835
(50,661 lots) — never shuffled. Annual dummies are replaced by `year_trend`
so future years can't leak into training; two house dummies with zero
training variance are dropped. OLS vs ridge (α chosen by CV = 162).

Test performance: R² = 0.199 on the log scale; **MdAPE 71.6%**, log-RMSE
1.47, median bias £0.35. OLS and ridge are virtually identical — with
137k rows and 100+ dummies there is little left for shrinkage to fix, and
the errors are dominated by idiosyncratic variation (condition, exact
attribution wording) that the features can't see. The plain MAPE (248%)
is reported but misleading: 47% of lots sell under £2, where tiny absolute
misses become huge percentage errors.

Back-transformation is naive `exp(predicted log price)`; Duan's smearing
correction was tested and rejected (worsened error under heteroscedasticity).

## 8. Segment comparison (`08_segment_comparison.py`)

- Per-subject hedonic fits: R² ranges from 0.19 (still life) to 0.28
  (religious) — characteristics explain prices differently per segment.
- F-test of segment × house-premium interactions vs the pooled model:
  **F = 29.5, p ≈ 0** → house premia differ significantly across subject
  segments (e.g. Christie's is +24% for still lifes but −47% for religious
  works, relative to baseline). Artists are not nested in subjects, so the
  interactions are identified.
