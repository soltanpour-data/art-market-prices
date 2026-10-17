# Data & Methods — Full Description

A complete account of the data, every transformation applied to it, how
relationships were measured, and what each model does. Companion to
`README.md` (results) and `METHODOLOGY.md` (step-by-step narrative).

---

## 1. Data source and provenance

Real historical data — no synthetic stand-ins. The analysis uses the
`sales` data frame from the public research package
**[mdlincoln/londonauctions](https://github.com/mdlincoln/londonauctions)**:

> Matthew Lincoln & Abram Fox (2016). "The Temporal Dimensions of the
> London Art Auction, 1780–1835." *British Art Studies* 4.
> Data originally transcribed from the **Getty Provenance Index** (British
> art sales catalogues). Archived on Zenodo: DOI 10.5281/zenodo.60284.

`src/00_fetch_data.py` downloads `sales.rda` (~5 MB) once and converts it
to CSV with `pyreadr`. All CSVs are gitignored and regenerated locally.

---

## 2. Dataset size and structure

### Raw data — `data/raw_london_auctions.csv`: 304,788 rows × 21 columns

| Column | Type | Meaning | Missing |
|---|---|---|---|
| `cat_no` | string | Getty catalogue identifier | 0 |
| `lot_no` | string | Lot number within catalogue | 0 |
| `objnum` | integer | Object/row number | 0 |
| `auction_house` | string | e.g. Christie's, Foster (Edward), Phillips (Harry) | 0 |
| `date` | string | Sale date (YYYY-MM-DD) | 0 |
| `transaction_type` | string | sold / bought in / withdrawn / unknown / … | 121 |
| `type_is_uncertain` | boolean-like | Getty's uncertainty flag on the type | 121 |
| `amt_is_uncertain` | boolean-like | Uncertainty flag on the amount | 73,932 |
| `has_double_price` | boolean-like | Two prices recorded for the lot | 73,932 |
| `transaction_amt` | float | Hammer price, decimal pounds (guineas converted) | 74,170 |
| `year`, `month`, `week`, `yday`, `mday`, `wday` | integer | Date parts | 0 |
| `transaction` | string | Verbatim price as printed (e.g. "2 10s 6d") | 73,932 |
| `artists` | string | Attribution(s), `;`-separated; `[Anonymous]` if unattributed | 1 |
| `title` | string | Catalogue title/description of the work | 2 |
| `sellers` | string | Seller as transcribed | 84,550 |
| `buyers` | string | Buyer as transcribed | 176,842 |

Coverage: London, **1780–1835** (56 years). Christie's accounts for
114,126 raw transactions; Foster (Edward) 64,399; Phillips (Harry) 27,466;
then Greenwood, Stanley, Coxe, Jones and dozens of smaller houses.
`transaction_type` breaks down as: sold 137,891 · unknown 91,398 ·
bought in 59,779 · sold-or-bought-in 9,791 · passed/withdrawn ≈ 5,700.

### Clean analysis data — `data/clean_london_auctions.csv`: 137,186 rows × 8 columns

| Column | Type | Meaning |
|---|---|---|
| `artist` | string | Primary attribution (first listed, brackets stripped) |
| `subject` | string | Classified segment: religious / mythological / portrait / landscape / still_life / animal / genre / other |
| `auction_house` | string | As transcribed |
| `sale_year` | integer | 1780–1835 |
| `sale_month` | integer | 1–12 |
| `title` | string | Original catalogue title (kept for reference) |
| `price_gbp` | float | Hammer price, nominal decimal pounds |
| `log_price` | float | `ln(price_gbp)` — **the dependent variable** |

The modelling matrix (`data/model_data.csv`) expands this to
**137,186 × 123 features** (see §4).

---

## 3. Data cleaning (`src/01_data_cleaning.py`)

Each step is a deliberate sample-definition choice, reported with counts:

1. **Keep only sold lots** (`transaction_type == "sold"`): 166,897
   non-sold lots dropped. Bought-in, withdrawn and "unknown" lots are
   *censored* observations — no meaningful hammer price exists — so this
   is disclosed as a selection limitation rather than silently kept.
2. **Drop missing/zero prices**: 705 sold lots had `NaN` or £0 amounts
   dropped. **137,186 lots remain.**
3. **Parse the artist**: the `artists` field lists attributions separated
   by `;` (catalogues list the principal attribution first). The first is
   kept and catalogue brackets stripped (`[Anonymous]` → `Anonymous`;
   multi-attribution lots like `"[Anonymous]; De Witt"` → `Anonymous`).
   14,845 distinct artist strings; the top is `Anonymous` (21,931 lots),
   then Rubens, Teniers, Rembrandt, Titian, Guido, Cuyp, Morland.
4. **Classify the subject**: the catalogue `title` is mapped to 8 market
   segments with a documented keyword rule (`SUBJECT_RULES` in the script,
   priority-ordered; first match wins). Keyword lists were built by
   inspecting the titles. Resulting mix: other 34.2%, landscape 30.1%,
   portrait 12.3%, religious 9.8%, genre 4.9%, mythological 3.4%,
   still_life 2.9%, animal 2.3%.
5. **Log-transform prices** (see §4) and **winsorize** `log_price` at the
   0.5/99.5 percentiles (1,369 lots) as protection against data-entry
   errors in the transcription.

---

## 4. Normalization and transformation

- **Log transformation** — the dependent variable is `log_price =
  ln(price_gbp)`. Art prices are extremely right-skewed (raw skewness
  **38.6**, median £2.30 vs mean £13.92, max £10,080); after logging,
  skewness is **0.0** — essentially symmetric. Modelling in logs both
  stabilises variance and turns every coefficient into a readable
  percentage effect (`exp(β) − 1`).
- **Winsorization** (0.5/99.5%) clips transcription outliers without
  deleting observations.
- **One-hot (dummy) encoding** of all categorical predictors:
  - artists → dummies for the **40 most-traded** artists (`Other` baseline);
  - houses → dummies for the **10 largest** houses (`Other` baseline);
  - subjects → dummies for all 8 segments (`other` baseline);
  - years → annual dummies 1781–1835 (**1780 baseline**) for the index;
  - months → dummies Feb–Dec (**January baseline**) for seasonality;
  - plus a numeric `year_trend = sale_year − 1780` for the prediction task.
- **No deflation**: prices are nominal pounds. No reliable 1780–1835
  deflator was available, so this is stated as a limitation; within-year
  hedonic comparisons are unaffected because year dummies absorb the
  price level.
- **No scaling** was needed for OLS (dummies are 0/1); features were
  standardised only inside the ridge-regression branch of the prediction
  step.

---

## 5. Exploratory data analysis (`src/02_eda.py`)

| Question | How measured | Finding |
|---|---|---|
| Price distribution | Histogram of `log_price` (60 bins) | Symmetric, bell-shaped after logging (skew 0.0) |
| Price by house | Boxplots of `log_price` for top-8 houses (outliers hidden) | Large level differences (e.g. Phillips high, Foster low) |
| Price by subject | Boxplots per segment | Religious/genre highest, portraits/still lifes lowest |
| Market activity | Bar chart of lots sold per year | ~1.1–1.5k/yr in the 1780s, rising to ~4.5–6k/yr in the 1830s; 1786 anomalously thin (372 lots) |
| Naive price path | Annual median `log_price` | Bumpy — motivates the quality-adjusted index |
| Volume leaders | Value counts | Artists: Anonymous, Rubens, Teniers, Rembrandt…; Houses: Christie's, Foster, Phillips… |

Numeric correlations with `log_price`: `price_gbp` 0.32 (nonlinear by
construction), `sale_year` −0.07, `sale_month` −0.12 (raw — the seasonal
pattern reverses once composition is controlled for; the regression finds
a May premium).

---

## 6. Collinearity and correlation — how they were checked

- **Variance Inflation Factors (VIF)** computed on the house + subject
  dummy block of the hedonic design matrix (via
  `statsmodels.stats.outliers_influence.variance_inflation_factor`).
  All VIFs **< 5** (highest: Christie's 4.6, Foster 4.0) — no problematic
  multicollinearity.
- **No exact nesting**: artists are not nested in subjects (every major
  artist's works span several subjects), houses or years, so artist fixed
  effects coexist with subject/house/year dummies without perfect
  collinearity. (This differs from designs where each artist belongs to
  one segment, which would force a choice.)
- **Correlation structure** beyond VIF was assessed through the regression
  itself: with 123 features and 137k rows, pairwise correlations are less
  informative than the partial effects the OLS coefficients report
  directly. Residual diagnostics (residual-vs-fitted plot) showed mild
  heteroscedasticity — which is why all inference uses HC1 robust
  standard errors.

---

## 7. Dependent variable

**`log_price`** — the natural log of the hammer price in nominal decimal
pounds — in every model. All GBP-scale performance metrics are obtained by
back-transforming predictions with the naive retransformation
`exp(predicted log price)` (Duan's smearing correction was tested and
rejected: under heteroscedastic errors it worsened out-of-sample error).

---

## 8. Dependencies and relationships analyzed

| Relationship | Where | How measured | Result |
|---|---|---|---|
| Artist → price | Hedonic OLS (§9.1) | 40 fixed-effect dummies, HC1 SEs | Dominant driver: Poussin +138%, Claude +128%, Canaletti +116%; Anonymous −70% (all p < 0.001) |
| Auction house → price | Hedonic OLS | 10 house dummies | Coxe +95%, Phillips +63%; Southgate −75%, Wheatley −71% |
| Subject → price | Hedonic OLS | 7 subject dummies | Religious +12%, genre +11%; still life −33%, portrait −39% |
| Sale timing → price | Hedonic OLS | 55 annual dummies + 11 month dummies | Seasonality: May +75% vs January; annual dummies → price index |
| Time → market level | Price index (§9.2) | `Index_t = 100·exp(β_t)`, 1780 = 100 | 1780 → 1835: 100 → ≈209 (nominal, quality-adjusted) |
| Index dynamics | ARIMA (§9.3) | ADF test, AIC grid search, Ljung-Box | Stationary (ADF p = 0.0001); ARIMA(0,1,2); no residual autocorrelation |
| Segment × house | Segment comparison (§9.5) | Per-segment OLS + F-test on segment×house interactions | **F = 29.5, p ≈ 0** — house premia differ across subjects (e.g. Christie's +24% for still lifes, −47% for religious works) |
| Feature set → individual prices | Prediction (§9.4) | Chronological holdout, OLS vs ridge | Test R² = 0.199 (log scale); MdAPE 71.6%; median bias £0.35 |

---

## 9. What was predicted, and the models used

### 9.1 Hedonic (multivariate) regression — `04_hedonic_regression.py`
**Predicts:** `log_price` of each lot from its characteristics.
**Model:** Ordinary Least Squares,
`log(price_i) = X_i β + ε_i`, with **HC1 heteroscedasticity-robust**
standard errors (price variance is not constant across the price range).
R² = **0.275** (adj. 0.275), F = 476.2. Coefficients are percentage price
effects; e.g. `artist_Canaletti = +0.771` means +116% vs the baseline,
holding house, subject, year and month constant.

### 9.2 Hedonic price index — `05_price_index.py`
**Predicts:** the market's pure time path. **Model:** the annual dummy
coefficients from §9.1 transformed as `Index_t = 100·exp(β_t − β_1780)` —
a Laspeyres-style quality-adjusted index holding the mix of artists,
houses and subjects constant.

### 9.3 Time-series model — `06_time_series.py`
**Predicts:** the index 5 years ahead (1836–1840) with 95% intervals.
**Model:** **ARIMA(0,1,2)** on the annual log index, selected by AIC grid
search over p,q ∈ {0,1,2}, d ∈ {0,1} (no seasonal component — annual data).
Augmented Dickey-Fuller test for stationarity; Ljung-Box test confirms no
leftover autocorrelation (p ≈ 1.0). Forecast: flat at ≈236 index points,
wide bands — honest uncertainty for a 56-point series.

### 9.4 Out-of-sample price prediction — `07_prediction.py`
**Predicts:** individual lot prices on a strictly future holdout
(train ≤ 1824: 86,525 lots; test 1825–1835: 50,661 lots — never shuffled).
**Models:** (a) plain **OLS**; (b) **ridge regression** (L2-penalised OLS,
`RidgeCV`, α = 162 chosen by cross-validation on standardised features).
Annual dummies are replaced by the numeric trend + month dummies so no
future information leaks into training. Reported on the GBP scale:
MdAPE 71.6%, log-RMSE 1.47, median bias £0.35. OLS ≈ ridge — with 137k
rows there is little left for shrinkage; errors are dominated by
unobserved condition/attribution nuance. (Plain MAPE = 248% is reported
but misleading: 47% of lots sell under £2.)

### 9.5 Segment comparison — `08_segment_comparison.py`
**Tests:** whether price drivers differ across subject segments.
**Models:** (a) the hedonic OLS re-fit **separately within each of the 8
subjects** (R² 0.19–0.28); (b) a pooled OLS vs a pooled OLS with
**segment × house-premium interaction terms**, compared with an **F-test**
for the joint significance of the interactions.

---

## 10. Limitations (affecting interpretation)

- Nominal pounds: the index mixes real appreciation with Napoleonic-era
  inflation; hedonic *comparisons* are unaffected (year dummies absorb the
  level).
- Only sold lots observed (bought-in/withdrawn censored) — selection bias.
- Subject labels are a keyword rule on 18th-century prose; 34% "other".
- Unobserved: condition, size, medium, provenance, buyer competition —
  they live in the residuals (hence R² = 0.28).
- 1786 is thinly covered (372 lots); its index point is noisy.
