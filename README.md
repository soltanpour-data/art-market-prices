# Statistical Modeling and Time Series Analysis of Art Market Prices

A reproducible analysis of London painting-auction prices: **predicting auction
prices** and **explaining price differences across market segments** using
artist, auction house, artwork characteristics, and sale timing.

The pipeline follows the classic applied-statistics workflow — exploratory
data analysis, hedonic (multivariate) regression, a quality-adjusted price
index, and seasonal ARIMA time-series modeling. No black boxes: every result
is a coefficient, a plot, or a test you can read directly.

## Results at a glance (synthetic sample data)

| Analysis | Result |
|---|---|
| Hedonic regression | R² = 0.77 — artist, size, signature, provenance and house all significant |
| Price index | 2005Q1 = 100 → 2015Q4 ≈ 97 (real terms, quality-adjusted) |
| Time series | SARIMA(1,1,2)×(0,0,0,4) on the log index; 4-quarter forecast with 95% CI |
| Out-of-sample prediction | Test MAPE ≈ 31.5% (OLS vs ridge on the 2014–2015 holdout) |
| Segment comparison | F-test confirms price drivers differ by segment (F = 8.4, p < 1e-15) |

## How to read the results

**What drives prices?** The coefficient plot tells the story: artist identity
dominates (blue-chip names carry large positive fixed effects), then artwork
size (elasticity ≈ 0.36 — a 10% larger canvas sells for ≈3.6% more), a
signature (≈+13%), provenance (≈+6% per recorded entry), and the auction
house (Christie's/Sotheby's ≈ +10–13% over Bonhams, holding the artwork
constant).

![Largest price effects](output/fig_coefficients.png)

**How did the market move?** The hedonic index strips out changes in *what*
was sold each quarter, leaving the pure time effect. In real terms the
sample market is roughly flat over 2005–2015 once quality is controlled for —
the naive median-price path (see `output/fig_quarterly_median.png`) looks
bumpier because it confounds price changes with mix changes.

![Hedonic price index](output/fig_price_index.png)

**Where is it heading?** The log index is non-stationary (ADF p ≈ 0.52), so
the model differences once and captures the quarterly London sale calendar
with seasonal ARIMA. Residuals show no leftover autocorrelation (Ljung-Box
p ≈ 0.56).

![Index forecast](output/fig_forecast.png)

**Do segments behave differently?** Yes. Per-segment regressions plus a formal
F-test on segment interactions reject equal slopes (p < 1e-15): the size
premium is steepest for Post-War & Contemporary, while the signature premium
matters most for Old Masters — exactly the pattern art economists expect.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py        # runs steps 00–08 in order
```

Results land in `output/` (tables, model summaries, figures).

## Project structure

```
├── README.md                  # this file
├── METHODOLOGY.md             # step-by-step of the analysis
├── requirements.txt
├── data/
│   ├── README.md              # data dictionary + how to plug in real data
│   ├── raw_auction_data.csv   # synthetic lot-level London sales, 2005–2015
│   ├── clean_auction_data.csv
│   ├── model_data.csv         # modelling-ready design matrix
│   └── price_index.csv        # hedonic price index (2005Q1 = 100)
├── src/
│   ├── run_all.py             # runs the whole pipeline
│   ├── 00_generate_sample_data.py
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_hedonic_regression.py
│   ├── 05_price_index.py
│   ├── 06_time_series.py
│   ├── 07_prediction.py
│   └── 08_segment_comparison.py
└── output/                    # figures, model summaries, result tables
```

## Method in brief

1. **Clean** — drop unsold lots, deflate to 2005 GBP, log-transform prices, winsorize outliers.
2. **Explore** — distributions, prices by house/segment/medium, quarterly medians.
3. **Engineer features** — dummies for house, medium, artist (fixed effects) and sale quarter; log artwork area; signature, provenance, exhibition counts.
4. **Hedonic regression** — OLS of log price on characteristics with robust (HC1) standard errors; VIF and residual diagnostics.
5. **Price index** — quarterly time-dummy coefficients → quality-adjusted index, 2005Q1 = 100.
6. **Time series** — ADF stationarity test, seasonal ARIMA (quarterly London sale calendar), Ljung-Box diagnostics, 4-quarter forecast.
7. **Prediction** — chronological train/test split (train ≤ 2013, test 2014–2015); OLS vs ridge, RMSE/MAE/MAPE on the GBP scale.
8. **Segment comparison** — per-segment hedonic models plus an F-test of segment interactions.

Full detail: [METHODOLOGY.md](METHODOLOGY.md).

## Notes on methodology choices

- **Log prices.** Art prices are log-normal; modelling in logs both stabilises
  variance and turns coefficients into readable percentage effects.
- **Robust standard errors.** Price variance is not constant (heteroscedastic),
  so the regression reports HC1 robust standard errors rather than classical ones.
- **Artist fixed effects absorb segment effects.** Each artist belongs to
  exactly one segment, so segment dummies would be perfectly collinear —
  they are omitted wherever artist dummies are present.
- **No leakage in prediction.** The test set is strictly later in time than
  the training set, and quarterly dummies are replaced by year+season dummies
  so no future information enters training.
- **Back-transformation.** GBP-scale metrics use the naive retransformation
  `exp(predicted log price)`. Duan's smearing correction was tested and
  rejected: under heteroscedastic errors a single global factor over-corrects
  and worsened out-of-sample MAPE (31.5% → 35.0%).

## Data note

The bundled data is **synthetic** (seeded, so fully reproducible) with the
structure of real auction results — 4,000 lots across Christie's, Sotheby's
and Bonhams, 2005–2015. To run on real data, replace
`data/raw_auction_data.csv` with actual lot-level results following the
schema in [data/README.md](data/README.md) and re-run from step 01.

## Limitations

- Only sold lots are observed (unsold "bought-in" lots are censored) — a
  classic selection issue in art-market research.
- The index controls for observed characteristics only; taste, hype and
  condition live in the residuals.
- Thin data for minor artists makes individual artist effects noisy.

## License

MIT — see LICENSE.
