# Statistical Modeling and Time Series Analysis of Art Market Prices

A reproducible analysis of **real historical London painting-auction prices,
1780–1835**: predicting hammer prices and explaining price differences
across market segments using artist, auction house, subject matter, and
sale timing.

The data are 137,186 sold lots from the Getty Provenance Index, via the
public research package
[mdlincoln/londonauctions](https://github.com/mdlincoln/londonauctions)
(Matthew Lincoln & Abram Fox, "The Temporal Dimensions of the London Art
Auction, 1780-1835", *British Art Studies* 4, 2016). Run
`python src/00_fetch_data.py` once to download it — the pipeline then runs
offline.

The analysis follows the classic applied-statistics workflow — exploratory
data analysis, hedonic (multivariate) regression, a quality-adjusted price
index, and ARIMA time-series modeling. No black boxes: every result is a
coefficient, a plot, or a test you can read directly.

## Results at a glance

| Analysis | Result |
|---|---|
| Hedonic regression | R² = 0.28 — artist, house, subject and timing all significant |
| Price index | 1780 = 100 → 1835 ≈ 209 (nominal, quality-adjusted) |
| Time series | ARIMA(0,1,2) on the annual log index; 5-year forecast with 95% CI |
| Out-of-sample prediction | MdAPE ≈ 72% on the 1825–1835 holdout (OLS vs ridge) |
| Segment comparison | F-test confirms house premia differ by subject segment (F = 29.5, p ≈ 0) |

## How to read the results

**What drives prices?** Artist identity dominates, exactly as art history
would predict: Poussin (+138%), Claude (+128%) and Canaletti (+116%) carry
the largest premia, while unattributed ("Anonymous") works sell for ~70%
less than comparable attributed lots. Auction houses differ sharply too —
Phillips (+63%) and Coxe (+95%) over the baseline — and so do subjects:
religious works (+12%) and genre scenes (+11%) outprice portraits (−39%)
and still lifes (−33%). Sales cluster in the London season (March–July),
and May lots command a clear seasonal premium.

![Largest price effects](output/fig_coefficients.png)

**How did the market move?** The hedonic index strips out changes in *what*
was sold each year, leaving the pure time effect. Nominal prices roughly
double over the period — but remember these are nominal pounds across the
Napoleonic wars, so the index confounds real appreciation with wartime
inflation (see Limitations).

![Hedonic price index](output/fig_price_index.png)

**Where is it heading?** The annual log index is stationary (ADF p ≈ 0.0001),
and AIC selects a simple ARIMA(0,1,2). Residuals show no leftover
autocorrelation (Ljung-Box p ≈ 1.0). The 5-year forecast is flat with wide
bands — honest uncertainty for a thin annual series.

![Index forecast](output/fig_forecast.png)

**Do segments behave differently?** Yes. Per-segment regressions plus a
formal F-test on segment×house interactions reject equal slopes (p ≈ 0):
Christie's premium, for instance, is positive for still lifes but negative
for religious works — the market valued houses differently depending on
what was on the block.

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
│   ├── README.md              # data dictionary + source notes
│   └── (fetched locally, gitignored)
│       ├── sales.rda                  # raw R data from londonauctions
│       ├── raw_london_auctions.csv    # 304,788 transactions
│       ├── clean_london_auctions.csv # 137,186 sold lots with valid prices
│       ├── model_data.csv             # modelling-ready design matrix
│       └── price_index.csv            # hedonic price index (1780 = 100)
├── src/
│   ├── run_all.py             # runs the whole pipeline
│   ├── 00_fetch_data.py       # downloads & converts the real data
│   ├── 01_data_cleaning.py    # sold lots, artist/subject parsing, log prices
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py  # artist/house/subject/year/month dummies
│   ├── 04_hedonic_regression.py
│   ├── 05_price_index.py
│   ├── 06_time_series.py
│   ├── 07_prediction.py       # chronological train/test, OLS vs ridge
│   └── 08_segment_comparison.py
└── output/                    # figures, model summaries, result tables
```

## Method in brief

1. **Fetch** — download the londonauctions `sales` data set (304,788
   transactions) and convert from R's .rda format.
2. **Clean** — keep sold lots with valid positive prices (137,186); parse
   primary artist; classify catalogue titles into 8 subject segments with a
   documented keyword rule; log-transform and lightly winsorize prices.
3. **Explore** — price distributions, prices by house/subject, lots per year.
4. **Engineer features** — dummies for the top-40 artists, top-10 houses,
   subjects, sale years and months.
5. **Hedonic regression** — OLS of log price on characteristics with robust
   (HC1) standard errors; VIF and residual diagnostics.
6. **Price index** — annual time-dummy coefficients → quality-adjusted index,
   1780 = 100.
7. **Time series** — ADF stationarity test, ARIMA by AIC, Ljung-Box
   diagnostics, 5-year forecast.
8. **Prediction** — chronological split (train ≤ 1824, test 1825–1835); OLS
   vs ridge, robust metrics on the GBP scale.
9. **Segment comparison** — per-segment hedonic models plus an F-test of
   segment interactions on house premia.

Full detail: [METHODOLOGY.md](METHODOLOGY.md).

## Notes on methodology choices

- **Log prices.** Art prices are log-normal; modelling in logs both stabilises
  variance and turns coefficients into readable percentage effects.
- **Robust standard errors.** Price variance is not constant
  (heteroscedastic), so the regression reports HC1 robust standard errors
  rather than classical ones.
- **Median-based prediction metrics.** ~47% of lots sell for under £2, so
  MAPE explodes (a £1 miss on a 50p lot is a 200% error). MdAPE (71.6%) and
  log-scale RMSE (1.47) are the honest headline numbers; median bias is just
  £0.35.
- **No leakage in prediction.** The test set is strictly later in time than
  the training set, and annual dummies are replaced by a year trend plus
  month dummies so no future information enters training.
- **Back-transformation.** GBP-scale metrics use the naive retransformation
  `exp(predicted log price)`. Duan's smearing correction was tested and
  rejected: under heteroscedastic errors a single global factor over-corrects
  and worsened out-of-sample error.

## Limitations

- Prices are **nominal pounds** — no historical deflator is applied, so the
  index mixes real price changes with Napoleonic-era inflation. Within-year
  hedonic *comparisons* (the coefficients) are unaffected, since year
  dummies absorb the price level.
- Only sold lots are observed (bought-in/withdrawn lots are censored) — a
  classic selection issue in art-market research.
- Subject classification is a keyword rule on 18th-century catalogue prose;
  34% of lots fall into "other".
- The index controls for observed characteristics only; condition,
  provenance and buyer competition live in the residuals.
- Thin data for minor artists makes individual artist effects noisy.

## License

MIT — see LICENSE.

## Data citation

Lincoln, Matthew & Fox, Abram (2016). "The Temporal Dimensions of the
London Art Auction, 1780-1835." *British Art Studies* 4. Data: Getty
Provenance Index, via https://github.com/mdlincoln/londonauctions
(archived on Zenodo, DOI 10.5281/zenodo.60284).
