"""
Step 6 — Time series analysis of the price index.

- Stationarity check (Augmented Dickey-Fuller test).
- Seasonal ARIMA on the log index: small grid search over
  (p,d,q) x (P,D,Q,4) by AIC (quarterly data -> seasonal period 4,
  capturing the London auction calendar).
- Residual diagnostics (Ljung-Box) and a 4-quarter forecast.

Output: output/arima_summary.txt, output/fig_forecast.png
"""

import itertools
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

idx = pd.read_csv("data/price_index.csv", index_col="quarter")
y = np.log(idx["index"])
y.index = pd.PeriodIndex(y.index, freq="Q")

# --- stationarity -----------------------------------------------------------
adf_stat, adf_p, *_ = adfuller(y)[:2]
print(f"ADF test on log index: statistic={adf_stat:.3f}, p-value={adf_p:.4f}")
print("-> " + ("stationary (reject unit root)."
               if adf_p < 0.05 else
               "non-stationary: differencing is needed."))

# --- seasonal ARIMA grid search by AIC --------------------------------------
best_aic, best_order, best_seasonal = np.inf, None, None
orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (0, 1, 1), (2, 1, 2)]
seasonals = [(1, 0, 0, 4), (0, 0, 0, 4), (1, 1, 1, 4)]
for order, seasonal in itertools.product(orders, seasonals):
    try:
        fit = SARIMAX(y, order=order, seasonal_order=seasonal,
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        if fit.aic < best_aic:
            best_aic, best_order, best_seasonal = fit.aic, order, seasonal
    except Exception:
        continue
print(f"\nBest SARIMA{best_order}x{best_seasonal}: AIC={best_aic:.1f}")

res = SARIMAX(y, order=best_order, seasonal_order=best_seasonal,
              enforce_stationarity=False,
              enforce_invertibility=False).fit(disp=False)
with open("output/arima_summary.txt", "w") as f:
    f.write(str(res.summary()))

# --- diagnostics -------------------------------------------------------------
lb = acorr_ljungbox(res.resid, lags=[8], return_df=True)
print(f"Ljung-Box p-value (lag 8): {lb['lb_pvalue'].iloc[0]:.4f} "
      f"(> 0.05 means no leftover autocorrelation)")

# --- 4-quarter forecast -------------------------------------------------------
fc = res.get_forecast(steps=4)
mean_gbp = np.exp(fc.predicted_mean)          # back to index scale
ci = np.exp(fc.conf_int(alpha=0.05))

future = pd.period_range(y.index[-1] + 1, periods=4, freq="Q").astype(str)
print("\nForecast (index points):")
for q_, m_, lo_, hi_ in zip(future, mean_gbp, ci.iloc[:, 0], ci.iloc[:, 1]):
    print(f"  {q_}: {m_:.1f}  (95% CI {lo_:.1f} - {hi_:.1f})")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(idx.index, idx["index"], label="Hedonic index")
ax.plot(future, mean_gbp.values, "o--", label="Forecast")
ax.fill_between(future, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.25)
ax.set_title("Art price index with 4-quarter SARIMA forecast")
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.legend(); fig.tight_layout()
fig.savefig("output/fig_forecast.png"); plt.close(fig)

print("\nSaved ARIMA summary and forecast plot to output/")
