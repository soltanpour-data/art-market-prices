"""
Step 6 — Time series analysis of the price index.

- Stationarity check (Augmented Dickey-Fuller test).
- ARIMA on the annual log index: grid search over (p,d,q) by AIC.
  (Annual data -> no seasonal component; 56 observations keep the grid small.)
- Residual diagnostics (Ljung-Box) and a 5-year forecast.

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
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

idx = pd.read_csv("data/price_index.csv")
y = np.log(idx.set_index("year")["index"])
y.index = pd.PeriodIndex(y.index, freq="Y")

# --- stationarity -----------------------------------------------------------
adf_stat, adf_p = adfuller(y)[:2]
print(f"ADF test on log index: statistic={adf_stat:.3f}, p-value={adf_p:.4f}")
print("-> " + ("stationary (reject unit root)."
               if adf_p < 0.05 else
               "non-stationary: differencing is needed."))

# --- ARIMA grid search by AIC -------------------------------------------------
best_aic, best_order = np.inf, None
for order in itertools.product([0, 1, 2], [0, 1], [0, 1, 2]):
    try:
        fit = ARIMA(y, order=order).fit()
        if fit.aic < best_aic:
            best_aic, best_order = fit.aic, order
    except Exception:
        continue
print(f"\nBest ARIMA{best_order}: AIC={best_aic:.1f}")

res = ARIMA(y, order=best_order).fit()
with open("output/arima_summary.txt", "w") as f:
    f.write(str(res.summary()))

# --- diagnostics -------------------------------------------------------------
lb = acorr_ljungbox(res.resid, lags=[8], return_df=True)
print(f"Ljung-Box p-value (lag 8): {lb['lb_pvalue'].iloc[0]:.4f} "
      f"(> 0.05 means no leftover autocorrelation)")

# --- 5-year forecast ----------------------------------------------------------
fc = res.get_forecast(steps=5)
mean_idx = np.exp(fc.predicted_mean)          # back to index scale
ci = np.exp(fc.conf_int(alpha=0.05))

future = (y.index[-1].year + np.arange(1, 6)).astype(int)
print("\nForecast (index points):")
for yr, m_, lo_, hi_ in zip(future, mean_idx, ci.iloc[:, 0], ci.iloc[:, 1]):
    print(f"  {yr}: {m_:.1f}  (95% CI {lo_:.1f} - {hi_:.1f})")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(idx["year"], idx["index"], label="Hedonic index")
ax.plot(future, mean_idx.values, "o--", label="Forecast")
ax.fill_between(future, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.25)
ax.set_title("Art price index with 5-year ARIMA forecast (nominal £)")
ax.set_xlabel("Year"); ax.legend()
fig.tight_layout(); fig.savefig("output/fig_forecast.png"); plt.close(fig)

print("\nSaved ARIMA summary and forecast plot to output/")
