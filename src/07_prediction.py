"""
Step 7 — Out-of-sample price prediction.

Chronological train/test split (train: sales up to 1824, test: 1825-1835 —
time series data must never be shuffled). Compares plain OLS against ridge
regression. Time enters as a numeric trend plus month dummies (annual
dummies would leak future years into training).

Metrics are reported on the original GBP scale after back-transformation.
Duan's smearing correction was tested and rejected (see comment in metrics):
with heteroscedastic errors it worsened out-of-sample MAPE.

Output: output/prediction_comparison.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/model_data.csv")

meta = ["artist", "subject", "auction_house", "sale_year", "sale_month",
        "log_price", "price_gbp", "year_trend"]
# Drop annual dummies (they would leak future years); keep the trend
drop_y = [c for c in df.columns if c.startswith("y_")]
X_all = df.drop(columns=meta + drop_y)
y_log = df["log_price"]
y_gbp = df["price_gbp"]

train = df["sale_year"] <= 1824
Xtr, Xte = X_all[train], X_all[~train]
# Drop zero-variance columns in training (e.g. artists absent before 1825)
nzv = Xtr.columns[Xtr.nunique() <= 1]
if len(nzv):
    print(f"Dropped {len(nzv)} zero-variance columns: {list(nzv)}")
Xtr, Xte = Xtr.drop(columns=nzv), Xte.drop(columns=nzv)
ytr_log, yte_log = y_log[train], y_log[~train]
yte_gbp = y_gbp[~train]
print(f"Train: {train.sum():,} lots (<=1824) | Test: {(~train).sum():,} lots (1825-1835)")


def metrics(y_true_gbp, pred_log, y_true_log):
    # Naive back-transformation. Duan's smearing (multiplying by
    # mean(exp(train residuals))) was tried and rejected: with
    # heteroscedastic errors a single global factor over-corrects and
    # worsened test MAPE here.
    # MAPE explodes because ~47% of lots sell under £2 (a £1 miss on a
    # 50p lot is a 200% error), so the median-based MdAPE is the honest
    # headline metric alongside log-scale RMSE.
    pred_gbp = np.exp(pred_log)
    err = y_true_gbp - pred_gbp
    ape = np.abs(err / y_true_gbp)
    return {
        "RMSE_GBP": float(np.sqrt(np.mean(err ** 2))),
        "MAE_GBP": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(ape) * 100),
        "MdAPE_%": float(np.median(ape) * 100),
        "RMSE_log": float(np.sqrt(np.mean((y_true_log - pred_log) ** 2))),
        "median_bias_GBP": float(np.median(err)),
    }


results = {}

# OLS
Xtr_c = sm.add_constant(Xtr, has_constant="add").astype(float)
Xte_c = sm.add_constant(Xte, has_constant="add").astype(float)
ols = sm.OLS(ytr_log, Xtr_c).fit()
results["OLS"] = metrics(yte_gbp, ols.predict(Xte_c), yte_log)
print(f"OLS      test R2 (log scale): {1 - np.var(yte_log - ols.predict(Xte_c)) / np.var(yte_log):.3f}")

# Ridge (standardised features, alpha chosen by cross-validation)
scaler = StandardScaler().fit(Xtr)
ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(scaler.transform(Xtr), ytr_log)
results[f"Ridge (alpha={ridge.alpha_:.3g})"] = metrics(
    yte_gbp, ridge.predict(scaler.transform(Xte)), yte_log)
print(f"Ridge CV-chosen alpha: {ridge.alpha_:.4g}")

comp = pd.DataFrame(results).T.round(2)
comp.to_csv("output/prediction_comparison.csv")
print("\nTest-set performance (GBP scale):")
print(comp.to_string())
print("\nSaved to output/prediction_comparison.csv")
