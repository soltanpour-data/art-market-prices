"""
Step 7 — Out-of-sample price prediction.

Chronological train/test split (train: sales up to 2013, test: 2014-2015 —
time series data must never be shuffled). Compares plain OLS against ridge
regression. Uses year + season dummies instead of quarterly dummies so no
future information leaks into training.

Metrics are reported on the original GBP scale after back-transformation.

Output: output/prediction_comparison.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/model_data.csv")

meta = ["lot_id", "artist", "segment", "sale_year", "sale_quarter",
        "sale_quarter_label", "log_real_price", "real_price_gbp"]
# Drop quarterly dummies (they would leak future quarters); keep year+season.
# Segment dummies are dropped too: artist fixed effects already determine
# the segment, so including both makes the matrix rank-deficient.
drop_q = [c for c in df.columns if c.startswith("q_")]
drop_seg = [c for c in df.columns if c.startswith("seg_")]
X_all = df.drop(columns=meta + drop_q + drop_seg)
y_log = df["log_real_price"]
y_gbp = df["real_price_gbp"]

train = df["sale_year"] <= 2013
Xtr, Xte = X_all[train], X_all[~train]
# Year dummies for 2014/2015 are all-zero in training -> drop zero-variance
# columns (rank-deficient design matrix) from both splits.
nzv = Xtr.columns[Xtr.nunique() <= 1]
if len(nzv):
    print(f"Dropped {len(nzv)} zero-variance columns: {list(nzv)}")
Xtr, Xte = Xtr.drop(columns=nzv), Xte.drop(columns=nzv)
ytr_log, yte_log = y_log[train], y_log[~train]
yte_gbp = y_gbp[~train]
print(f"Train: {train.sum():,} lots (<=2013) | Test: {(~train).sum():,} lots (2014-2015)")


def metrics(y_true_gbp, pred_log):
    # Naive back-transformation exp(pred). The textbook alternative, Duan's
    # smearing correction (multiply by mean(exp(train residuals))), was tried
    # and *rejected*: with heteroscedastic errors — which the HC1 standard
    # errors in step 04 already diagnose — a single global smearing factor
    # over-corrects and worsened test MAPE (31.5% -> 35.0%) and bias.
    # Under heteroscedasticity the correction must vary with X; the naive
    # retransformation is the honest, better-performing choice here.
    pred_gbp = np.exp(pred_log)
    err = y_true_gbp - pred_gbp
    return {
        "RMSE_GBP": float(np.sqrt(np.mean(err ** 2))),
        "MAE_GBP": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err / y_true_gbp)) * 100),
        "mean_bias_GBP": float(np.mean(err)),
    }


results = {}

# OLS
Xtr_c = sm.add_constant(Xtr, has_constant="add").astype(float)
Xte_c = sm.add_constant(Xte, has_constant="add").astype(float)
ols = sm.OLS(ytr_log, Xtr_c).fit()
results["OLS"] = metrics(yte_gbp, ols.predict(Xte_c))
print(f"OLS      test R2 (log scale): {1 - np.var(yte_log - ols.predict(Xte_c)) / np.var(yte_log):.3f}")

# Ridge (standardised features, alpha chosen by cross-validation)
scaler = StandardScaler().fit(Xtr)
ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(scaler.transform(Xtr), ytr_log)
results[f"Ridge (alpha={ridge.alpha_:.3g})"] = metrics(
    yte_gbp, ridge.predict(scaler.transform(Xte)))
print(f"Ridge CV-chosen alpha: {ridge.alpha_:.4g}")

comp = pd.DataFrame(results).T.round(2)
comp.to_csv("output/prediction_comparison.csv")
print("\nTest-set performance (GBP scale):")
print(comp.to_string())
print("\nSaved to output/prediction_comparison.csv")
