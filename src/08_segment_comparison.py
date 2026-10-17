"""
Step 8 — Explaining price differences across market segments.

1. Fits the hedonic model separately within each subject segment and
   compares which characteristics drive prices in each.
2. Formally tests whether segment effects differ with an F-test:
   pooled model vs a model with segment interactions on the key price
   drivers (auction-house premia). Artists are not nested in subjects, so
   the interactions are identified.

Output: output/segment_comparison.txt
"""

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/model_data.csv")

meta = ["artist", "subject", "auction_house", "sale_year", "sale_month",
        "log_price", "price_gbp", "year_trend"]
drop_y = [c for c in df.columns if c.startswith("y_")]
base_cols = [c for c in df.columns if c not in meta + drop_y
             and not c.startswith("subj_")]

lines = []
key_vars = [c for c in base_cols if c.startswith("house_")]

for subj in sorted(df["subject"].unique()):
    d = df[df["subject"] == subj]
    cols = [c for c in base_cols if d[c].nunique() > 1]
    # Within one subject the artist dummies need no extra handling here
    # (artists painted many subjects, so no sum-to-one issue).
    X = sm.add_constant(d[cols], has_constant="add").astype(float)
    fit = sm.OLS(d["log_price"], X).fit()
    lines.append(f"--- {subj} (n={len(d):,}) ---")
    lines.append(f"R-squared: {fit.rsquared:.3f}")
    for v in key_vars:
        if v in fit.params.index:
            lines.append(f"  {v:28s} coef={fit.params[v]:+.3f}  "
                         f"p={fit.pvalues[v]:.3g}")
    lines.append("")

# F-test: pooled vs segment-interacted (house premia x subject)
X_base = sm.add_constant(df[base_cols], has_constant="add").astype(float)
restricted = sm.OLS(df["log_price"], X_base).fit()

seg_dummies = pd.get_dummies(df["subject"], prefix="SEG", dtype=float)
inter = pd.DataFrame(index=df.index)
for s in seg_dummies.columns[1:]:
    for v in key_vars:
        inter[f"{s}x{v}"] = seg_dummies[s] * df[v]
X_unr = pd.concat([X_base, inter], axis=1).astype(float)
unrestricted = sm.OLS(df["log_price"], X_unr).fit()

F, p, _ = unrestricted.compare_f_test(restricted)
lines.append("F-test: pooled model vs segment-interacted model")
lines.append(f"F = {F:.2f}, p-value = {p:.3g}")
lines.append("-> " + ("Auction-house premia differ significantly across "
                      "subject segments." if p < 0.05 else
                      "No significant segment differences detected."))

open("output/segment_comparison.txt", "w").write("\n".join(lines))
print("\n".join(lines))
print("\nSaved to output/segment_comparison.txt")
