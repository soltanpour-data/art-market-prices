"""
Step 8 — Explaining price differences across market segments.

1. Fits the hedonic model separately within each segment and compares
   which characteristics drive prices in each.
2. Formally tests whether segment effects differ with an F-test:
   restricted (pooled) model vs unrestricted model with segment
   interactions on the key price drivers.

Output: output/segment_comparison.txt
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/model_data.csv")

meta = ["lot_id", "artist", "segment", "sale_year", "sale_quarter",
        "sale_quarter_label", "log_real_price", "real_price_gbp"]
drop_q = [c for c in df.columns if c.startswith("q_")]
base_cols = [c for c in df.columns if c not in meta + drop_q
             and not c.startswith("seg_")]
# Note: segment dummies are excluded because we fit within segments.

lines = []
key_vars = (["log_area", "signed", "provenance_count", "exhibition_count"]
            + [c for c in base_cols if c.startswith("house_")])

for seg in df["segment"].unique():
    d = df[df["segment"] == seg]
    cols = [c for c in base_cols if d[c].nunique() > 1]
    # Within one segment the artist dummies sum to 1 (every lot is by one of
    # the segment's artists), so drop one artist dummy to keep the constant.
    artist_cols = sorted(c for c in cols if c.startswith("artist_"))
    if artist_cols:
        cols = [c for c in cols if c != artist_cols[0]]
    X = sm.add_constant(d[cols], has_constant="add").astype(float)
    fit = sm.OLS(d["log_real_price"], X).fit()
    lines.append(f"--- {seg} (n={len(d):,}) ---")
    lines.append(f"R-squared: {fit.rsquared:.3f}")
    for v in key_vars:
        if v in fit.params.index:
            lines.append(f"  {v:22s} coef={fit.params[v]:+.3f}  "
                         f"p={fit.pvalues[v]:.3g}")
    lines.append("")

# F-test: pooled vs segment-interacted model.
# Segment *main* effects are omitted: artist fixed effects already pin down
# the segment, so only the slope interactions are identified. The F-test is
# then exactly H0: "price drivers work the same in every segment".
X_base = sm.add_constant(df[base_cols], has_constant="add").astype(float)
restricted = sm.OLS(df["log_real_price"], X_base).fit()

seg_dummies = pd.get_dummies(df["segment"], prefix="SEG", dtype=float)
inter = pd.DataFrame(index=df.index)
for s in seg_dummies.columns[1:]:          # interact with 2 of 3 segments
    for v in key_vars:
        if v in df.columns:
            inter[f"{s}x{v}"] = seg_dummies[s] * df[v]
X_unr = pd.concat([X_base, inter], axis=1).astype(float)
unrestricted = sm.OLS(df["log_real_price"], X_unr).fit()

F, p, _ = unrestricted.compare_f_test(restricted)
lines.append("F-test: pooled model vs segment-interacted model")
lines.append(f"F = {F:.2f}, p-value = {p:.3g}")
lines.append("-> " + ("Segment differences are statistically significant."
                      if p < 0.05 else
                      "No significant segment differences detected."))

open("output/segment_comparison.txt", "w").write("\n".join(lines))
print("\n".join(lines))
print("\nSaved to output/segment_comparison.txt")
