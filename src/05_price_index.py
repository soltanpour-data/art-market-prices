"""
Step 5 — Quality-adjusted art price index.

The quarterly time dummies from the hedonic regression capture the pure
time effect on prices, holding artwork characteristics constant. Turning
them into an index gives the series analysed in step 6.

    Index_t = 100 * exp(beta_t - beta_base),  base quarter 2005Q1 = 100

Output: data/price_index.csv, output/fig_price_index.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

coefs = pd.read_csv("output/hedonic_coefficients.csv")
q = coefs[coefs["variable"].str.startswith("q_")].copy()
q["quarter"] = q["variable"].str.replace("q_", "", regex=False)
q = q.sort_values("quarter").reset_index(drop=True)

# Baseline quarter 2005Q1 has an implicit coefficient of 0
base = pd.DataFrame({"variable": ["(baseline)"], "quarter": ["2005Q1"],
                     "coef": [0.0]})
q = pd.concat([base, q], ignore_index=True)
q["index"] = 100 * np.exp(q["coef"])

q[["quarter", "index"]].to_csv("data/price_index.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(q["quarter"], q["index"], marker="o", ms=3)
ax.axhline(100, color="red", ls="--", lw=1)
ax.set_title("Hedonic art price index (2005Q1 = 100)")
ax.set_xlabel("Quarter"); ax.set_ylabel("Index")
ax.tick_params(axis="x", rotation=90, labelsize=7)
fig.tight_layout(); fig.savefig("output/fig_price_index.png"); plt.close(fig)

print(f"Index 2005Q1: 100.0 -> 2015Q4: {q['index'].iloc[-1]:.1f}")
print(f"Total real appreciation: {q['index'].iloc[-1] - 100:+.1f} index points")
print("\nSaved to data/price_index.csv and output/fig_price_index.png")
