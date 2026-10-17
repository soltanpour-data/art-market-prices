"""
Step 5 — Quality-adjusted art price index.

The annual time dummies from the hedonic regression capture the pure time
effect on prices, holding artist, house, subject and seasonality constant.
Turning them into an index gives the series analysed in step 6.

    Index_t = 100 * exp(beta_t - beta_base),  base year 1780 = 100

Output: data/price_index.csv, output/fig_price_index.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

coefs = pd.read_csv("output/hedonic_coefficients.csv")
y = coefs[coefs["variable"].str.startswith("y_")].copy()
y["year"] = y["variable"].str.replace("y_", "", regex=False).astype(int)
y = y.sort_values("year").reset_index(drop=True)

base = pd.DataFrame({"variable": ["(baseline)"], "year": [1780], "coef": [0.0]})
y = pd.concat([base, y], ignore_index=True)
y["index"] = 100 * np.exp(y["coef"])

y[["year", "index"]].to_csv("data/price_index.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(y["year"], y["index"], marker="o", ms=3)
ax.axhline(100, color="red", ls="--", lw=1)
ax.set_title("Hedonic art price index, London 1780-1835 (1780 = 100)")
ax.set_xlabel("Year"); ax.set_ylabel("Index (nominal £)")
fig.tight_layout(); fig.savefig("output/fig_price_index.png"); plt.close(fig)

print(f"Index 1780: 100.0 -> 1835: {y['index'].iloc[-1]:.1f}")
print(f"Total nominal change: {y['index'].iloc[-1] - 100:+.1f} index points")
print("Note: nominal pounds — no historical deflator applied (see README).")
print("\nSaved to data/price_index.csv and output/fig_price_index.png")
