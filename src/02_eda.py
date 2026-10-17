"""
Step 2 — Exploratory data analysis.

Descriptive statistics and the standard first plots for an art-market
dataset: price distribution, prices by auction house / segment / medium,
and the quarterly median-price path.

Output: output/fig_*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

df = pd.read_csv("data/clean_auction_data.csv")

print("=== Descriptive statistics (real GBP) ===")
print(df["real_price_gbp"].describe().to_string())
print("\nLots by auction house:\n", df["auction_house"].value_counts().to_string())
print("\nLots by segment:\n", df["segment"].value_counts().to_string())
print("\nLots by medium:\n", df["medium"].value_counts().to_string())
print("\nMissing values per column:\n",
      df.isna().sum()[df.isna().sum() > 0].to_string() or "none")

# 1. Distribution of log prices (should look roughly normal)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["log_real_price"], bins=40, edgecolor="white")
ax.set_title("Distribution of log real prices (2005 GBP)")
ax.set_xlabel("log(price)")
fig.tight_layout(); fig.savefig("output/fig_price_distribution.png")
plt.close(fig)

# 2. Prices by auction house
fig, ax = plt.subplots(figsize=(7, 4.5))
df.boxplot(column="log_real_price", by="auction_house", ax=ax, showfliers=False)
ax.set_title("Log real price by auction house")
ax.set_xlabel(""); plt.suptitle("")
fig.tight_layout(); fig.savefig("output/fig_price_by_house.png")
plt.close(fig)

# 3. Prices by market segment
fig, ax = plt.subplots(figsize=(8, 4.5))
df.boxplot(column="log_real_price", by="segment", ax=ax, showfliers=False)
ax.set_title("Log real price by market segment")
ax.set_xlabel(""); plt.suptitle("")
fig.tight_layout(); fig.savefig("output/fig_price_by_segment.png")
plt.close(fig)

# 4. Quarterly median price path (a first, naive index)
med = (df.groupby("sale_quarter_label")["log_real_price"].median()
         .sort_index())
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(med.index, med.values, marker="o", ms=3)
ax.set_title("Quarterly median log price (naive index)")
ax.set_xlabel("Quarter"); ax.tick_params(axis="x", rotation=90, labelsize=7)
fig.tight_layout(); fig.savefig("output/fig_quarterly_median.png")
plt.close(fig)

print("\nSaved 4 figures to output/")
