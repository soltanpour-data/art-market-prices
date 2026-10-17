"""
Step 2 — Exploratory data analysis.

Descriptive statistics and first plots for the historical data: price
distribution, prices by auction house / subject, lots per year, and the
most-traded artists.

Output: output/fig_*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

df = pd.read_csv("data/clean_london_auctions.csv")

print("=== Descriptive statistics (nominal GBP) ===")
print(df["price_gbp"].describe().to_string())
print("\nLots by auction house (top 10):\n",
      df["auction_house"].value_counts().head(10).to_string())
print("\nLots by subject:\n", df["subject"].value_counts().to_string())
print("\nMost-traded artists:\n", df["artist"].value_counts().head(10).to_string())

# 1. Distribution of log prices
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["log_price"], bins=60, edgecolor="white")
ax.set_title("Distribution of log prices, London 1780-1835")
ax.set_xlabel("log(price in £)")
fig.tight_layout(); fig.savefig("output/fig_price_distribution.png")
plt.close(fig)

# 2. Prices by auction house (top 8)
top_houses = df["auction_house"].value_counts().head(8).index
fig, ax = plt.subplots(figsize=(9, 4.5))
(df[df["auction_house"].isin(top_houses)]
   .boxplot(column="log_price", by="auction_house", ax=ax, showfliers=False))
ax.set_title("Log price by auction house")
ax.set_xlabel(""); plt.suptitle("")
fig.tight_layout(); fig.savefig("output/fig_price_by_house.png")
plt.close(fig)

# 3. Prices by subject segment
order = df["subject"].value_counts().index
fig, ax = plt.subplots(figsize=(9, 4.5))
(df.assign(subject=pd.Categorical(df["subject"], categories=order,
                                  ordered=True)).sort_values("subject")
   .boxplot(column="log_price", by="subject", ax=ax, showfliers=False))
ax.set_title("Log price by subject segment")
ax.set_xlabel(""); plt.suptitle("")
fig.tight_layout(); fig.savefig("output/fig_price_by_subject.png")
plt.close(fig)

# 4. Lots per year (market activity over 1780-1835)
per_year = df["sale_year"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(per_year.index, per_year.values, width=0.9)
ax.set_title("Lots sold per year, London 1780-1835")
ax.set_xlabel("Year"); ax.set_ylabel("Lots")
fig.tight_layout(); fig.savefig("output/fig_lots_per_year.png")
plt.close(fig)

# 5. Median log price per year (naive index, before hedonic adjustment)
med = df.groupby("sale_year")["log_price"].median()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(med.index, med.values, marker="o", ms=3)
ax.set_title("Annual median log price (naive index)")
ax.set_xlabel("Year")
fig.tight_layout(); fig.savefig("output/fig_annual_median.png")
plt.close(fig)

print("\nSaved 5 figures to output/")
