"""
Step 1 — Data cleaning.

- Parse dates, derive sale_quarter labels ("2005Q1").
- Drop unsold lots (censored observations — flagged in the thesis limitations).
- Deflate hammer prices to constant 2005 GBP with the UK CPI.
- Build log_real_price (art prices are log-normal) and artwork area.
- Winsorize log prices at the 0.5 / 99.5 percentiles.

Output: data/clean_auction_data.csv
"""

import numpy as np
import pandas as pd

# UK CPI, annual average, 2005 = 100 (ONS, rounded)
CPI = {2005: 100.0, 2006: 102.3, 2007: 104.7, 2008: 108.5, 2009: 110.8,
       2010: 114.5, 2011: 119.6, 2012: 123.0, 2013: 126.1, 2014: 128.0,
       2015: 128.0}

df = pd.read_csv("data/raw_auction_data.csv")
print(f"Raw lots: {len(df):,}")

unsold = (~df["sold"]).sum()
print(f"Unsold lots dropped (censored data): {unsold:,}")
df = df[df["sold"]].copy()

df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_quarter_label"] = (df["sale_year"].astype(str) + "Q"
                            + df["sale_quarter"].astype(str))

# Deflate to constant 2005 pounds
df["cpi"] = df["sale_year"].map(CPI)
df["real_price_gbp"] = df["hammer_price_gbp"] * 100.0 / df["cpi"]
df["log_real_price"] = np.log(df["real_price_gbp"])
df["area_cm2"] = df["width_cm"] * df["height_cm"]
df["log_area"] = np.log(df["area_cm2"])

# Winsorize extreme log prices (data-entry errors / trophy lots)
lo, hi = df["log_real_price"].quantile([0.005, 0.995])
n_win = ((df["log_real_price"] < lo) | (df["log_real_price"] > hi)).sum()
df["log_real_price"] = df["log_real_price"].clip(lo, hi)
print(f"Winsorized log-price outliers: {n_win:,}")

df["real_price_gbp"] = np.exp(df["log_real_price"])  # keep consistent

keep = ["lot_id", "artist", "artist_tier", "segment", "auction_house",
        "sale_date", "sale_year", "sale_quarter", "sale_quarter_label",
        "medium", "width_cm", "height_cm", "area_cm2", "log_area",
        "signed", "provenance_count", "exhibition_count",
        "hammer_price_gbp", "real_price_gbp", "log_real_price"]
df[keep].to_csv("data/clean_auction_data.csv", index=False)

print(f"\nClean lots: {len(df):,}")
print(f"Period: {df['sale_quarter_label'].min()} to {df['sale_quarter_label'].max()}")
print(f"Median real price: £{df['real_price_gbp'].median():,.0f}")
print("\nSaved to data/clean_auction_data.csv")
