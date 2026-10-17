"""
Step 3 — Feature engineering.

- Artists: individual dummies for the 40 most-traded artists (everyone else
  -> "Other"). With ~137k lots, 40 artist fixed effects are easily estimable
  and capture the dominant price driver; artist identity also absorbs
  school/period differences.
- Auction houses: dummies for the 10 largest houses ("Other" baseline).
- Subject: dummies for all 8 classified segments ("other" baseline).
- Time: annual dummies for the hedonic price index (baseline 1780), plus a
  numeric year trend and month dummies for the prediction exercise, where
  future years must not leak into training.

Output: data/model_data.csv  (gitignored)
"""

import pandas as pd

df = pd.read_csv("data/clean_london_auctions.csv")

# --- artists: top 40 get fixed effects --------------------------------------
top_artists = df["artist"].value_counts().head(40).index.tolist()
df["artist_top"] = df["artist"].where(df["artist"].isin(top_artists), "Other")
df["artist_top"] = pd.Categorical(df["artist_top"],
                                  categories=["Other"] + sorted(top_artists))

# --- houses: top 10 -----------------------------------------------------------
top_houses = df["auction_house"].value_counts().head(10).index.tolist()
df["house_top"] = df["auction_house"].where(df["auction_house"].isin(top_houses),
                                            "Other")
df["house_top"] = pd.Categorical(df["house_top"],
                                 categories=["Other"] + sorted(top_houses))

# --- subjects -----------------------------------------------------------------
subjects = sorted(df["subject"].unique())
df["subject"] = pd.Categorical(df["subject"],
                               categories=["other"] + [s for s in subjects
                                                       if s != "other"])

dummies = pd.get_dummies(
    df[["artist_top", "house_top", "subject"]],
    prefix=["artist", "house", "subj"],
    drop_first=True, dtype=float)

# Annual dummies for the hedonic index (baseline 1780)
y_dummies = pd.get_dummies(df["sale_year"].astype(str), prefix="y", dtype=float)
y_dummies = y_dummies[[c for c in sorted(y_dummies.columns) if c != "y_1780"]]

# Month dummies for seasonality (baseline January)
m_dummies = pd.get_dummies(df["sale_month"].astype(str), prefix="m", dtype=float)
m_dummies = m_dummies[[c for c in sorted(m_dummies.columns,
                                        key=lambda x: int(x[2:]))
                       if c != "m_1"]]

model = pd.concat([
    df[["artist", "subject", "auction_house", "sale_year", "sale_month",
        "log_price", "price_gbp"]].assign(year_trend=df["sale_year"] - 1780),
    dummies, y_dummies, m_dummies,
], axis=1)

model.to_csv("data/model_data.csv", index=False)
n_feat = model.shape[1] - 8
print(f"Design matrix: {model.shape[0]:,} rows x {n_feat} features (+ 8 id/target cols).")
print(f"Artist dummies: {sum(c.startswith('artist_') for c in dummies.columns)}, "
      f"house: {sum(c.startswith('house_') for c in dummies.columns)}, "
      f"subject: {sum(c.startswith('subj_') for c in dummies.columns)}, "
      f"years: {y_dummies.shape[1]}, months: {m_dummies.shape[1]}")
print("\nSaved to data/model_data.csv (gitignored)")
