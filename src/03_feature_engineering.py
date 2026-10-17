"""
Step 3 — Feature engineering.

Turns the cleaned data into a modelling-ready design matrix:
- Categorical variables -> dummy variables with explicit baselines.
- Artists: individual dummies for the 10 most-traded artists; everyone
  else grouped into artist-tier dummies (avoids hundreds of dummies).
- Time: quarterly dummies for the hedonic price index (script 05),
  plus year + season dummies for the prediction exercise (script 07),
  where future quarters must not leak into training.

Output: data/model_data.csv
"""

import pandas as pd

df = pd.read_csv("data/clean_auction_data.csv")

# Explicit baselines (dropped categories)
df["auction_house"] = pd.Categorical(df["auction_house"],
                                    categories=["Bonhams", "Christie's", "Sotheby's"])
df["segment"] = pd.Categorical(df["segment"],
                               categories=["Old Masters", "Impressionist & Modern",
                                           "Post-War & Contemporary"])
df["medium"] = pd.Categorical(df["medium"],
                              categories=["Works on paper", "Oil on canvas",
                                          "Oil on panel", "Watercolor", "Sculpture"])
df["artist_tier"] = pd.Categorical(df["artist_tier"],
                                   categories=["emerging", "mid_career", "blue_chip"])
# (artist_tier is kept as a descriptive column; the model uses full artist
# fixed effects instead, which nest the tier information.)

# Artists: individual fixed-effect dummies for every artist (24 artists —
# easily estimable; artist identity is the single biggest price driver).
df["artist"] = pd.Categorical(df["artist"], categories=sorted(df["artist"].unique()))

dummies = pd.get_dummies(
    df[["auction_house", "segment", "medium", "artist"]],
    prefix=["house", "seg", "med", "artist"],
    drop_first=True, dtype=float)

# Quarterly dummies for the hedonic index (baseline: 2005Q1)
quarters = sorted(df["sale_quarter_label"].unique())
q_dummies = pd.get_dummies(df["sale_quarter_label"], prefix="q", dtype=float)
q_dummies = q_dummies[[c for c in q_dummies.columns if c != "q_2005Q1"]]

# Year + season dummies for prediction (baseline: 2005, season 1)
df["season"] = df["sale_quarter"]
y_dummies = pd.get_dummies(df["sale_year"].astype(str), prefix="y", dtype=float)
y_dummies = y_dummies[[c for c in y_dummies.columns if c != "y_2005"]]
s_dummies = pd.get_dummies(df["season"].astype(str), prefix="s", dtype=float)
s_dummies = s_dummies[[c for c in s_dummies.columns if c != "s_1"]]

model = pd.concat([
    df[["lot_id", "artist", "segment", "sale_year", "sale_quarter",
        "sale_quarter_label", "log_real_price", "real_price_gbp",
        "log_area", "signed", "provenance_count", "exhibition_count"]],
    dummies, q_dummies, y_dummies, s_dummies,
], axis=1)
model["signed"] = model["signed"].astype(float)

model.to_csv("data/model_data.csv", index=False)
n_feat = model.shape[1] - 12
print(f"Design matrix: {model.shape[0]:,} rows x {n_feat} features "
      f"(+ 12 id/target columns).")
print(f"Quarter dummies: {q_dummies.shape[1]}, "
      f"year dummies: {y_dummies.shape[1]}, season dummies: {s_dummies.shape[1]}")
print("\nSaved to data/model_data.csv")
