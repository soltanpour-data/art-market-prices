"""
Step 1 — Data cleaning.

- Keep sold lots with a valid, positive hammer price (bought-in / withdrawn /
  unknown lots are censored observations — disclosed as a limitation).
- Primary artist: first attribution listed (catalogues list the principal
  attribution first); "[Anonymous]" kept as its own category.
- Subject matter: rule-based classification of the catalogue title into
  market segments (portrait, landscape, religious, ...). Keyword lists were
  built by inspecting the titles; priority order resolves multi-matches.
- Prices are in nominal decimal pounds (guineas converted). No historical
  deflator is applied — see the limitations section in README.md.
- Log-transform prices (art prices are log-normal) and lightly winsorize.

Output: data/clean_london_auctions.csv  (gitignored)
"""

import re

import numpy as np
import pandas as pd

# --- subject classification -------------------------------------------------
# (keyword, segment) in priority order; first match wins.
SUBJECT_RULES = [
    (r"holy|madon|maria\b|saint|saviour|christ|virgin|crucifix|adoration|"
     r"magi\b|apostle|martyr|annunciation|susanna|judith|magdalen", "religious"),
    (r"venus|apollo|diana|nymph|scipio|caesar|alexander|dido|cleopatra|"
     r"bacchus|jupiter|mercury|mytholog", "mythological"),
    (r"portrait|effig|\bhead\b|\bheads\b", "portrait"),
    (r"landscape|seascape|\bview\b|\bviews\b|sea piece|shipping|prospect|"
     r"waterfall", "landscape"),
    (r"flower|fruit|dead game|breakfast|still life|shell\b|shells\b",
     "still_life"),
    (r"cattle|horse|horses|dog\b|dogs\b|sheep|deer|stag\b", "animal"),
    (r"interior|tavern|kitchen|conversation|boor|merrymak|farrier|peasant|"
     r"card\b|cards\b|wake\b|fair\b", "genre"),
]
COMPILED = [(re.compile(pat), seg) for pat, seg in SUBJECT_RULES]


def classify_subject(title: str) -> str:
    s = title.lower()
    for pat, seg in COMPILED:
        if pat.search(s):
            return seg
    return "other"


def primary_artist(artists: str) -> str:
    """First attribution listed, with catalogue brackets stripped."""
    first = str(artists).split(";")[0].strip()
    return first.strip("[]").strip() or "Unknown"


# --- load & filter -----------------------------------------------------------
df = pd.read_csv("data/raw_london_auctions.csv", low_memory=False)
print(f"Raw transactions: {len(df):,}")

sold = df["transaction_type"] == "sold"
has_price = df["transaction_amt"].notna() & (df["transaction_amt"] > 0)
print(f"Dropped non-sold lots: {(~sold).sum():,}")
print(f"Dropped sold lots with missing/zero price: {(sold & ~has_price).sum():,}")
df = df[sold & has_price].copy()
print(f"Lots kept: {len(df):,}")

# --- build analysis columns --------------------------------------------------
df["artist"] = df["artists"].apply(primary_artist)
df["subject"] = df["title"].fillna("").apply(classify_subject)
df["sale_year"] = df["year"].astype(int)
df["sale_month"] = df["month"].astype(int)
df["price_gbp"] = df["transaction_amt"].astype(float)
df["log_price"] = np.log(df["price_gbp"])

# Light winsorization against data-entry errors
lo, hi = df["log_price"].quantile([0.005, 0.995])
n_win = ((df["log_price"] < lo) | (df["log_price"] > hi)).sum()
df["log_price"] = df["log_price"].clip(lo, hi)
print(f"Winsorized log-price outliers: {n_win:,}")

keep = ["artist", "subject", "auction_house", "sale_year", "sale_month",
        "title", "price_gbp", "log_price"]
df[keep].to_csv("data/clean_london_auctions.csv", index=False)

print(f"\nPeriod: {df['sale_year'].min()}-{df['sale_year'].max()}")
print(f"Median price: £{df['price_gbp'].median():.2f}")
print("\nSubject mix:")
print(df["subject"].value_counts().to_string())
print("\nSaved to data/clean_london_auctions.csv (gitignored)")
