"""
Step 0 — Generate a synthetic London painting-auction dataset (2005-2015).

The figures below mimic the structure of real auction results (Christie's,
Sotheby's, Bonhams) so the whole pipeline runs end-to-end without access to
proprietary data. Replace `data/raw_auction_data.csv` with real lot-level
results to run the analysis on actual data — see data/README.md.

Output: data/raw_auction_data.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_LOTS = 4000

# ---------------------------------------------------------------- artists ---
# Fictional artists, 8 per market segment. The first two of each segment are
# "blue-chip" names with a large positive price effect.
SEGMENTS = {
    "Old Masters": (
        ["Elena Marsh", "Tomas Reber"]                      # blue-chip
        + ["Ingrid Solberg", "Marcus Feld", "Ana Duarte"]    # mid-career
        + ["Piotr Nowak", "Lucia Ferreira", "Omar Haddad"]   # emerging
    ),
    "Impressionist & Modern": (
        ["Sofia Almeida", "David Okafor"]
        + ["Yuki Tanaka", "Clara Weiss", "Rafael Costa"]
        + ["Nadia Rahal", "Elias Berg", "Mara Ionescu"]
    ),
    "Post-War & Contemporary": (
        ["Jonas Lindqvist", "Priya Natarajan"]
        + ["Kenji Sato", "Amara Diallo", "Leo Fischer"]
        + ["Zara Ahmed", "Nils Johansson", "Ines Rocha"]
    ),
}

ARTIST_TIER = {}
ARTIST_EFFECT = {}
for seg, names in SEGMENTS.items():
    for i, name in enumerate(names):
        tier = "blue_chip" if i < 2 else ("mid_career" if i < 5 else "emerging")
        ARTIST_TIER[name] = tier
        base = {"blue_chip": 0.70, "mid_career": 0.10, "emerging": -0.35}[tier]
        ARTIST_EFFECT[name] = base + RNG.normal(0, 0.10)

SEGMENT_EFFECT = {"Old Masters": 0.25, "Impressionist & Modern": 0.15,
                  "Post-War & Contemporary": 0.0}
HOUSE_EFFECT = {"Christie's": 0.08, "Sotheby's": 0.06, "Bonhams": -0.06}
MEDIUM_EFFECT = {"Oil on canvas": 0.12, "Oil on panel": 0.08,
                 "Watercolor": -0.05, "Works on paper": -0.12,
                 "Sculpture": 0.05}
SEASON_EFFECT = {1: -0.02, 2: 0.03, 3: -0.04, 4: 0.05}  # London sale seasons
# A few price drivers genuinely work differently per segment (as in real
# art markets): the size premium is steepest for Contemporary works,
# the signature premium matters most for Old Masters.
SIZE_ELAST = {"Old Masters": 0.28, "Impressionist & Modern": 0.35,
              "Post-War & Contemporary": 0.45}
SIGNED_PREM = {"Old Masters": 0.18, "Impressionist & Modern": 0.12,
               "Post-War & Contemporary": 0.08}

# ------------------------------------------------------------------ lots ---
all_artists = [(seg, a) for seg, names in SEGMENTS.items() for a in names]
seg_choice = RNG.choice(len(SEGMENTS), size=N_LOTS, p=[0.3, 0.35, 0.35])

rows = []
for s in seg_choice:
    seg = list(SEGMENTS)[s]
    artist = RNG.choice(SEGMENTS[seg])
    year = int(RNG.integers(2005, 2016))
    quarter = int(RNG.integers(1, 5))
    month = (quarter - 1) * 3 + int(RNG.integers(1, 4))
    day = int(RNG.integers(1, 29))
    house = RNG.choice(["Christie's", "Sotheby's", "Bonhams"],
                       p=[0.40, 0.40, 0.20])
    medium = RNG.choice(list(MEDIUM_EFFECT),
                        p=[0.45, 0.15, 0.12, 0.20, 0.08])
    width = float(np.exp(RNG.normal(3.9, 0.45)))    # cm, median ~50
    height = float(np.exp(RNG.normal(3.7, 0.45)))   # cm, median ~40
    signed = bool(RNG.random() < 0.70)
    provenance = int(RNG.poisson(1.5))
    exhibitions = int(RNG.poisson(1.0))

    log_price = (
        8.30
        + ARTIST_EFFECT[artist]
        + SEGMENT_EFFECT[seg]
        + HOUSE_EFFECT[house]
        + MEDIUM_EFFECT[medium]
        + SIZE_ELAST[seg] * np.log(width * height)  # size elasticity varies by segment
        + SIGNED_PREM[seg] * signed                 # signature premium varies by segment
        + 0.06 * provenance
        + 0.05 * exhibitions
        + 0.022 * (year - 2005)                  # market trend
        + SEASON_EFFECT[quarter]
        + RNG.normal(0, 0.35)
    )
    sold = bool(RNG.random() < 0.92)              # ~8% bought in (unsold)
    rows.append({
        "lot_id": f"LOT-{len(rows)+1:05d}",
        "artist": artist,
        "artist_tier": ARTIST_TIER[artist],
        "segment": seg,
        "auction_house": house,
        "sale_date": f"{year}-{month:02d}-{day:02d}",
        "sale_year": year,
        "sale_quarter": quarter,
        "medium": medium,
        "width_cm": round(width, 1),
        "height_cm": round(height, 1),
        "signed": signed,
        "provenance_count": provenance,
        "exhibition_count": exhibitions,
        "sold": sold,
        "hammer_price_gbp": round(float(np.exp(log_price)), 2) if sold else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv("data/raw_auction_data.csv", index=False)
print(f"Generated {len(df):,} lots "
      f"({df['sold'].sum():,} sold, {(~df['sold']).sum():,} unsold).")
print(df.head(3).to_string(index=False))
print("\nSaved to data/raw_auction_data.csv")
