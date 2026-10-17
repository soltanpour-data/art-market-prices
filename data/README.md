# Data

## Source

Real data — no synthetic stand-ins. The pipeline uses the `sales` data set
from the public research package
**[mdlincoln/londonauctions](https://github.com/mdlincoln/londonauctions)**:

> Matthew Lincoln & Abram Fox (2016). "The Temporal Dimensions of the
> London Art Auction, 1780-1835." *British Art Studies* 4.
> Data originally from the **Getty Provenance Index** (British sales
> catalogues). Archived on Zenodo: DOI 10.5281/zenodo.60284.

Run `python src/00_fetch_data.py` to download `sales.rda` (~5 MB) and
convert it to `data/raw_london_auctions.csv`. All CSVs in this folder are
**gitignored** — they are regenerated locally, not committed.

## Files (generated locally)

| File | Contents |
|---|---|
| `sales.rda` | Raw download (R format) |
| `raw_london_auctions.csv` | 304,788 transactions, 21 columns |
| `clean_london_auctions.csv` | 137,186 sold lots with valid prices + parsed artist/subject |
| `model_data.csv` | Modelling-ready design matrix (137,186 × 123 features) |
| `price_index.csv` | Hedonic price index, 1780 = 100 |

## Raw schema (selected columns)

| Column | Meaning |
|---|---|
| `cat_no`, `lot_no` | Getty catalogue / lot identifiers |
| `auction_house` | e.g. Christie's, Foster (Edward), Phillips (Harry) |
| `date`, `year`, `month` | Sale date |
| `transaction_type` | sold / bought in / withdrawn / unknown / ... |
| `transaction_amt` | Hammer price in decimal pounds (guineas converted); NaN if unknown |
| `artists` | Attribution(s), `;`-separated; `[Anonymous]` if unattributed |
| `title` | Catalogue title/description (used for subject classification) |
| `sellers`, `buyers` | As transcribed (not used) |

## Cleaning rules (see `src/01_data_cleaning.py`)

- Keep `transaction_type == "sold"` with `transaction_amt > 0`.
- Primary artist = first attribution listed, brackets stripped.
- Subject = keyword rule on `title` → religious / mythological / portrait /
  landscape / still_life / animal / genre / other.
- `log_price = ln(transaction_amt)`, winsorized at 0.5/99.5%.
- Prices are **nominal pounds** (no historical deflator applied).
