"""
Run the full pipeline, steps 00-08, in order.
Usage: python src/run_all.py   (from the project root)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "00_generate_sample_data.py",
    "01_data_cleaning.py",
    "02_eda.py",
    "03_feature_engineering.py",
    "04_hedonic_regression.py",
    "05_price_index.py",
    "06_time_series.py",
    "07_prediction.py",
    "08_segment_comparison.py",
]

for script in SCRIPTS:
    print(f"\n{'='*70}\n>>> {script}\n{'='*70}")
    r = subprocess.run([sys.executable, str(ROOT / "src" / script)],
                       cwd=ROOT)
    if r.returncode != 0:
        print(f"\nFAILED at {script} — stopping.")
        sys.exit(1)

print(f"\n{'='*70}\nPipeline complete. See output/ for results.\n{'='*70}")
