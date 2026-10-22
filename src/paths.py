from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# CRSP daily stock file (not redistributable, see data/README.md)
CRSP_CSV = DATA / "crsp_sp500_daily.csv"
