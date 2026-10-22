# Data

| File | Source | In repo |
|---|---|---|
| `spy_features.csv` | Built by `src/features.py` from SPY daily OHLC (Yahoo Finance), 2002-01-31 to 2024-12-30 | yes |
| `sleeve_returns.csv` | Built by `src/build_sleeves.py` from CRSP; daily returns of the two sleeves, 2006-01-03 to 2024-12-31 | yes |
| `sleeve_turnover.csv` | Built by `src/build_sleeves.py` from the sleeves' daily holdings; one-way turnover for each sleeve-to-sleeve transition | yes |
| `crsp_sp500_daily.csv` | CRSP daily stock file via WRDS | **no**, licensed |

## CRSP extract

The sleeves are built from CRSP daily data for S&P 500 constituents (CRSP index
`INDNO = 1000500`), with each stock included only inside its index membership window
(`MbrStartDt` to `MbrEndDt`). Using point-in-time membership keeps the universe free of
survivorship bias. Sample: 2005-01-03 to 2024-12-31, about 2.5 million stock-days and
roughly 500 names per day.

Columns used: `PERMNO`, `DlyCalDt`, `DlyRetx` (falls back to `DlyRet` if absent).

CRSP data cannot be redistributed. With WRDS access, export the extract above to
`data/crsp_sp500_daily.csv` to rebuild `sleeve_returns.csv`. Every other step runs
from the committed files.
