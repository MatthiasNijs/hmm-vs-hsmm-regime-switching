"""Build daily momentum and mean-reversion sleeve returns from CRSP.

Writes data/sleeve_returns.csv. Requires the CRSP extract described in
data/README.md; the committed sleeve_returns.csv lets every later step run
without it.
"""

import pandas as pd

from mean_reversion_sleeve import mean_reversion_returns
from metrics import compute_perf_metrics, print_perf_table
from momentum_sleeve import momentum_returns
from paths import CRSP_CSV, DATA


def main():
    if not CRSP_CSV.exists():
        raise FileNotFoundError(f"{CRSP_CSV} not found. See data/README.md for the WRDS extract.")

    mom = momentum_returns(CRSP_CSV).rename("mom_ret")
    mr = mean_reversion_returns(CRSP_CSV).rename("mr_ret")

    # momentum needs 12 months of history, so the joint sample starts in 2006
    sleeves = pd.concat([mom, mr], axis=1, join="inner")
    sleeves.index.name = "date"
    sleeves.to_csv(DATA / "sleeve_returns.csv")

    print(f"Saved {len(sleeves)} days to data/sleeve_returns.csv")
    print_perf_table([
        compute_perf_metrics(sleeves["mom_ret"], "Momentum"),
        compute_perf_metrics(sleeves["mr_ret"], "Mean-reversion"),
    ])


if __name__ == "__main__":
    main()
