"""Build daily momentum and mean-reversion sleeve returns and turnover from CRSP.

Writes data/sleeve_returns.csv and data/sleeve_turnover.csv. Requires the CRSP
extract described in data/README.md; the committed CSVs let every later step
run without it.
"""

import pandas as pd

from mean_reversion_sleeve import mean_reversion_returns
from metrics import compute_perf_metrics, print_perf_table
from momentum_sleeve import momentum_returns
from paths import CRSP_CSV, DATA


def l1_distance(a: pd.Series, b: pd.Series) -> float:
    """Sum of absolute weight changes needed to move from holdings b to holdings a."""
    return float(a.sub(b, fill_value=0.0).abs().sum())


def build_turnover(dates: pd.DatetimeIndex, mom_w: dict, mr_w: dict) -> pd.DataFrame:
    """
    One-way turnover (fraction of capital traded) to hold each day's portfolio, given
    the previous day's holdings, for the four possible sleeve transitions. Weight drift
    within a holding period is ignored, consistent with the equal-weighted returns.
    """
    empty = pd.Series(dtype=float)
    rows = []
    prev_mom, prev_mr = empty, empty
    for d in dates:
        mom, mr = mom_w.get(d, empty), mr_w.get(d, empty)
        rows.append({
            "date": d,
            "mom_to_mom": l1_distance(mom, prev_mom),
            "mr_to_mr": l1_distance(mr, prev_mr),
            "mom_to_mr": l1_distance(mr, prev_mom),
            "mr_to_mom": l1_distance(mom, prev_mr),
        })
        prev_mom, prev_mr = mom, mr
    return pd.DataFrame(rows).set_index("date")


def main():
    if not CRSP_CSV.exists():
        raise FileNotFoundError(f"{CRSP_CSV} not found. See data/README.md for the WRDS extract.")

    mom, mom_w = momentum_returns(CRSP_CSV)
    mr, mr_w = mean_reversion_returns(CRSP_CSV)

    # momentum needs 12 months of history, so the joint sample starts in 2006
    sleeves = pd.concat([mom.rename("mom_ret"), mr.rename("mr_ret")], axis=1, join="inner")
    sleeves.index.name = "date"
    sleeves.to_csv(DATA / "sleeve_returns.csv")
    build_turnover(sleeves.index, mom_w, mr_w).to_csv(DATA / "sleeve_turnover.csv")

    print(f"Saved {len(sleeves)} days to data/sleeve_returns.csv and data/sleeve_turnover.csv")
    print_perf_table([
        compute_perf_metrics(sleeves["mom_ret"], "Momentum"),
        compute_perf_metrics(sleeves["mr_ret"], "Mean-reversion"),
    ])


if __name__ == "__main__":
    main()
