"""Apply the HSMM regime history from R to the sleeves (same hard gate as the HMM).

Reads results/hsmm_regimes.csv and data/sleeve_returns.csv,
writes results/hsmm_gated.csv.
"""

import numpy as np
import pandas as pd

from metrics import compute_perf_metrics, print_perf_table
from paths import DATA, RESULTS


def build_hsmm_gated_results(sleeves: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    dates = sleeves.index.intersection(regimes.index).sort_values()
    if len(dates) == 0:
        raise RuntimeError("No overlapping dates between sleeves and HSMM regimes.")

    reg = regimes.loc[dates]
    df = sleeves.loc[dates, ["mom_ret", "mr_ret"]].astype(float)

    # Use momentum when the most likely state at t-1 is the trend state
    use_mom = reg["state_ml"].astype(int) == reg["trend_state_id"].astype(int)
    df["chosen_ret"] = np.where(use_mom, df["mom_ret"], df["mr_ret"])
    df["sleeve"] = np.where(use_mom, "mom", "mr")
    df["state_label"] = np.where(use_mom, "trend", "chop")
    df["p_trend"] = reg["trend_prob"].astype(float)
    return df


def main():
    sleeves = pd.read_csv(DATA / "sleeve_returns.csv", parse_dates=["date"], index_col="date")
    regimes = pd.read_csv(RESULTS / "hsmm_regimes.csv", parse_dates=["date"], index_col="date")

    out = build_hsmm_gated_results(sleeves, regimes)
    out.to_csv(RESULTS / "hsmm_gated.csv")

    print(f"Saved results/hsmm_gated.csv: {(out['sleeve'] == 'mom').sum()} momentum days, "
          f"{(out['sleeve'] == 'mr').sum()} mean-reversion days")
    print_perf_table([
        compute_perf_metrics(out["chosen_ret"], "HSMM-gated"),
        compute_perf_metrics(out.loc[out["sleeve"] == "mom", "mom_ret"], "Momentum (HSMM days)"),
        compute_perf_metrics(out.loc[out["sleeve"] == "mr", "mr_ret"], "Mean-reversion (HSMM days)"),
    ])


if __name__ == "__main__":
    main()
