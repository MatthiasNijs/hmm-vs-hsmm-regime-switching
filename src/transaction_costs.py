"""Transaction-cost sensitivity of the sleeves and the gated strategies.

Costs are charged as c x turnover each day, where turnover is the one-way fraction
of capital traded (data/sleeve_turnover.csv) and c is a cost per unit traded in
basis points. The break-even cost is the c at which the mean net return is zero.

Writes results/tables/transaction_costs.csv.
"""

import numpy as np
import pandas as pd

from metrics import TRADING_DAYS
from paths import DATA, RESULTS, TABLES

COSTS_BPS = [0, 2, 5, 10]


def gated_turnover(sleeve: pd.Series, turnover: pd.DataFrame) -> pd.Series:
    """Daily turnover of a strategy that holds the sleeve named in `sleeve` ('mom' or 'mr')."""
    prev = sleeve.shift(1).fillna(sleeve.iloc[0])
    key = prev + "_to_" + sleeve
    return pd.Series(
        [turnover.at[d, k] for d, k in zip(sleeve.index, key)], index=sleeve.index
    )


def cost_row(name: str, gross: pd.Series, turnover: pd.Series) -> dict:
    row = {
        "strategy": name,
        "ann_turnover": turnover.mean() * TRADING_DAYS,
        "break_even_bps": gross.mean() / turnover.mean() * 1e4,
    }
    for c in COSTS_BPS:
        net = gross - c / 1e4 * turnover
        row[f"ann_return_{c}bps"] = net.mean() * TRADING_DAYS
        row[f"sharpe_{c}bps"] = net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return row


def main():
    sleeves = pd.read_csv(DATA / "sleeve_returns.csv", parse_dates=["date"], index_col="date")
    turnover = pd.read_csv(DATA / "sleeve_turnover.csv", parse_dates=["date"], index_col="date")
    hmm = pd.read_csv(RESULTS / "hmm_gated.csv", parse_dates=["date"], index_col="date")
    hsmm = pd.read_csv(RESULTS / "hsmm_gated.csv", parse_dates=["date"], index_col="date")

    dates = hmm.index.intersection(hsmm.index)
    sleeves, turnover = sleeves.loc[dates], turnover.loc[dates]

    rows = [
        cost_row("Momentum", sleeves["mom_ret"], turnover["mom_to_mom"]),
        cost_row("Mean-reversion", sleeves["mr_ret"], turnover["mr_to_mr"]),
        cost_row("HMM-gated", hmm.loc[dates, "chosen_ret"], gated_turnover(hmm.loc[dates, "sleeve"], turnover)),
        cost_row("HSMM-gated", hsmm.loc[dates, "chosen_ret"], gated_turnover(hsmm.loc[dates, "sleeve"], turnover)),
    ]
    table = pd.DataFrame(rows).set_index("strategy")
    table.to_csv(TABLES / "transaction_costs.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
