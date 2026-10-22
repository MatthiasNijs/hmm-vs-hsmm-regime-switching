"""Stationary bootstrap tests of Sharpe-ratio differences (Politis & Romano, 1994).

Resampling blocks of days with geometric lengths keeps the serial dependence and the
cross-correlation between strategies. The p-value is two-sided: twice the share of
bootstrap differences on the opposite side of zero, capped at 1.

Writes results/tables/sharpe_tests.csv.
"""

import numpy as np
import pandas as pd

from metrics import TRADING_DAYS
from paths import DATA, RESULTS, TABLES

N_BOOT = 10_000
MEAN_BLOCK = 21  # trading days
SEED = 547


def stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """A new block starts with probability 1/mean_block at a random day; otherwise continue (wrapping)."""
    new_block = rng.random(n) < 1.0 / mean_block
    new_block[0] = True
    starts = rng.integers(n, size=n)
    pos = np.arange(n)
    block_first = np.maximum.accumulate(np.where(new_block, pos, 0))
    return (starts[block_first] + pos - block_first) % n


def sharpe(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)


def main():
    sleeves = pd.read_csv(DATA / "sleeve_returns.csv", parse_dates=["date"], index_col="date")
    hmm = pd.read_csv(RESULTS / "hmm_gated.csv", parse_dates=["date"], index_col="date")
    hsmm = pd.read_csv(RESULTS / "hsmm_gated.csv", parse_dates=["date"], index_col="date")

    dates = hmm.index.intersection(hsmm.index)
    rets = pd.DataFrame({
        "HMM-gated": hmm.loc[dates, "chosen_ret"],
        "HSMM-gated": hsmm.loc[dates, "chosen_ret"],
        "Momentum": sleeves.loc[dates, "mom_ret"],
        "Mean-reversion": sleeves.loc[dates, "mr_ret"],
    })
    pairs = [
        ("HMM-gated", "Mean-reversion"),
        ("HMM-gated", "Momentum"),
        ("HSMM-gated", "Mean-reversion"),
        ("HSMM-gated", "Momentum"),
        ("HMM-gated", "HSMM-gated"),
    ]

    X = rets.values
    observed = dict(zip(rets.columns, sharpe(X)))
    rng = np.random.default_rng(SEED)
    boot = np.array([
        sharpe(X[stationary_bootstrap_indices(len(X), MEAN_BLOCK, rng)]) for _ in range(N_BOOT)
    ])
    boot = pd.DataFrame(boot, columns=rets.columns)

    rows = []
    for a, b in pairs:
        diff = boot[a] - boot[b]
        obs = observed[a] - observed[b]
        p = min(1.0, 2 * ((diff <= 0).mean() if obs > 0 else (diff >= 0).mean()))
        rows.append({
            "strategy": a, "benchmark": b,
            "sharpe_diff": obs,
            "ci_low_95": diff.quantile(0.025), "ci_high_95": diff.quantile(0.975),
            "p_value": p,
        })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "sharpe_tests.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
