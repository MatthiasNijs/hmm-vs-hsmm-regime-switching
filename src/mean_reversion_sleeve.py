"""Mean-reversion sleeve: short-term reversal on S&P 500 constituents.

Each day, rank stocks by their 5-day return: buy the 50 biggest losers and
short the 50 biggest winners, equal-weighted. Each formation is held for
3 trading days, so the sleeve return is the average of 3 overlapping
long-short portfolios.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COL = "DlyCalDt"
ID_COL = "PERMNO"
RET_PREFERRED = "DlyRetx"
RET_FALLBACK = "DlyRet"

START_YEAR = 2005
END_YEAR = 2024

TOP_N_PER_SIDE = 50      # long 50, short 50
LOOKBACK_DAYS = 5        # 5-day reversal window
HOLD_DAYS = 3            # hold each position for 3 trading days
MIN_UNIVERSE = 200       # skip days with too few names


def load_panel(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df[(df[DATE_COL].dt.year >= START_YEAR) & (df[DATE_COL].dt.year <= END_YEAR)]

    ret_col = RET_PREFERRED if RET_PREFERRED in df.columns else RET_FALLBACK
    df = df[[ID_COL, DATE_COL, ret_col]].rename(columns={ret_col: "ret"})
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["ret"])
    return df.sort_values([DATE_COL, ID_COL])


def compute_reversal_signal(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.pivot(index=DATE_COL, columns=ID_COL, values="ret").sort_index()

    # rolling 5-day compounded return
    roll5 = np.log1p(panel).rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()
    r5d = np.expm1(roll5)

    # cross-sectional z-scores each day
    mu = r5d.mean(axis=1)
    sd = r5d.std(axis=1, ddof=0)
    z = r5d.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    z_long = z.stack().rename("z_rev").reset_index().rename(columns={"level_1": ID_COL})
    return df.merge(z_long, on=[DATE_COL, ID_COL], how="left")


def backtest_mean_reversion(signals: pd.DataFrame):
    """Return (daily returns DataFrame, {date: portfolio weights held that day})."""
    panel_ret = signals.pivot(index=DATE_COL, columns=ID_COL, values="ret").sort_index()
    panel_sig = signals.pivot(index=DATE_COL, columns=ID_COL, values="z_rev").sort_index()

    dates = panel_ret.index.to_list()
    port_ret = pd.Series(index=dates, dtype=float)
    holdings = {}

    # long and short ID lists for the last HOLD_DAYS formations
    active_formations = deque(maxlen=HOLD_DAYS)

    for t in range(len(dates) - 1):  # last day has no next-day return
        dt = dates[t]
        dt_next = dates[t + 1]

        # form a new portfolio at the close of dt if the universe is large enough;
        # otherwise keep holding the existing formations
        sig_t = panel_sig.loc[dt].dropna()
        if sig_t.size >= max(MIN_UNIVERSE, 2 * TOP_N_PER_SIDE):
            longs = sig_t.nsmallest(TOP_N_PER_SIDE).index.to_numpy()
            shorts = sig_t.nlargest(TOP_N_PER_SIDE).index.to_numpy()
            active_formations.append((longs, shorts))

        if len(active_formations) == 0:
            port_ret.loc[dt_next] = 0.0
            continue

        # PnL on dt_next, averaged across all overlapping formations
        rets_next = panel_ret.loc[dt_next]
        daily_ls = []
        for longs, shorts in active_formations:
            r_long = rets_next.reindex(longs).mean()
            r_short = rets_next.reindex(shorts).mean()
            daily_ls.append(r_long - r_short)

        port_ret.loc[dt_next] = np.nanmean(daily_ls)

        # each formation gets an equal share of capital: +1/50 per long, -1/50 per short
        w = 1.0 / (len(active_formations) * TOP_N_PER_SIDE)
        parts = [pd.Series(w, index=longs) for longs, _ in active_formations]
        parts += [pd.Series(-w, index=shorts) for _, shorts in active_formations]
        holdings[dt_next] = pd.concat(parts).groupby(level=0).sum()

    return pd.DataFrame({"port_ret": port_ret.dropna()}), holdings


def mean_reversion_returns(csv_path: Path):
    """Daily mean-reversion sleeve returns and holdings."""
    signals = compute_reversal_signal(load_panel(csv_path))
    port_df, holdings = backtest_mean_reversion(signals)
    return port_df["port_ret"], holdings
