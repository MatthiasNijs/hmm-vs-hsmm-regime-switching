"""Momentum sleeve: long the top 50 S&P 500 stocks by blended 12/6/3-month momentum.

Scores are formed on each Friday close and held equal-weighted from the next
trading day through the following Friday.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATE_COL = "DlyCalDt"
ID_COL = "PERMNO"
RET_PREFERRED = "DlyRetx"
RET_FALLBACK = "DlyRet"
TOP_N = 50

# Momentum windows in trading days (approx.)
W12 = 252
W6 = 126
W3 = 63


def load_panel(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    ret_col = RET_PREFERRED if RET_PREFERRED in df.columns else RET_FALLBACK
    df = df[[ID_COL, DATE_COL, ret_col]].rename(columns={ret_col: "ret"})
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["ret"])
    return df.sort_values([ID_COL, DATE_COL])


def compute_momentum_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log1p = np.log1p(df["ret"]).groupby(df[ID_COL])

    # compounded return over each lookback window, per stock
    for col, window in [("mom12", W12), ("mom6", W6), ("mom3", W3)]:
        df[col] = np.expm1(log1p.transform(lambda s: s.rolling(window, min_periods=window).sum()))

    # cross-sectional z-scores per day
    def zscore(s: pd.Series) -> pd.Series:
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=0)
        return (s - mu) / (sd if sd and np.isfinite(sd) else np.nan)

    df["z12"] = df.groupby(DATE_COL)["mom12"].transform(zscore)
    df["z6"] = df.groupby(DATE_COL)["mom6"].transform(zscore)
    df["z3"] = df.groupby(DATE_COL)["mom3"].transform(zscore)

    # weight the z-scores of the momentum to create a score
    df["score"] = df["z12"] + 0.5 * df["z6"] + 0.25 * df["z3"]
    return df


def weekly_rebalance_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    all_dates = pd.Index(df[DATE_COL].sort_values().unique())
    dd = pd.DataFrame(index=all_dates, data={"dummy": 1})
    # W-FRI labels each week by its Friday; keep only Fridays that are trading days
    weekly = dd.resample("W-FRI").last().index
    return weekly.intersection(all_dates)


def build_portfolio_returns(df_scored: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    df = df_scored.dropna(subset=["score"])
    df = df.drop_duplicates(subset=[ID_COL, DATE_COL], keep="last")

    weekly_dates = weekly_rebalance_dates(df)

    ret_panel = df.pivot(index=DATE_COL, columns=ID_COL, values="ret").sort_index()
    score_panel = df.pivot(index=DATE_COL, columns=ID_COL, values="score").sort_index()
    weekly_dates = weekly_dates.intersection(score_panel.index)

    port_ret = pd.Series(index=ret_panel.index, dtype=float)

    for i, t in enumerate(weekly_dates):
        # hold from the next trading day after t through the next rebalance date
        loc = ret_panel.index.get_loc(t)
        if loc + 1 >= len(ret_panel.index):
            break

        start = ret_panel.index[loc + 1]
        end = (
            ret_panel.index[-1]
            if i + 1 >= len(weekly_dates)
            else ret_panel.index[ret_panel.index.get_loc(weekly_dates[i + 1])]
        )
        mask = (ret_panel.index >= start) & (ret_panel.index <= end)

        # choose top N by score at rebalance time t
        top_ids = score_panel.loc[t].dropna().nlargest(top_n).index

        if len(top_ids) > 0:
            port_ret.loc[mask] = ret_panel.loc[mask, top_ids].mean(axis=1).values
        else:
            port_ret.loc[mask] = 0.0

    return pd.DataFrame({"port_ret": port_ret.dropna()})


def momentum_returns(csv_path: Path) -> pd.Series:
    scored = compute_momentum_scores(load_panel(csv_path))
    return build_portfolio_returns(scored, top_n=TOP_N)["port_ret"]
