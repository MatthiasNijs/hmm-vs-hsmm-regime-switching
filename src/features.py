"""Build the five SPY regime features shared by the HMM and the HSMM.

Writes data/spy_features.csv. The committed copy was downloaded on 2025-11-28;
re-running pulls fresh data from Yahoo Finance, which may differ in the last
decimals.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from paths import DATA


def build_ohlc_features(start_date: str, end_date: str, ticker: str = "SPY") -> pd.DataFrame:
    """
    Download daily OHLC data and build the regime features:
      - ret_1d        : 1-day close-to-close log return
      - ret_20d       : 20-day cumulative log return (trend)
      - park_vol_20d  : 20-day Parkinson volatility from high/low
      - body_range_10d: 10-day average |close - open| / (high - low)
      - flip_rate_10d : 10-day rate of sign flips in daily returns
    """
    px = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if px.empty:
        raise RuntimeError("No price data downloaded for the given range.")

    # yfinance returns (field, ticker) MultiIndex columns
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)

    px = px[["Open", "High", "Low", "Close"]].dropna().sort_index()
    px.index.name = "date"

    # 1-day close-to-close log return
    cc_ret = np.log(px["Close"] / px["Close"].shift(1))

    # 20-day cumulative log return (trend)
    ret_20d = cc_ret.rolling(20).sum()

    # Parkinson 20-day realised volatility from high/low
    hl_log = np.log(px["High"] / px["Low"])
    park_var_20d = (hl_log ** 2).rolling(20).mean() / (4.0 * np.log(2.0))
    park_vol_20d = np.sqrt(park_var_20d)

    # Body-to-range ratio (averaged over 10 days)
    body = (px["Close"] - px["Open"]).abs()
    bar_range = px["High"] - px["Low"]
    body_range = body / bar_range.replace(0.0, np.nan)
    body_range = body_range.replace([np.inf, -np.inf], np.nan)
    body_range_10d = body_range.rolling(10).mean()

    # 10-day sign flip rate of daily returns
    sign = np.sign(cc_ret)
    sign_change = (sign != sign.shift(1)) & sign.notna() & sign.shift(1).notna()
    flip_rate_10d = sign_change.rolling(10).mean()

    features = pd.DataFrame(
        {
            "ret_1d": cc_ret,
            "ret_20d": ret_20d,
            "park_vol_20d": park_vol_20d,
            "body_range_10d": body_range_10d,
            "flip_rate_10d": flip_rate_10d,
        },
        index=px.index,
    )

    # Drop initial rows where rolling windows are incomplete
    return features.dropna()


def main():
    features = build_ohlc_features("2002-01-01", "2024-12-31", ticker="SPY")
    features.to_csv(DATA / "spy_features.csv")
    print(f"Saved {len(features)} rows to data/spy_features.csv "
          f"({features.index[0].date()} to {features.index[-1].date()})")


if __name__ == "__main__":
    main()
