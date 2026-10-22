import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Conditional VaR of daily returns at tail level alpha (loss convention)."""
    rets = pd.Series(returns).dropna()
    if rets.empty:
        return np.nan
    losses = -rets.values
    var = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())


def compute_equity_and_drawdown(returns: pd.Series):
    """Return (equity_curve, drawdown_series) from daily returns."""
    rets = returns.dropna()
    equity = (1.0 + rets).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return equity, drawdown


def compute_perf_metrics(returns: pd.Series, name: str) -> dict:
    """Annualised mean, volatility, Sharpe (rf = 0), max drawdown and daily CVaR 95%."""
    rets = pd.Series(returns).dropna()
    n = len(rets)
    if n == 0:
        return {"name": name, "n_days": 0}

    mean_daily = rets.mean()
    vol_daily = rets.std(ddof=1)
    mean_ann = mean_daily * TRADING_DAYS
    vol_ann = vol_daily * np.sqrt(TRADING_DAYS)
    _, drawdown = compute_equity_and_drawdown(rets)

    return {
        "name": name,
        "n_days": n,
        "mean_daily": mean_daily,
        "vol_daily": vol_daily,
        "mean_ann": mean_ann,
        "vol_ann": vol_ann,
        "sharpe": mean_ann / vol_ann if vol_ann > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "cvar_95": compute_cvar(rets, alpha=0.95),
    }


def print_perf_table(stats_list: list[dict]) -> None:
    table = pd.DataFrame(stats_list).set_index("name")
    cols = ["n_days", "mean_ann", "vol_ann", "sharpe", "max_drawdown", "cvar_95"]
    print(table[cols].to_string(float_format=lambda x: f"{x:.3f}"))
