"""Tables and figures for the paper (exhibit numbers refer to report/).

Reads data/ and results/{hmm,hsmm}_gated.csv; writes results/tables and results/figures.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from metrics import TRADING_DAYS, compute_cvar, compute_equity_and_drawdown, compute_perf_metrics
from paths import DATA, FIGURES, RESULTS, TABLES

REGIME_COLORS = {"trend": "tab:green", "chop": "tab:red"}
STATE_ID_MAP = {"trend": 1, "chop": 2}


# ==============================
# Helpers
# ==============================

def run_lengths(series: pd.Series) -> pd.DataFrame:
    """Run lengths of a categorical series as a DataFrame with columns 'value', 'length'."""
    s = series.dropna().values
    if len(s) == 0:
        return pd.DataFrame(columns=["value", "length"])

    values, lengths = [], []
    current_val, current_len = s[0], 1
    for v in s[1:]:
        if v == current_val:
            current_len += 1
        else:
            values.append(current_val)
            lengths.append(current_len)
            current_val, current_len = v, 1
    values.append(current_val)
    lengths.append(current_len)

    return pd.DataFrame({"value": values, "length": lengths})


def count_switches(series: pd.Series) -> int:
    """Number of changes in a categorical series (the first day is not a switch)."""
    s = series.dropna()
    return int((s != s.shift(1)).iloc[1:].sum())


def estimate_transition_and_stationary(state_series: pd.Series, n_states: int = 2):
    """Empirical one-step transition matrix and its stationary distribution (states 1..n)."""
    states = state_series.dropna().astype(int).values
    if len(states) < 2:
        return np.eye(n_states), np.ones(n_states) / n_states

    counts = np.zeros((n_states, n_states))
    for s0, s1 in zip(states[:-1] - 1, states[1:] - 1):
        counts[s0, s1] += 1.0

    trans = np.zeros_like(counts)
    for i in range(n_states):
        row_sum = counts[i].sum()
        if row_sum > 0:
            trans[i] = counts[i] / row_sum
        else:
            trans[i, i] = 1.0

    vals, vecs = np.linalg.eig(trans.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    pi = np.where(pi < 0, 0, pi)
    pi = pi / pi.sum() if pi.sum() > 0 else np.ones(n_states) / n_states

    return trans, pi


def color_line_by_regime(ax, dates, values, regimes):
    """Plot a line on ax, colouring segments by regime label."""
    idx = pd.to_datetime(dates)
    s_vals = pd.Series(np.asarray(values), index=idx)
    pos = 0
    for val, length in zip(*run_lengths(pd.Series(np.asarray(regimes), index=idx)).values.T):
        seg_idx = idx[pos: pos + length]
        ax.plot(seg_idx, s_vals.loc[seg_idx], color=REGIME_COLORS.get(val, "k"), linewidth=1.2)
        pos += length


def shade_regimes(ax, labels: pd.Series):
    """Shade the background of ax by regime label."""
    s = labels.dropna()
    pos = 0
    for val, length in zip(*run_lengths(s).values.T):
        ax.axvspan(s.index[pos], s.index[pos + length - 1],
                   color=REGIME_COLORS.get(val, "lightgray"), alpha=0.12)
        pos += length


def add_regime_legend(ax):
    handles = [
        Patch(facecolor=REGIME_COLORS["trend"], edgecolor="none", alpha=0.6, label="Trend regime"),
        Patch(facecolor=REGIME_COLORS["chop"], edgecolor="none", alpha=0.6, label="Chop regime"),
    ]
    ax.legend(handles=handles, loc="best")


def apply_date_format(axes, minticks=3, maxticks=8):
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in np.atleast_1d(axes):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


# ==============================
# Drawdown episode detection
# ==============================

def _peak_indices(eq: pd.Series) -> np.ndarray:
    """Index of the running peak for every date."""
    arr = eq.values
    peaks = np.zeros(len(arr), dtype=int)
    last_peak = 0
    for i in range(len(arr)):
        if arr[i] >= arr[last_peak]:
            last_peak = i
        peaks[i] = last_peak
    return peaks


def _drawdown_episodes(eq: pd.Series, vol_21: pd.Series, threshold: float, min_duration: int,
                       max_duration: int, tail_extension_days: int,
                       exclude_below: float | None = None,
                       whole_drawdown: bool = False) -> pd.DataFrame:
    """
    Episodes where the drawdown falls below `threshold`:
    peak = running peak before the episode, trough = lowest point in the run below
    threshold (with whole_drawdown, the lowest point before the market regains
    that peak, so brief recoveries above threshold do not split an episode),
    end = trough + tail_extension_days. Episodes outside
    [min_duration, max_duration] (peak to end; no upper limit if max_duration is
    None) are dropped, as are episodes that also fall below `exclude_below`.
    Overlapping episodes keep the earliest one.
    """
    dd = eq / eq.cummax() - 1.0
    below = dd <= threshold
    peaks = _peak_indices(eq)

    episodes = []
    i, n = 0, len(eq)
    while i < n:
        if not below.iloc[i]:
            i += 1
            continue

        run_start = i
        peak_idx = peaks[run_start]
        run_end = i
        while i + 1 < n and (below.iloc[i + 1] or (whole_drawdown and peaks[i + 1] == peak_idx)):
            i += 1
            if below.iloc[i]:
                run_end = i

        if exclude_below is not None and (dd.iloc[run_start: run_end + 1] <= exclude_below).any():
            i += 1
            continue

        trough_idx = run_start + int(eq.iloc[run_start: run_end + 1].values.argmin())
        end_idx = min(trough_idx + tail_extension_days, n - 1)
        duration = end_idx - peak_idx + 1
        if min_duration <= duration and (max_duration is None or duration <= max_duration):
            episodes.append({
                "peak_date": eq.index[peak_idx],
                "trough_date": eq.index[trough_idx],
                "end_date": eq.index[end_idx],
                "duration": duration,
                "drawdown": eq.iloc[trough_idx] / eq.iloc[peak_idx] - 1.0,
                "avg_vol_21": vol_21.loc[eq.index[peak_idx]:eq.index[end_idx]].dropna().mean(),
            })
        i += 1

    cols = ["peak_date", "trough_date", "end_date", "duration", "drawdown", "avg_vol_21"]
    return pd.DataFrame(episodes, columns=cols).sort_values("peak_date").reset_index(drop=True)


def _drop_overlapping(episodes: pd.DataFrame) -> pd.DataFrame:
    accepted, last_end = [], None
    for _, row in episodes.iterrows():
        if last_end is None or row["peak_date"] > last_end:
            accepted.append(row)
            last_end = row["end_date"]
    return pd.DataFrame(accepted, columns=episodes.columns).reset_index(drop=True)


def find_persistent_drawdowns(eq_spy, vol_21, dd_threshold=-0.25, min_duration=60,
                              max_duration=None, tail_extension_days=40, vol_quantile=0.75):
    """Long drawdowns whose average volatility is above the vol_quantile of the full sample."""
    eps = _drawdown_episodes(eq_spy.dropna(), vol_21, dd_threshold, min_duration,
                             max_duration, tail_extension_days, whole_drawdown=True)
    eps = eps[eps["avg_vol_21"] >= vol_21.dropna().quantile(vol_quantile)]
    return _drop_overlapping(eps)


def find_transient_spikes(eq_spy, vol_21, spike_threshold=-0.08, persistent_threshold=-0.30,
                          min_duration=2, max_duration=40, tail_extension_days=10):
    """Short, sharp drawdowns that never reach persistent_threshold."""
    eps = _drawdown_episodes(eq_spy.dropna(), vol_21, spike_threshold, min_duration,
                             max_duration, tail_extension_days, exclude_below=persistent_threshold)
    return _drop_overlapping(eps)


def remove_spikes_overlapping_persistent(spikes: pd.DataFrame, persistent: pd.DataFrame) -> pd.DataFrame:
    if spikes.empty or persistent.empty:
        return spikes
    keep = [
        not ((persistent["peak_date"] <= row["end_date"]) & (persistent["end_date"] >= row["peak_date"])).any()
        for _, row in spikes.iterrows()
    ]
    return spikes.loc[keep].reset_index(drop=True)


# ==============================
# Episode analysis
# ==============================

def episode_metrics(episodes: pd.DataFrame, models: dict, id_col: str) -> pd.DataFrame:
    rows = []
    for eid, row in episodes.iterrows():
        for name, df in models.items():
            win = df.loc[row["peak_date"]:row["end_date"]]
            if win.empty:
                continue
            rets = win["chosen_ret"]
            _, dd_ep = compute_equity_and_drawdown(rets)
            rows.append({
                id_col: eid,
                "model": name,
                "episode_return": (1.0 + rets).prod() - 1.0,
                "episode_max_drawdown": dd_ep.min(),
                "episode_cvar_95": compute_cvar(rets),
                "regime_switches": count_switches(win["state_label"]),
                "fraction_defensive_mr": (win["sleeve"] == "mr").mean(),
            })
    return pd.DataFrame(rows)


def plot_episode(title: str, hmm_w, hsmm_w, spy_eq, spy_vol_21, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
    for col, (name, win) in enumerate([("HMM", hmm_w), ("HSMM", hsmm_w)]):
        ax = axes[0, col]
        ax.set_title(f"{title}: SPY price ({name} regimes)")
        color_line_by_regime(ax, win.index, spy_eq.loc[win.index], win["state_label"])
        ax.set_ylabel("Price index")

        ax = axes[1, col]
        ax.set_title(f"Realised vol ({name} regimes)")
        color_line_by_regime(ax, win.index, spy_vol_21.loc[win.index], win["state_label"])
        ax.set_ylabel("Ann. vol (21d)")
        ax.set_xlabel("Date")

    for ax in axes.flatten():
        ax.grid(True)
        add_regime_legend(ax)
    apply_date_format(axes.flatten())
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ==============================
# Main analysis
# ==============================

def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # -------- load data --------
    feat = pd.read_csv(DATA / "spy_features.csv", parse_dates=["date"], index_col="date").sort_index()
    spy_ret = feat["ret_1d"].dropna()
    spy_eq = np.exp(spy_ret.cumsum())
    spy_vol_21 = spy_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)

    sleeves = pd.read_csv(DATA / "sleeve_returns.csv", parse_dates=["date"], index_col="date")
    hmm = pd.read_csv(RESULTS / "hmm_gated.csv", parse_dates=["date"], index_col="date")
    hsmm = pd.read_csv(RESULTS / "hsmm_gated.csv", parse_dates=["date"], index_col="date")

    common = sleeves.index.intersection(hmm.index).intersection(hsmm.index).sort_values()
    sleeves, hmm, hsmm = sleeves.loc[common], hmm.loc[common], hsmm.loc[common]
    for df in (hmm, hsmm):
        df["spy_ret"] = spy_ret.reindex(df.index)
        df["state_id"] = df["state_label"].map(STATE_ID_MAP).astype("Int64")
    alloc_start = common[0]

    # -------- Feature summary --------
    cols = ["ret_1d", "ret_20d", "park_vol_20d", "body_range_10d", "flip_rate_10d"]
    sub = feat[cols].dropna()
    pd.DataFrame({
        "mean": sub.mean(), "std": sub.std(ddof=1), "skew": sub.skew(), "kurtosis": sub.kurtosis(),
    }).to_csv(TABLES / "feature_summary.csv")

    # -------- SPY price and volatility --------
    roll_vol_60 = spy_ret.rolling(60).std() * np.sqrt(TRADING_DAYS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(spy_eq.index, spy_eq.values)
    axes[0].set_ylabel("SPY price index")
    axes[0].set_title("SPY price index and rolling 60-day realized volatility")
    axes[1].plot(roll_vol_60.index, roll_vol_60.values)
    axes[1].set_ylabel("Ann. vol (60d)")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(True)
    apply_date_format(axes)
    axes[0].tick_params(labelbottom=False)
    plt.tight_layout()
    fig.savefig(FIGURES / "spy_price_and_vol.png", dpi=150)
    plt.close(fig)

    # -------- Exhibit 1: sleeve performance --------
    pd.DataFrame([
        compute_perf_metrics(sleeves["mom_ret"], "Momentum"),
        compute_perf_metrics(sleeves["mr_ret"], "Mean-reversion"),
    ]).set_index("name").to_csv(TABLES / "exhibit01_sleeve_performance.csv")

    # -------- Exhibit 2: SPY returns by regime --------
    runs = {"HMM": run_lengths(hmm["state_id"]), "HSMM": run_lengths(hsmm["state_id"])}
    rows = []
    for model_name, df in [("HMM", hmm), ("HSMM", hsmm)]:
        for regime, sid in STATE_ID_MAP.items():
            r = df.loc[df["state_id"] == sid, "spy_ret"].dropna()
            state_runs = runs[model_name].query("value == @sid")["length"]
            rows.append({
                "model": model_name,
                "regime": regime,
                "ann_return": r.mean() * TRADING_DAYS,
                "ann_vol": r.std(ddof=1) * np.sqrt(TRADING_DAYS),
                "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS),
                "mean_run_length": state_runs.mean(),
                "median_run_length": state_runs.median(),
            })
    pd.DataFrame(rows).to_csv(TABLES / "exhibit02_regime_stats.csv", index=False)

    # -------- Exhibit 3: empirical transition matrix --------
    rows_t, rows_pi = [], []
    for name, df in [("HMM", hmm), ("HSMM", hsmm)]:
        trans, pi = estimate_transition_and_stationary(df["state_id"])
        for from_regime, i in STATE_ID_MAP.items():
            rows_pi.append({"model": name, "regime": from_regime, "stationary_prob": pi[i - 1]})
            for to_regime, j in STATE_ID_MAP.items():
                rows_t.append({"model": name, "from": from_regime, "to": to_regime,
                               "probability": trans[i - 1, j - 1]})
    pd.DataFrame(rows_t).to_csv(TABLES / "exhibit03_transition_matrix.csv", index=False)
    pd.DataFrame(rows_pi).to_csv(TABLES / "stationary_distribution.csv", index=False)

    # -------- Exhibit 4: run-length ECDFs --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for (regime, sid), ax in zip(STATE_ID_MAP.items(), axes):
        lengths = {m: runs[m].query("value == @sid")["length"].values for m in runs}
        for m, style in [("HMM", "-"), ("HSMM", "--")]:
            v = np.sort(lengths[m])
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=m, linestyle=style)
        ax.set_xlim(0, np.percentile(np.concatenate(list(lengths.values())), 99) * 1.05)
        ax.set_title(f"{regime.capitalize()} regime run-length ECDF")
        ax.set_xlabel("Run length (days)")
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("Cumulative probability")
    plt.tight_layout()
    fig.savefig(FIGURES / "exhibit04_run_length_ecdf.png", dpi=150)
    plt.close(fig)

    # -------- Exhibit 5: SPY with regime highlights --------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (name, df) in zip(axes, [("HMM", hmm), ("HSMM", hsmm)]):
        ax.plot(spy_eq.index, spy_eq.values, color="black", linewidth=1.0)
        shade_regimes(ax, df["state_label"])
        ax.set_ylabel("SPY price index")
        ax.set_title(f"{name} regimes (vertical highlights)")
        ax.grid(True)
        add_regime_legend(ax)
    axes[1].set_xlabel("Date")
    apply_date_format(axes)
    axes[0].tick_params(labelbottom=False)
    plt.tight_layout()
    fig.savefig(FIGURES / "exhibit05_regime_highlights.png", dpi=150)
    plt.close(fig)

    # -------- Exhibit 6: gated and conditional sleeve performance --------
    rows6 = []
    for name, df in [("HMM", hmm), ("HSMM", hsmm)]:
        rows6 += [
            compute_perf_metrics(df["chosen_ret"], f"{name}-gated portfolio"),
            compute_perf_metrics(df.loc[df["sleeve"] == "mom", "mom_ret"], f"Momentum sleeve ({name} days)"),
            compute_perf_metrics(df.loc[df["sleeve"] == "mr", "mr_ret"], f"Mean-reversion sleeve ({name} days)"),
        ]
    pd.DataFrame(rows6).set_index("name").to_csv(TABLES / "exhibit06_gated_performance.csv")

    # -------- Exhibit 7: cumulative returns --------
    eq = (1.0 + pd.DataFrame({
        "Momentum (ungated)": sleeves["mom_ret"],
        "Mean-reversion (ungated)": sleeves["mr_ret"],
        "HMM-gated": hmm["chosen_ret"],
        "HSMM-gated": hsmm["chosen_ret"],
    })).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in eq.columns:
        ax.plot(eq.index, eq[col], label=col)
    ax.set_ylabel("Growth of 1")
    ax.set_xlabel("Date")
    ax.set_title("Cumulative returns: sleeves and gated portfolios")
    ax.legend()
    ax.grid(True)
    apply_date_format(ax)
    plt.tight_layout()
    fig.savefig(FIGURES / "exhibit07_cumulative_returns.png", dpi=150)
    plt.close(fig)

    # -------- Gated drawdowns --------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (name, df) in zip(axes, [("HMM", hmm), ("HSMM", hsmm)]):
        _, dd = compute_equity_and_drawdown(df["chosen_ret"])
        ax.plot(dd.index, dd.values)
        ax.set_title(f"{name}-gated portfolio drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True)
    axes[1].set_xlabel("Date")
    apply_date_format(axes)
    axes[0].tick_params(labelbottom=False)
    plt.tight_layout()
    fig.savefig(FIGURES / "gated_drawdowns.png", dpi=150)
    plt.close(fig)

    models = {"HMM": hmm, "HSMM": hsmm}

    # -------- Exhibits 8-11: persistent drawdowns --------
    persistent = find_persistent_drawdowns(spy_eq, spy_vol_21)
    episodes_p = persistent[persistent["peak_date"] >= alloc_start].reset_index(drop=True)
    episodes_p.index = episodes_p.index + 1
    episodes_p.to_csv(TABLES / "exhibit08_persistent_drawdowns.csv", index_label="episode_id")
    episode_metrics(episodes_p, models, "episode_id").to_csv(
        TABLES / "exhibit09_persistent_drawdown_metrics.csv", index=False)
    for eid, row in episodes_p.iterrows():
        win = slice(row["peak_date"], row["end_date"])
        plot_episode(f"Drawdown {eid}", hmm.loc[win], hsmm.loc[win], spy_eq, spy_vol_21,
                     FIGURES / f"persistent_drawdown_{eid}.png")

    # -------- Exhibits 12-15: short drawdown spikes --------
    spikes = find_transient_spikes(spy_eq, spy_vol_21)
    spikes = remove_spikes_overlapping_persistent(spikes, persistent)
    spikes = spikes[spikes["peak_date"] >= alloc_start].reset_index(drop=True)
    spikes.index = spikes.index + 1
    spikes.to_csv(TABLES / "exhibit12_drawdown_spikes.csv", index_label="spike_id")
    episode_metrics(spikes, models, "spike_id").to_csv(
        TABLES / "exhibit13_spike_metrics.csv", index=False)
    for sid, row in spikes.iterrows():
        win = slice(row["peak_date"], row["end_date"])
        plot_episode(f"Spike {sid}", hmm.loc[win], hsmm.loc[win], spy_eq, spy_vol_21,
                     FIGURES / f"spike_{sid}.png")

    print(f"Wrote {len(list(TABLES.glob('*.csv')))} tables and "
          f"{len(list(FIGURES.glob('*.png')))} figures to results/")


if __name__ == "__main__":
    main()
