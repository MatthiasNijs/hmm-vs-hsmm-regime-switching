"""Expanding-window HMM backtest gating between the momentum and mean-reversion sleeves.

A 2-state Gaussian HMM (full covariance) is fitted on z-scored SPY features and
re-estimated every 21 trading days on all data up to the previous day. Each
day's allocation uses the filtered state probabilities at t-1; the filter is
then updated with day t's features after the close.

Writes results/hmm_gated.csv.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from metrics import compute_perf_metrics, print_perf_table
from paths import DATA, RESULTS

FEATURE_COLS = ["ret_1d", "ret_20d", "park_vol_20d", "body_range_10d", "flip_rate_10d"]


# -------------------------------------------------
# HMM helpers
# -------------------------------------------------

def precompute_gaussians(covars: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Return Σ^{-1} and log det Σ per state, with a small diagonal regulariser."""
    cov = np.asarray(covars)
    n_states, n_features, _ = cov.shape

    precisions = np.zeros_like(cov)
    log_dets = np.zeros(n_states)
    for j in range(n_states):
        Sigma = cov[j] + eps * np.eye(n_features)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            # fall back to identity if the covariance is not positive definite
            Sigma = np.eye(n_features)
            sign, logdet = np.linalg.slogdet(Sigma)
        precisions[j] = np.linalg.inv(Sigma)
        log_dets[j] = logdet

    return precisions, log_dets


def emission_probs(x: np.ndarray, means: np.ndarray, precisions: np.ndarray,
                   log_dets: np.ndarray) -> np.ndarray:
    """Gaussian emission densities for one observation, rescaled so the max is 1."""
    n_states, n_features = means.shape
    diff = x - means
    quad = np.array([diff[j] @ precisions[j] @ diff[j] for j in range(n_states)])
    log_emis = -0.5 * (quad + n_features * np.log(2.0 * np.pi) + log_dets)
    return np.exp(log_emis - log_emis.max())


def normalise(alpha: np.ndarray) -> np.ndarray:
    s = alpha.sum()
    if (not np.isfinite(s)) or (s <= 0):
        return np.ones(len(alpha)) / len(alpha)
    return alpha / s


def compute_filtered_last(startprob: np.ndarray, transmat: np.ndarray, means: np.ndarray,
                          covars: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Scaled forward pass; returns the filtered state probabilities at the last row of X."""
    precisions, log_dets = precompute_gaussians(covars)

    alpha = None
    for x_t in X:
        emis = emission_probs(x_t, means, precisions, log_dets)
        alpha = startprob * emis if alpha is None else (alpha @ transmat) * emis
        alpha = normalise(alpha)

    return alpha


def fit_hmm(X: np.ndarray, n_states: int = 2, random_state: int = 42):
    """Fit a Gaussian HMM and return (model, filtered probs at the last row of X)."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        tol=1e-4,
        verbose=False,
    )
    model.fit(X)

    alpha_last = compute_filtered_last(model.startprob_, model.transmat_,
                                       model.means_, model.covars_, X)
    return model, alpha_last


# -------------------------------------------------
# Backtest core
# -------------------------------------------------

def run_backtest(features: pd.DataFrame, sleeves: pd.DataFrame,
                 train_start_date: str = "2002-01-01", retrain_interval: int = 21) -> pd.DataFrame:
    feat = features.sort_index()
    sleeves = sleeves.sort_index()

    # Only dates where we have both features and sleeve returns
    dates = feat.index.intersection(sleeves.index).sort_values()
    if len(dates) == 0:
        raise RuntimeError("No overlapping dates between features and sleeve returns")

    train_start_ts = max(pd.Timestamp(train_start_date), feat.index.min())

    def train(train_mask):
        X_raw = feat.loc[train_mask, FEATURE_COLS].values
        # Scaling uses only the training window
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0, ddof=1)
        std[std == 0.0] = 1.0
        model, alpha_last = fit_hmm((X_raw - mean) / std)
        # The state with the highest mean ret_1d is "trend"
        trend_state = int(np.argmax(model.means_[:, 0]))
        precisions, log_dets = precompute_gaussians(model.covars_)
        return model, alpha_last, mean, std, trend_state, precisions, log_dets

    # ---------- initial training: everything before the first allocation date ----------
    train_mask = (feat.index >= train_start_ts) & (feat.index < dates[0])
    if train_mask.sum() < 50:
        raise RuntimeError("Not enough data in initial training window.")
    model, alpha, scale_mean, scale_std, trend_state, precisions, log_dets = train(train_mask)

    rows = []
    for idx, d in enumerate(dates):
        # Retrain every retrain_interval trading days on an expanding window ending at d-1
        if idx > 0 and idx % retrain_interval == 0:
            train_mask = (feat.index >= train_start_ts) & (feat.index <= dates[idx - 1])
            model, alpha, scale_mean, scale_std, trend_state, precisions, log_dets = train(train_mask)

        # ---------- allocation for day d using filtered probs at d-1 ----------
        alpha_alloc = normalise(alpha.copy())
        in_trend = int(np.argmax(alpha_alloc)) == trend_state
        sleeve_today = "mom" if in_trend else "mr"

        mom_r = float(sleeves.loc[d, "mom_ret"])
        mr_r = float(sleeves.loc[d, "mr_ret"])
        rows.append({
            "date": d,
            "mom_ret": mom_r,
            "mr_ret": mr_r,
            "chosen_ret": mom_r if in_trend else mr_r,
            "sleeve": sleeve_today,
            "state_label": "trend" if in_trend else "chop",
            "p_trend": float(alpha_alloc[trend_state]),
        })

        # ---------- end-of-day filter update with day d's features ----------
        x_d = (feat.loc[d, FEATURE_COLS].values - scale_mean) / scale_std
        alpha_pred = normalise(alpha @ model.transmat_)  # p(S_d | X_{<=d-1})
        alpha = normalise(alpha_pred * emission_probs(x_d, model.means_, precisions, log_dets))

    return pd.DataFrame(rows).set_index("date")


def main():
    features = pd.read_csv(DATA / "spy_features.csv", parse_dates=["date"], index_col="date")
    sleeves = pd.read_csv(DATA / "sleeve_returns.csv", parse_dates=["date"], index_col="date")

    out = run_backtest(features, sleeves, train_start_date="2002-01-01", retrain_interval=21)
    out.to_csv(RESULTS / "hmm_gated.csv")

    print(f"Saved results/hmm_gated.csv: {(out['sleeve'] == 'mom').sum()} momentum days, "
          f"{(out['sleeve'] == 'mr').sum()} mean-reversion days")
    print_perf_table([
        compute_perf_metrics(out["chosen_ret"], "HMM-gated"),
        compute_perf_metrics(out.loc[out["sleeve"] == "mom", "mom_ret"], "Momentum (HMM days)"),
        compute_perf_metrics(out.loc[out["sleeve"] == "mr", "mr_ret"], "Mean-reversion (HMM days)"),
    ])


if __name__ == "__main__":
    main()
