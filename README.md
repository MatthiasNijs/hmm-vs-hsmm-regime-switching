# Does regime duration matter? HMM vs HSMM for strategy allocation

Individual research paper, McGill University (FINE 547), December 2025; revised November 2026.
**[Read the paper (PDF)](report/hmm_vs_hsmm_regime_switching.pdf)**

A standard hidden Markov model assumes regimes are memoryless: the chance of leaving
a regime is the same on day 2 as on day 200, so regime lengths are geometric. A hidden
semi-Markov model (HSMM) drops that assumption and models how long each regime lasts.
This project tests whether that extra structure improves a practical decision:
switching daily between a momentum sleeve and a mean-reversion sleeve on S&P 500 stocks.

The two models are identical in every other respect (same five SPY features, two
Gaussian states, expanding-window retraining every 21 days, and the same hard gate),
so any difference comes from the duration model alone.

**Finding:** regime gating itself helps a lot, but duration modelling barely does.
- **Gating:** both gated strategies lift the Sharpe ratio from 0.47–0.69 for the
  standalone sleeves to about 0.9. A block bootstrap puts this at the edge of
  significance (p = 0.05–0.15).
- **HSMM vs HMM:** the HSMM separates "good" and "bad" regimes more sharply and stays
  defensive for longer in drawdowns, but its Sharpe ratio is indistinguishable from the
  HMM's (p = 0.72).
- **Costs:** after transaction costs the HMM comes out ahead, because the HSMM spends
  more days in the high-turnover sleeve.

This agrees with Baitinger & Hoch (2024) and extends their result from
regime-conditional weights to hard switching between strategies.

![Cumulative returns](results/figures/exhibit07_cumulative_returns.png)

## Out-of-sample results, 2006–2024

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown | Daily CVaR 95% |
|---|---:|---:|---:|---:|---:|
| Momentum sleeve (always on) | 9.8% | 20.8% | 0.47 | −53.2% | 3.2% |
| Mean-reversion sleeve (always on) | 14.7% | 21.4% | 0.69 | −49.5% | 2.9% |
| **HMM-gated** | 21.4% | 23.4% | **0.91** | −51.3% | 3.3% |
| **HSMM-gated** | 20.5% | 23.2% | **0.89** | −49.5% | 3.2% |

Returns are gross of costs, annualised arithmetically, with a risk-free rate of zero.

Significance of the Sharpe differences comes from a stationary bootstrap (10,000
resamples, mean block length 21 days):

| Comparison | Sharpe difference | 95% CI | p-value |
|---|---:|---:|---:|
| HMM-gated vs momentum | +0.44 | [0.00, 0.90] | 0.05 |
| HMM-gated vs mean-reversion | +0.23 | [−0.04, 0.51] | 0.09 |
| HSMM-gated vs momentum | +0.41 | [−0.02, 0.87] | 0.07 |
| HSMM-gated vs mean-reversion | +0.20 | [−0.07, 0.49] | 0.15 |
| HMM-gated vs HSMM-gated | +0.03 | [−0.13, 0.19] | 0.72 |

### Transaction costs

The mean-reversion sleeve rebuilds a third of its book every day and turns over about
91% of capital daily. Costs are charged per unit of capital traded (one-way):

| Strategy | Annual turnover | Break-even cost | Sharpe at 0 / 2 / 5 / 10 bps |
|---|---:|---:|---:|
| Momentum | 16× | 61 bps | 0.47 / 0.46 / 0.43 / 0.40 |
| Mean-reversion | 230× | 6 bps | 0.69 / 0.47 / 0.15 / −0.39 |
| HMM-gated | 96× | 22 bps | 0.91 / 0.83 / 0.71 / 0.50 |
| HSMM-gated | 115× | 18 bps | 0.89 / 0.79 / 0.64 / 0.39 |

On its own, the mean-reversion sleeve does not survive realistic costs. The gated
strategies hold it only about a third of the time, so they are far more robust.

### Regimes

| Regime (SPY, by model) | Ann. return | Ann. vol | Mean / median length (days) |
|---|---:|---:|---:|
| HMM trend | 8.2% | 12.1% | 84.5 / 43 |
| HMM chop | 8.2% | 29.2% | 42.4 / 24 |
| HSMM trend | 10.5% | 12.4% | 44.5 / 13.5 |
| HSMM chop | 4.5% | 27.1% | 27.9 / 7 |

The HMM's two states differ only in volatility. The HSMM's differ in both return and
volatility. Its fitted Poisson durations come out *shorter* than the HMM's implied
geometric ones, so the extra flexibility sharpens regime separation rather than
lengthening regimes.

### Behaviour in drawdowns

Episodes are detected by an algorithm, not chosen by hand, and every detected episode is
reported. A persistent drawdown runs from the market peak to the lowest close before SPY
regains that peak.

| Persistent drawdown | SPY drawdown | HMM return | HSMM return | Switches (HMM / HSMM) | Days defensive (HMM / HSMM) |
|---|---:|---:|---:|---:|---:|
| 2007–09 financial crisis | −56.5% | +179.4% | +167.5% | 5 / 1 | 92% / 98% |
| 2020 Covid crash | −34.1% | +44.0% | +44.0% | 1 / 1 | 94% / 94% |
| 2022 rate-hike drawdown | −25.4% | +3.3% | +8.6% | 6 / 1 | 93% / 100% |

| Short spike | SPY drawdown | HMM return | HSMM return |
|---|---:|---:|---:|
| Aug 2007 | −9.0% | +4.9% | +7.3% |
| Feb 2018 | −10.1% | −7.4% | −1.0% |
| Oct 2018 | −10.1% | −3.9% | −6.4% |
| Sep 2020 | −8.6% | −1.4% | +3.5% |
| Aug 2024 | −8.4% | −10.3% | −10.3% |

The HSMM, once defensive, tends to stay defensive. That pays off when a drawdown keeps
going (2022) and costs return when there are tradeable rallies inside it (2008). So the
HSMM is better seen as a way to encode a preferred defensive behaviour than as a source
of extra alpha.

## Method

- **Features (SPY, daily):** 1-day and 20-day log return, 20-day Parkinson
  volatility, 10-day average candle body-to-range ratio, and 10-day sign-flip rate.
  The features are z-scored using only the statistics of each training window.
- **Stock universe:** CRSP daily returns of S&P 500 constituents, using point-in-time
  index membership.
- **Momentum sleeve:** long the top 50 stocks by a blended 12/6/3-month momentum score
  (z-scores weighted 1 / 0.5 / 0.25), rebalanced weekly.
- **Mean-reversion sleeve:** each day, long the 50 worst and short the 50 best
  5-day performers, with each formation held for 3 days (three overlapping portfolios).
- **HMM:** 2-state Gaussian HMM with full covariance (`hmmlearn`). Daily allocation
  uses a forward filter written for this project.
- **HSMM:** 2-state Gaussian HSMM with shifted-Poisson sojourn times (`mhsmm` in R).
  The embedded transition matrix has a zero diagonal, so all persistence comes from
  the duration distribution.
- **Backtest:** initial training on 2002–2005; allocations start on 2006-01-03. Both
  models are re-estimated every 21 trading days on all data up to t−1. Day t's
  allocation uses only the filtered state probability at t−1, and the state with the
  higher mean daily return is labelled "trend".

### Look-ahead and bias controls

- Feature scaling, model fitting and regime labelling only ever see data up to t−1.
- The HSMM regime for day t comes from a sequence that *ends* at t−1. The smoothed
  probability at the last point of a sequence equals the filtered probability, so
  there is no smoothing look-ahead.
- Sleeve portfolios are formed at the close of day t and earn returns from t+1.
- The universe uses point-in-time S&P 500 membership, so there is no survivorship bias.

## Limitations

- Drawdowns stay large (about 50%) even after gating. During the Covid crash the gated
  portfolios fell 46% from peak to trough within the episode before recovering.
- A single market, a single pair of sleeves, two states, and one duration family
  (shifted Poisson). Financing costs and short-sale constraints are ignored.
- Few extreme episodes exist in the data, so the episode-level comparison is anecdotal.
- The mean-reversion sleeve's return is concentrated in time: 2008 and 2020 contribute
  about 60% of its summed daily returns, and without September 2008 to June 2009 its
  Sharpe ratio falls from 0.69 to 0.57.
- Stock returns are CRSP price returns excluding dividends (`DlyRetx`). This understates
  the long-only momentum sleeve by roughly the index dividend yield and leaves the
  long-short mean-reversion sleeve largely unaffected.

## Reproducing

Tested with Python 3.12 and R 4.5.2 (`mhsmm` 0.4.21). Run everything from the repository root.

```bash
pip install -r requirements.txt
Rscript -e 'install.packages("mhsmm")'

python src/hmm_backtest.py       # -> results/hmm_gated.csv      (~2 min)
Rscript src/hsmm_backtest.R      # -> results/hsmm_regimes.csv   (~7 min: 228 refits, daily re-filtering)
python src/hsmm_gate.py          # -> results/hsmm_gated.csv
python src/analysis.py           # -> results/tables/, results/figures/
python src/transaction_costs.py  # -> results/tables/transaction_costs.csv
python src/sharpe_tests.py       # -> results/tables/sharpe_tests.csv
```

The HMM and sleeve steps reproduce the committed results exactly. The committed
`hsmm_regimes.csv` comes from the original run, which used unseeded k-means to initialise
EM. The script now fixes the seed, and a re-run lands on different local optima in some
windows: 103 of 4,780 days (2.2%) get a different regime, and the HSMM-gated Sharpe moves
from 0.89 to 0.90. That initialisation noise is more than half the HMM–HSMM gap
(0.91 vs 0.89), which reinforces the conclusion that the two models are practically
indistinguishable here.

Optional upstream steps: `src/features.py` re-downloads SPY data, and
`src/build_sleeves.py` rebuilds the sleeve returns and turnover. The second needs a
licensed CRSP extract; see [data/README.md](data/README.md).

## Repository layout

```
report/    the paper
src/       features, sleeves, HMM backtest (Python), HSMM backtest (R), analysis, cost and significance tests
data/      SPY features, sleeve returns and turnover (CRSP inputs not included)
results/   gated return series, tables and figures (file names follow the paper's exhibit numbers)
```
