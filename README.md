# machine-learning

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
![tests](https://img.shields.io/badge/tests-121%20passing-success)
![data](https://img.shields.io/badge/data-public%20only-success)
![license](https://img.shields.io/badge/license-MIT-green)

**Ten machine-learning projects on public data. Each one is built around the mistake that
makes its answer wrong.**

Every dataset is real, public, and downloadable without an account. Every figure states its
source and its sample size. Nothing here is a tutorial.

---

## The through-line

Nine of these ten projects fail in the same way, and it is not a modelling failure:

> **The number was computed correctly and the question was miscounted.**

| | Project | What was counted | What was actually independent |
|---|---|---|---|
| 01 | Backtest overfitting | one strategy's Sharpe | 846 strategies searched |
| 05 | Target leakage | 45,211 calls | 45,211 calls *and* a column from the future |
| 08 | Overlapping windows | 1,593 monthly windows | 13 decades |
| 03 | Imbalanced failure | 99% accuracy | a 3.4% base rate |
| 10 | Price elasticity | 63,011 product-months | 3,739 different products |

In every case the arithmetic downstream was perfect.

## The ten

### Quant and trading

**[01 · Backtest overfitting](projects/01-backtest-overfitting/)**
846 trading rules on the real S&P 500 found a winner at Sharpe 0.620. The same 846 rules on
**shuffled** S&P 500 found a better one at 0.695. Deflated Sharpe Ratio says the
probability the edge is real is **23.5%**.

**[06 · Volatility regimes](projects/06-volatility-regimes/)**
35 years of VIX. Set out to prove hindsight bias in regime labelling; **both hypotheses
failed** — real-time labelling agrees with hindsight 89.5% of the time and catches stress
episodes with zero median delay. Reported as measured.

**[08 · Overlapping windows](projects/08-overlapping-windows/)** ⭐
Does valuation predict returns? Naive OLS: **t = −13.75**. Disjoint windows: **t = −1.15**.
Same data, same slope. The inflation factor is 12.0; `sqrt(120) = 11.0`.

### Industry and operations

**[03 · Imbalanced failure prediction](projects/03-imbalanced-maintenance/)**
10,000 machine cycles, 3.4% failures. A model that predicts "no failure" scores **97.97%
accuracy**. ROC-AUC calls two models close (0.902 / 0.930); average precision says one is
twice as good (0.405 / 0.757). Cost-based threshold saves 31%.

**[09 · Forecast backtesting](projects/09-forecast-backtest/)**
604 trading days, 60 rolling folds. A **flat 28-day mean beats both models that know about
the weekly cycle** — at a 7-day horizon the weekday shape averages away. Every model runs
systematically low, which no accuracy metric shows.

### Marketing and customers

**[04 · Customer value](projects/04-customer-value/)**
1,055,823 invoice lines. Mean value £2,844, median £855 — **the mean sits at the 80th
percentile**. The top 1% of customers produce 31% of revenue; 24% bought exactly once.

**[05 · Target leakage](projects/05-leakage/)** ⭐
The published benchmark for this dataset is 0.93 ROC-AUC. **Fix two things and it is 0.63.**
One column that does not exist at decision time (−0.11), and a random split that shuffles
time (−0.18).

**[07 · Segmentation stability](projects/07-segmentation-stability/)**
Do the five customer segments survive a reseed? Shuffle every feature independently —
destroying all structure, keeping all distributions — and k-means still scores a silhouette
of **0.25**. Real data scores 0.33. Most of the score is skew.

**[10 · Price elasticity](projects/10-price-elasticity/)**
Pooled estimate: **−0.77, inelastic, raise prices.** Within-product: **−1.97, elastic,
raising prices costs you money.** Same ledger, opposite instruction.

### Sport

**[02 · Sports forecasting](projects/02-sports-calibration/)**
3,420 Premier League matches against Pinnacle closing odds. Elo beats the base rate on every
metric — and its value bets return **−3.38%**, worse than blindly backing favourites at
−1.53%.

## Data

Seven public sources, all verified reachable on 2026-09-11, all cached after first fetch
with a SHA-256 recorded in `data/sources.json`.

| Source | Size | Licence |
|---|---|---|
| Daily closes, 5 tickers (`plotly/datasets`) | 2,306 days | MIT |
| CBOE VIX daily since 1990 | 9,269 days | public |
| Shiller S&P 500 monthly since 1871 | 1,868 months | public (Yale) |
| football-data.co.uk, EPL ×9 seasons | 3,420 matches | free w/ attribution |
| UCI AI4I 2020 predictive maintenance | 10,000 cycles | CC BY 4.0 |
| UCI Online Retail II | 1,055,823 lines | CC BY 4.0 |
| UCI Bank Marketing | 45,211 calls | CC BY 4.0 |

Three planned sources were **dropped rather than left in as aspiration**: Stooq and FRED are
now behind JavaScript browser challenges, and `files.grouplens.org` refused connections.
`uv run python shared/data.py --check` re-verifies every URL and reports which have gone.

## Running it

```bash
uv sync --all-groups
uv run pytest -q                                    # 121 tests, no downloads
uv run python projects/01-backtest-overfitting/run.py
```

Each project is self-contained: `run.py` fetches what it needs, prints its findings, writes
its figures and a `result.json`. No GPU, no API key, no notebook.

## Layout

```
shared/
  plotting.py     one visual language - Okabe-Ito palette, mandatory source captions
  validation.py   time splits, purged splits, group splits, cost thresholds, calibration
  data.py         public-dataset catalogue with provenance recording
projects/01..10/
  <module>.py     the method, with the trap documented
  run.py          fetch, compute, draw, report
  README.md       the finding, and what it cost to get right
  figures/        every chart, captioned
tests/            121 tests
```

## On charts

One style, defined once in `shared/plotting.py`, because ten charts drawn ten ways read as
ten weekend projects.

- Okabe-Ito palette — roughly one man in twelve cannot separate matplotlib's default red
  from its green
- **Every figure carries a caption naming the source and the sample size.** A chart without
  `n` cannot be argued with
- Gridlines behind the data, horizontal only; no top or right spine
- Counts kept visible in binned plots, so a bin holding nine samples is not drawn at the
  same weight as one holding nine hundred

## Where I was wrong

Each project's README has a *Problems hit while building this* section. The ones worth
naming here:

- **01** — shipped a units bug. Fed an *annualised* Sharpe to a formula expecting the
  per-period one; the answer was wrong by 252× in the direction that makes an unproven
  strategy look proven. Caught because "15 days" was implausible, not because anything
  crashed.
- **06** — two hypotheses, both wrong, neither rescued by tuning the parameters until
  something appeared.
- **07** — expected "segments are fake"; the data says they are real, thin, and exhausted
  by about k=6.
- **09** — predicted MAPE and MASE would rank models differently. They didn't, and the
  README says so.
- **10** — expected the pooled elasticity to come out positive. It didn't — it came out
  *plausible and wrong*, which does more damage.
- **02** — oversold a trap and cut it back: removing the bookmaker's overround moves Brier
  by 0.0001, not the headline I'd assumed.

## License

MIT. Datasets retain their own licences, recorded per-source above and in
`data/sources.json`.
