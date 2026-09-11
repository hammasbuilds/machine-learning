"""Run the search, deflate the winner, draw the four charts.

uv run python projects/01-backtest-overfitting/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from deflated import (  # noqa: E402
    deflated_sharpe,
    minimum_track_record_length,
    sharpe,
)
from search import rule_grid, run_search, shuffled_prices, strategy_returns  # noqa: E402

from shared.data import STOCK_DAILY, fetch_csv  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"
TICKER = "GSPC"  # the S&P 500 index
SOURCE_NOTE = (
    "Source: plotly/datasets daily closes, 2007-01-03 to 2016-03-01 (n=2,306 trading days)"
)


def load_prices() -> pd.Series:
    frame = fetch_csv(STOCK_DAILY, "stock-daily-5tickers.csv", parse_dates=["Date"])
    return frame.set_index("Date")[TICKER].astype(float).sort_index()


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    prices = load_prices()
    rules = rule_grid()
    print(
        f"{TICKER}: {len(prices):,} daily closes, {prices.index[0].date()} to {prices.index[-1].date()}"
    )
    print(f"searching {len(rules):,} strategies ...\n")

    real = run_search(prices, rules)
    null = run_search(shuffled_prices(prices, seed=7), rules)

    best = real.iloc[0]
    best_rule = next(r for r in rules if r.name == best["rule"])
    best_returns = strategy_returns(prices, best_rule).to_numpy()

    buy_and_hold = prices.pct_change().fillna(0.0).to_numpy()
    bh_sharpe = sharpe(buy_and_hold)

    verdict = deflated_sharpe(
        best_returns, n_trials=len(rules), all_trial_sharpes=real["sharpe"].to_numpy()
    )

    # --- the finding ------------------------------------------------------
    print(
        f"  best of {len(rules):,} real strategies : {best['rule']:<22} Sharpe {best['sharpe']:.3f}"
    )
    print(
        f"  best of {len(rules):,} on SHUFFLED data: {null.iloc[0]['rule']:<22} Sharpe {null.iloc[0]['sharpe']:.3f}"
    )
    print(f"  buy and hold                          : {'':<22} Sharpe {bh_sharpe:.3f}\n")
    print("  " + verdict.sentence())
    print("\n  " + json.dumps(verdict.summary()))

    needed = minimum_track_record_length(
        verdict.per_period_sharpe, skew=verdict.skew, kurtosis=verdict.kurtosis
    )
    print(
        f"\n  minimum track record for this Sharpe to beat zero: {needed:,.0f} days "
        f"({needed / 252:.1f} years). We have {len(prices):,} ({len(prices) / 252:.1f} years)."
    )

    # --- 1. the distribution that makes the point -------------------------
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(
        min(real["sharpe"].min(), null["sharpe"].min()),
        max(real["sharpe"].max(), null["sharpe"].max()),
        60,
    )
    ax.hist(
        null["sharpe"],
        bins=bins,
        color=PALETTE["orange"],
        alpha=0.55,
        label="shuffled data (no edge exists)",
    )
    ax.hist(real["sharpe"], bins=bins, color=PALETTE["blue"], alpha=0.55, label="real S&P 500")
    ax.axvline(best["sharpe"], color=PALETTE["blue"], lw=2)
    ax.axvline(null.iloc[0]["sharpe"], color=PALETTE["orange"], lw=2, ls="--")
    annotate(
        ax,
        best["sharpe"],
        ax.get_ylim()[1] * 0.82,
        f"best real\n{best['sharpe']:.2f}",
        color=PALETTE["blue"],
    )
    annotate(
        ax,
        null.iloc[0]["sharpe"],
        ax.get_ylim()[1] * 0.55,
        f"best on noise\n{null.iloc[0]['sharpe']:.2f}",
        dx=-70,
        color=PALETTE["orange"],
    )
    ax.set_title(
        f"Searching {len(rules):,} strategies finds a winner even when there is nothing to find"
    )
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_ylabel("Strategies")
    ax.legend(loc="upper left")
    caption(fig, SOURCE_NOTE + ". Shuffled series keeps every return, destroys their order.")
    print(
        "\n  "
        + str(save(fig, FIGURES / "01-search-distribution.png").relative_to(HERE.parent.parent))
    )

    # --- 2. the deflation bar ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4))
    labels = [
        "Buy and hold",
        f"Best of {len(rules):,}\n(what gets published)",
        "Bar the best of\nthat many must clear",
    ]
    values = [bh_sharpe, verdict.observed_sharpe, verdict.benchmark_sharpe]
    colours = [PALETTE["sky"], PALETTE["blue"], PALETTE["red"]]
    bars = ax.bar(labels, values, color=colours, width=0.58)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold"
        )
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title(f"After deflation: {verdict.probability_real:.0%} probability the edge is real")
    caption(
        fig,
        f"Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). "
        f"{len(rules):,} trials, {verdict.n_observations:,} observations, "
        f"skew {verdict.skew:.2f}, kurtosis {verdict.kurtosis:.2f}.",
    )
    print("  " + str(save(fig, FIGURES / "02-deflation.png").relative_to(HERE.parent.parent)))

    # --- 3. the equity curve that sells it, split in half -----------------
    half = len(prices) // 2
    equity = pd.Series((1 + best_returns).cumprod(), index=prices.index)
    market = pd.Series((1 + buy_and_hold).cumprod(), index=prices.index)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(
        equity.index,
        equity.to_numpy(),
        color=PALETTE["blue"],
        lw=1.6,
        label=f"best strategy: {best['rule']}",
    )
    ax.plot(
        market.index, market.to_numpy(), color=MUTED_LINE, lw=1.4, ls="--", label="buy and hold"
    )
    ax.axvline(prices.index[half], color=PALETTE["red"], lw=1.2)
    in_s = sharpe(best_returns[:half])
    out_s = sharpe(best_returns[half:])
    ax.text(
        prices.index[half],
        ax.get_ylim()[1] * 0.97,
        f"  chosen on this half (Sharpe {in_s:.2f})  |  held out (Sharpe {out_s:.2f})",
        fontsize=9,
        va="top",
        color=PALETTE["red"],
        fontweight="bold",
    )
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.set_title("The same rule, before and after the half it was selected on")
    caption(fig, SOURCE_NOTE + ". Costs of 5bp charged on every position change.")
    print("  " + str(save(fig, FIGURES / "03-equity-split.png").relative_to(HERE.parent.parent)))

    # --- 4. how the bar rises with the number of trials -------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    from deflated import expected_max_sharpe

    variance = float(np.var(real["sharpe"].to_numpy(), ddof=1))
    trials = np.unique(np.logspace(0, 4.2, 60).astype(int))
    bar = [expected_max_sharpe(int(t), variance) for t in trials]
    ax.plot(trials, bar, color=PALETTE["red"], lw=2)
    ax.axhline(verdict.observed_sharpe, color=PALETTE["blue"], ls="--", lw=1.4)
    ax.axvline(len(rules), color=PALETTE["purple"], ls=":", lw=1.4)
    annotate(
        ax,
        len(rules),
        min(bar) + 0.05,
        f"  this search\n  {len(rules):,} trials",
        color=PALETTE["purple"],
    )
    annotate(
        ax,
        1.2,
        verdict.observed_sharpe,
        f"observed {verdict.observed_sharpe:.2f}",
        dy=6,
        color=PALETTE["blue"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of strategies tried (log scale)")
    ax.set_ylabel("Sharpe the best one reaches by luck alone")
    ax.set_title("Every extra strategy you try raises the bar your winner has to clear")
    caption(
        fig,
        "Expected maximum of N draws, from the observed spread of Sharpe ratios in this search.",
    )
    print("  " + str(save(fig, FIGURES / "04-bar-vs-trials.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "ticker": TICKER,
                "observations": int(len(prices)),
                "trials": len(rules),
                "best_rule": best["rule"],
                "best_sharpe_real": round(float(best["sharpe"]), 4),
                "best_sharpe_shuffled": round(float(null.iloc[0]["sharpe"]), 4),
                "buy_and_hold_sharpe": round(bh_sharpe, 4),
                "in_sample_sharpe": round(in_s, 4),
                "out_of_sample_sharpe": round(out_s, 4),
                "minimum_track_record_days": round(needed, 1),
                **verdict.summary(),
            },
            indent=2,
        )
    )


MUTED_LINE = "#9a9a9a"

if __name__ == "__main__":
    main()
