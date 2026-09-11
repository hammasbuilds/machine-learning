"""155 years of monthly S&P 500, and the question that decides every forecast.

    uv run python projects/17-time-series/run.py
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

from timeseries import (  # noqa: E402
    Arima,
    Drift,
    Holt,
    Mean,
    RandomWalk,
    rolling_backtest,
    spurious_regression,
    stationarity_verdict,
)

from shared.data import SHILLER_SP500, fetch_csv  # noqa: E402
from shared.plotting import PALETTE, caption, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
HORIZON = 12  # months
FOLDS = 40


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = fetch_csv(SHILLER_SP500)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame[frame["Real Price"] > 0].reset_index(drop=True)

    price = frame["Real Price"].to_numpy(dtype=float)
    log_price = np.log(price)
    returns = np.diff(log_price)
    cpi = frame["Consumer Price Index"].to_numpy(dtype=float)

    print(f"\nShiller S&P 500, {len(frame):,} months, "
          f"{frame['Date'].min():%Y-%m} to {frame['Date'].max():%Y-%m}")
    print("  Real Price is inflation-adjusted, so the trend is not a price-level artefact.\n")

    # --- 1. the two tests, and what they say about each other ---------------
    print("  === ADF and KPSS test opposite nulls, and often disagree ===")
    print(f"  {'series':<26}{'ADF p':>9}{'KPSS p':>9}   {'ADF':>16}{'KPSS':>16}   verdict")
    candidates = {
        "Real price (level)": price,
        "Log real price": log_price,
        "Log return (1st diff)": returns,
        "CPI (level)": cpi,
        "CPI inflation (diff)": np.diff(np.log(cpi)),
    }
    verdicts = [stationarity_verdict(v, name) for name, v in candidates.items()]
    for v in verdicts:
        print(f"  {v.series:<26}{v.adf_p:>9.4f}{v.kpss_p:>9.4f}   "
              f"{'stationary' if v.adf_says_stationary else 'unit root':>16}"
              f"{'stationary' if v.kpss_says_stationary else 'not stationary':>16}   {v.verdict}")

    disagreements = [v for v in verdicts if not v.agree]
    print(f"\n    {len(disagreements)} of {len(verdicts)} series: the two tests disagree.")
    if disagreements:
        print(f"    {', '.join(v.series for v in disagreements)}: neither 'stationary' nor")
        print("    'not stationary' is a defensible conclusion, and the analyst picks one anyway.\n")

    # --- 2. spurious regression ---------------------------------------------
    print("  === regressing two independent random walks on each other ===")
    spurious = spurious_regression(len(price), trials=300, seed=7)
    summary = spurious.groupby("basis").agg(
        significant=("significant", "mean"),
        median_r2=("r_squared", "median"),
        median_abs_t=("abs_t", "median"),
        max_r2=("r_squared", "max"),
    )
    print(f"  {'regressed on':<16}{'p<0.05':>10}{'median R2':>12}{'median |t|':>12}{'max R2':>10}")
    for basis in ("levels", "differences"):
        row = summary.loc[basis]
        print(f"  {basis:<16}{row['significant']:>10.1%}{row['median_r2']:>12.3f}"
              f"{row['median_abs_t']:>12.2f}{row['max_r2']:>10.3f}")
    levels_rate = float(summary.loc["levels", "significant"])
    diff_rate = float(summary.loc["differences", "significant"])
    print(f"\n    300 pairs of series with no relationship whatsoever. In levels, "
          f"{levels_rate:.0%} come back\n    significant. In differences, {diff_rate:.0%} - "
          f"which is what a 5% test is supposed to give.\n")

    # --- 3. does any model beat the last value? -----------------------------
    print(f"  === {FOLDS} expanding-window folds, {HORIZON}-month horizon, log real price ===")
    models = [RandomWalk(), Drift(), Mean(24), Arima((1, 1, 1)), Arima((2, 1, 2)), Holt()]
    backtest = rolling_backtest(
        log_price, models, horizon=HORIZON, folds=FOLDS, minimum=240
    )
    scores = (
        backtest.groupby("model")
        .agg(mase=("mase", "mean"), median_mase=("mase", "median"), bias=("bias", "mean"))
        .sort_values("mase")
    )
    naive = float(scores.loc["Random walk (last value)", "mase"])

    print(f"  {'model':<26}{'MASE':>8}{'median':>9}{'bias':>9}   vs naive")
    for name, row in scores.iterrows():
        delta = row["mase"] / naive - 1
        verdict = "worse" if delta > 0.01 else ("better" if delta < -0.01 else "same")
        print(f"  {name:<26}{row['mase']:>8.3f}{row['median_mase']:>9.3f}{row['bias']:>9.3f}"
              f"   {delta:+7.1%}  {verdict}")

    winner = scores.index[0]
    beaten = [n for n in scores.index if scores.loc[n, "mase"] < naive and n != "Random walk (last value)"]
    print(f"\n    best: {winner}. "
          f"{len(beaten)} of {len(models) - 1} models beat predicting the last value.")

    # --- 4. what happens on a stationary series -----------------------------
    print("\n  === the same models on the *differenced* series ===")
    diff_backtest = rolling_backtest(returns, models, horizon=HORIZON, folds=FOLDS, minimum=240)
    diff_scores = diff_backtest.groupby("model")["mase"].mean().sort_values()
    diff_naive = float(diff_scores.loc["Random walk (last value)"])
    for name, value in diff_scores.items():
        print(f"  {name:<26}{value:>8.3f}   {value / diff_naive - 1:+7.1%}")
    print(f"\n    On returns the ranking inverts: {diff_scores.index[0]} wins and the")
    print("    random walk is last. Same models, same data, one difference applied.")

    # --- figure 1: the series, and the two tests ----------------------------
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top.plot(frame["Date"], price, color=PALETTE["blue"], lw=1)
    top.set_yscale("log")
    top.set_ylabel("Real price (log)")
    top.set_title("A trending series: ADF and KPSS both call it non-stationary")
    bottom.plot(frame["Date"].iloc[1:], returns, color=PALETTE["orange"], lw=0.5)
    bottom.axhline(0, color="#333333", lw=0.8)
    bottom.set_ylabel("Monthly log return")
    bottom.set_title("One difference later: mean-reverting, and both tests agree it is stationary")
    fig.suptitle("Everything downstream depends on which of these you model",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    caption(fig, f"Shiller real S&P 500, {frame['Date'].min():%Y}-{frame['Date'].max():%Y}, "
                 f"{len(frame):,} months. Inflation-adjusted.")
    print("\n  " + str(save(fig, FIGURES / "01-the-series.png").relative_to(HERE.parent.parent)))

    # --- figure 2: spurious regression --------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    for basis, colour in (("levels", PALETTE["red"]), ("differences", PALETTE["blue"])):
        subset = spurious[spurious["basis"] == basis]
        left.hist(subset["abs_t"], bins=40, alpha=0.65, label=basis, color=colour)
        right.hist(subset["r_squared"], bins=40, alpha=0.65, label=basis, color=colour)
    left.axvline(1.96, color="#333333", ls="--", lw=1.5)
    left.text(2.2, left.get_ylim()[1] * 0.85, "t = 1.96\n(p = 0.05)", fontsize=8)
    left.set_xlabel("|t-statistic| on the slope"); left.set_ylabel("Trials")
    left.set_title(f"{levels_rate:.0%} of unrelated pairs look significant")
    left.legend(fontsize=8)
    right.set_xlabel("R-squared"); right.set_ylabel("Trials")
    right.set_title("and the fit looks convincing too")
    right.legend(fontsize=8)
    fig.suptitle("300 pairs of independent random walks. No relationship exists in any of them.",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    caption(fig, "Granger & Newbold (1974). Regressing non-stationary series in levels "
                 "manufactures significance out of nothing; differencing removes it.")
    print("  " + str(save(fig, FIGURES / "02-spurious.png").relative_to(HERE.parent.parent)))

    # --- figure 3: the backtest ---------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    order = scores.sort_values("mase", ascending=False)
    colours = [PALETTE["green"] if n == "Random walk (last value)" else PALETTE["blue"]
               for n in order.index]
    bars = left.barh(list(order.index), order["mase"], color=colours)
    left.axvline(naive, color=PALETTE["green"], ls="--", lw=1.5)
    for bar, value in zip(bars, order["mase"], strict=True):
        left.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                  va="center", fontsize=9, fontweight="bold")
    left.set_xlabel(f"MASE over {FOLDS} folds (lower is better)")
    left.set_title("On the level: nothing beats the last value")
    left.grid(axis="x"); left.grid(axis="y", visible=False)

    diff_order = diff_scores.sort_values(ascending=False)
    dcolours = [PALETTE["green"] if n == "Random walk (last value)" else PALETTE["orange"]
                for n in diff_order.index]
    bars = right.barh(list(diff_order.index), diff_order.to_numpy(), color=dcolours)
    right.axvline(diff_naive, color=PALETTE["green"], ls="--", lw=1.5)
    for bar, value in zip(bars, diff_order.to_numpy(), strict=True):
        right.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                   va="center", fontsize=9, fontweight="bold")
    right.set_xlabel("MASE on the differenced series")
    right.set_title("On returns: it is the worst thing you can do")
    right.set_yticklabels([])
    right.grid(axis="x"); right.grid(axis="y", visible=False)
    caption(fig, f"Expanding window, {HORIZON}-month horizon, minimum 240 months of history. "
                 f"Green = the naive benchmark.")
    print("  " + str(save(fig, FIGURES / "03-backtest.png").relative_to(HERE.parent.parent)))

    # --- figure 4: fold-by-fold instability ---------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.3))
    dates = frame["Date"].to_numpy()
    for name, colour in (("Random walk (last value)", PALETTE["green"]),
                         ("ARIMA(2, 1, 2)", PALETTE["blue"]),
                         ("Drift", PALETTE["red"])):
        rows = backtest[backtest["model"] == name]
        ax.plot(dates[rows["origin"].to_numpy()], rows["mase"], "o-", lw=1.5, ms=3,
                color=colour, label=name, alpha=0.85)
    ax.axhline(1.0, color="#333333", ls="--", lw=1)
    ax.set_ylabel(f"MASE at a {HORIZON}-month horizon"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("The mean hides the variance: any model wins on some folds")
    caption(fig, "Which model 'wins' depends on which decade you evaluate. Reporting the "
                 "best average over 40 folds is already a mild version of the search in "
                 "project 01.")
    print("  " + str(save(fig, FIGURES / "04-by-fold.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "months": int(len(frame)),
                "span": [str(frame["Date"].min().date()), str(frame["Date"].max().date())],
                "stationarity": [
                    {"series": v.series, "adf_p": round(v.adf_p, 5),
                     "kpss_p": round(v.kpss_p, 5), "agree": v.agree, "verdict": v.verdict}
                    for v in verdicts
                ],
                "spurious": {
                    "levels_significant": round(levels_rate, 4),
                    "differences_significant": round(diff_rate, 4),
                    "levels_median_r2": round(float(summary.loc["levels", "median_r2"]), 4),
                },
                "levels_backtest": [
                    {"model": n, "mase": round(float(r["mase"]), 4),
                     "bias": round(float(r["bias"]), 5)}
                    for n, r in scores.iterrows()
                ],
                "returns_backtest": [
                    {"model": n, "mase": round(float(v), 4)} for n, v in diff_scores.items()
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
