"""Daily demand on a real wholesaler's ledger, scored against doing nothing.

uv run python projects/09-forecast-backtest/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from backtest import (  # noqa: E402
    drifting_mean,
    evaluate,
    rolling_origin,
    seasonal_mean,
    seasonal_naive,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, caption, money_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
HORIZON = 7
SEASON = 7


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    ledger = online_retail()
    daily = ledger.set_index("InvoiceDate")["revenue"].resample("D").sum().astype(float)
    # The business does not trade on Saturdays; those zeroes are closures, not demand of
    # zero, and leaving them in makes every model look like it fails one day in seven.
    closed = daily == 0
    daily = daily[~closed]
    print(f"\n{len(daily):,} trading days, {daily.index[0].date()} to {daily.index[-1].date()}")
    print(f"  {int(closed.sum())} zero-revenue days removed as closures")
    print(f"  daily revenue: median GBP {daily.median():,.0f}, max GBP {daily.max():,.0f}\n")

    models = {
        "Seasonal naive (last week)": lambda h, n: seasonal_naive(h, n, season=SEASON),
        "28-day mean": lambda h, n: drifting_mean(h, n, window=28),
        "Weekday profile (last 8)": lambda h, n: seasonal_mean(h, n, season=SEASON, lookback=8),
    }

    scores = evaluate(daily, models, horizon=HORIZON, step=HORIZON, min_train=180, season=SEASON)
    table = pd.DataFrame([s.summary() for s in scores])
    print(table.to_string(index=False))

    winner = scores[0]
    print(f"\n  Best: {winner.name} at MASE {winner.mase:.3f} over {winner.folds} folds")
    if winner.beats_naive:
        print(f"  It beats repeating last week by {(1 - winner.mase):.1%}.")
    else:
        print("  Nothing here beats repeating last week.")

    # --- MAPE vs MASE disagree ---------------------------------------------
    by_mape = table.sort_values("mape")
    print(f"\n  ranked by MASE: {list(table['model'])}")
    print(f"  ranked by MAPE: {list(by_mape['model'])}")

    # --- 1. the series -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily.to_numpy(), lw=0.8, color="#444444")
    ax.plot(
        daily.index,
        daily.rolling(28).mean().to_numpy(),
        lw=2,
        color=PALETTE["blue"],
        label="28-day mean",
    )
    ax.set_ylabel("Daily revenue")
    money_axis(ax, symbol="GBP ")
    ax.legend()
    ax.set_title("Two years of daily revenue, with a hard weekly cycle and a Christmas peak")
    caption(
        fig,
        f"UCI Online Retail II, {len(daily):,} trading days. Saturdays excluded: the "
        "business is closed and a zero is not a forecast failure.",
    )
    print("\n  " + str(save(fig, FIGURES / "01-the-series.png").relative_to(HERE.parent.parent)))

    # --- 2. MASE, with the line that matters ------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = table.sort_values("mase", ascending=False)
    colours = [PALETTE["green"] if m < 1 else PALETTE["red"] for m in order["mase"]]
    bars = ax.barh(order["model"], order["mase"], color=colours)
    ax.axvline(1.0, color="#444444", lw=2)
    ax.text(1.02, -0.45, "doing nothing", fontsize=9, color="#444444", fontweight="bold")
    for bar, v in zip(bars, order["mase"], strict=True):
        ax.text(
            v + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_xlabel("MASE — mean absolute error / seasonal naive error")
    ax.set_title("Everything is measured against repeating last week")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        f"Rolling origin: {winner.folds} folds, {HORIZON}-day horizon, expanding "
        "training window starting at 180 days.",
    )
    print("  " + str(save(fig, FIGURES / "02-mase.png").relative_to(HERE.parent.parent)))

    # --- 3. MAPE is the wrong ruler ---------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    left.barh(table["model"], table["mase"], color=PALETTE["blue"])
    left.axvline(1.0, color="#444444", lw=1.6)
    left.set_title("Ranked by MASE")
    left.set_xlabel("MASE")
    left.grid(axis="x")
    left.grid(axis="y", visible=False)
    right.barh(by_mape["model"], by_mape["mape"], color=PALETTE["orange"])
    right.set_title("Ranked by MAPE")
    right.set_xlabel("MAPE")
    right.grid(axis="x")
    right.grid(axis="y", visible=False)
    fig.suptitle(
        "MAPE divides by the actual, so quiet days dominate the average",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "A forecast can improve its MAPE by systematically under-predicting; MASE "
        "has no such loophole.",
    )
    print("  " + str(save(fig, FIGURES / "03-mape-vs-mase.png").relative_to(HERE.parent.parent)))

    # --- 4. one fold, drawn ------------------------------------------------
    folds = list(rolling_origin(daily, horizon=HORIZON, step=HORIZON, min_train=180))
    train, test = folds[len(folds) // 2]
    tail = train.iloc[-28:]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(tail.index, tail.to_numpy(), lw=1.6, color="#444444", label="history the model saw")
    ax.plot(test.index, test.to_numpy(), lw=2.4, color=PALETTE["black"], label="what happened")
    for (name, model), colour in zip(
        models.items(), (PALETTE["blue"], PALETTE["orange"], PALETTE["green"]), strict=True
    ):
        ax.plot(test.index, model(train, len(test)), "o--", lw=1.6, ms=4, color=colour, label=name)
    ax.axvline(test.index[0], color="#cccccc", lw=1.2)
    ax.set_ylabel("Daily revenue")
    money_axis(ax, symbol="GBP ")
    ax.legend(fontsize=8)
    ax.set_title(f"One fold: forecasting {test.index[0].date()} to {test.index[-1].date()}")
    caption(fig, "Every fold fits only on data to the left of the line.")
    print("  " + str(save(fig, FIGURES / "04-one-fold.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "trading_days": int(len(daily)),
                "closures_removed": int(closed.sum()),
                "horizon": HORIZON,
                "folds": winner.folds,
                "scores": [s.summary() for s in scores],
                "ranked_by_mase": list(table["model"]),
                "ranked_by_mape": list(by_mape["model"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
