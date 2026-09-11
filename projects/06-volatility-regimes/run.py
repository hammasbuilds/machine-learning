"""35 years of VIX: regimes you can see afterwards, and regimes you could have seen.

uv run python projects/06-volatility-regimes/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from regimes import (  # noqa: E402
    CALM,
    REGIME_NAMES,
    STRESSED,
    compare,
    hindsight_labels,
    persistence,
    realtime_labels,
    runs,
    transition_matrix,
)

from shared.data import VIX_DAILY, fetch_csv  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
REGIME_COLOURS = {CALM: PALETTE["green"], 1: "#e8e8e8", STRESSED: PALETTE["red"]}


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = fetch_csv(VIX_DAILY, "cboe-vix-daily.csv", parse_dates=["DATE"])
    frame = frame.dropna(subset=["CLOSE"]).sort_values("DATE").reset_index(drop=True)
    vix = frame.set_index("DATE")["CLOSE"].astype(float)
    print(f"\nVIX: {len(vix):,} trading days, {vix.index[0].date()} to {vix.index[-1].date()}")
    print(f"  level: min {vix.min():.2f}  median {vix.median():.2f}  max {vix.max():.2f}\n")

    after = hindsight_labels(vix)
    before = realtime_labels(vix, warmup=500)
    comparable = before >= 0

    print(
        "  regime mix (hindsight): "
        + "  ".join(f"{REGIME_NAMES[r]} {np.mean(after == r):.1%}" for r in range(3))
    )
    print(
        "  regime mix (real time): "
        + "  ".join(f"{REGIME_NAMES[r]} {np.mean(before[comparable] == r):.1%}" for r in range(3))
    )

    # --- persistence -------------------------------------------------------
    matrix = transition_matrix(after)
    print(f"\n  persistence: {persistence(after):.1%} of days share yesterday's regime")
    print("  transition matrix P(tomorrow | today):")
    print("      " + "".join(f"{n:>11}" for n in REGIME_NAMES))
    for i, name in enumerate(REGIME_NAMES):
        print(f"    {name:<9}" + "".join(f"{matrix[i, j]:>10.1%} " for j in range(3)))

    # --- the disagreement --------------------------------------------------
    difference = compare(after, before, window=10)
    print(f"\n  {json.dumps(difference.summary())}")
    print(f"\n  The two labellings disagree on {difference.rate:.1%} of days overall,")
    print(f"  but on {difference.rate_at_turning_points:.1%} of days near a regime change")
    print(f"  - {difference.concentration}x more likely exactly where the label would matter.")

    stretches = runs(after)
    longest = stretches.sort_values("length", ascending=False).head(3)
    print("\n  longest hindsight regimes:")
    for _, r in longest.iterrows():
        start = vix.index[int(r["start"])].date()
        print(f"    {REGIME_NAMES[int(r['regime'])]:<9} {int(r['length']):>4} days from {start}")

    # --- 1. the chart everybody publishes ---------------------------------
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot(vix.index, vix.to_numpy(), lw=0.7, color="#333333")
    for _, r in runs(after).iterrows():
        s, e = int(r["start"]), int(r["start"] + r["length"]) - 1
        top.axvspan(
            vix.index[s],
            vix.index[min(e, len(vix) - 1)],
            color=REGIME_COLOURS[int(r["regime"])],
            alpha=0.30,
            lw=0,
        )
    top.set_ylabel("VIX")
    top.set_title("Regimes fitted with full hindsight — the version that gets published")
    top.set_yscale("log")

    disagree = np.zeros(len(vix))
    disagree[comparable & (after != before)] = 1
    bottom.fill_between(vix.index, 0, disagree, color=PALETTE["purple"], lw=0)
    bottom.set_ylabel("disagrees")
    bottom.set_yticks([])
    bottom.set_title("Days where a real-time label would have said something else", fontsize=10)
    caption(
        fig,
        f"CBOE VIX daily close, {len(vix):,} days. Green = calm, red = stressed. "
        f"Real-time labels use an expanding window with a 500-day warm-up.",
    )
    print(
        "\n  "
        + str(save(fig, FIGURES / "01-regimes-in-hindsight.png").relative_to(HERE.parent.parent))
    )

    # --- 2. where the disagreement lives ----------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(
        ["Any day", "Within 10 days\nof a regime change"],
        [difference.rate, difference.rate_at_turning_points],
        color=[PALETTE["sky"], PALETTE["red"]],
        width=0.5,
    )
    for bar, v in zip(bars, (difference.rate, difference.rate_at_turning_points), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontweight="bold"
        )
    percent_axis(ax)
    ax.set_ylabel("Share of days the two labellings disagree")
    ax.set_title(
        f"The disagreement is {difference.concentration}x concentrated where the label matters"
    )
    caption(
        fig,
        "In the calm middle of a regime both methods agree, and neither is useful. "
        "At the turning points they part company.",
    )
    print(
        "  " + str(save(fig, FIGURES / "02-where-it-disagrees.png").relative_to(HERE.parent.parent))
    )

    # --- 3. transition matrix ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), REGIME_NAMES)
    ax.set_yticks(range(3), REGIME_NAMES)
    ax.set_xlabel("tomorrow")
    ax.set_ylabel("today")
    ax.grid(visible=False)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if matrix[i, j] > 0.5 else "#333333",
            )
    ax.set_title(f"{persistence(after):.0%} of days repeat yesterday")
    caption(
        fig, "A forecast of 'tomorrow is like today' is right 97% of the time and says nothing."
    )
    print("  " + str(save(fig, FIGURES / "03-transitions.png").relative_to(HERE.parent.parent)))

    # --- 4. how late is the live label? -----------------------------------
    #
    # Two hypotheses went into this project and both were wrong, so the analysis is split
    # to show why. Counting *every* stressed run mixes 170 episodes, most of which are
    # one- and two-day flickers that any method catches instantly and nobody trades. The
    # question worth asking is about the episodes that mattered.
    MAJOR = 20  # trading days: about a month

    def lag_for(minimum_length: int) -> list[int]:
        found = []
        for _, r in runs(after).iterrows():
            if int(r["regime"]) != STRESSED or int(r["length"]) < minimum_length:
                continue
            s = int(r["start"])
            if s < 500:
                continue
            ahead = before[s : s + 120]
            hit = np.flatnonzero(ahead == STRESSED)
            found.append(int(hit[0]) if len(hit) else 120)
        return found

    lags = lag_for(1)
    major_lags = lag_for(MAJOR)
    print(
        f"\n  stressed episodes of any length : {len(lags):>3}, median lag "
        f"{np.median(lags):.0f} trading days"
    )
    print(
        f"  episodes lasting {MAJOR}+ days        : {len(major_lags):>3}, median lag "
        f"{np.median(major_lags):.0f} trading days"
    )
    print("\n  Both hypotheses this project started with were wrong. Real-time labelling of")
    print("  VIX works: 89.5% agreement with hindsight, and no median delay on the episodes")
    print("  that lasted. VIX is an observed level, not a latent state that has to be inferred.")

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bins = range(0, 46, 2)
    ax.hist(
        lags,
        bins=bins,
        color=PALETTE["sky"],
        alpha=0.75,
        label=f"all {len(lags)} stressed episodes",
    )
    ax.hist(
        major_lags,
        bins=bins,
        color=PALETTE["orange"],
        alpha=0.9,
        label=f"{len(major_lags)} episodes lasting {MAJOR}+ days",
    )
    median_lag = float(np.median(major_lags)) if major_lags else 0.0
    ax.axvline(median_lag, color=PALETTE["red"], lw=2)
    ax.text(
        median_lag + 0.6,
        ax.get_ylim()[1] * 0.8,
        f"  median {median_lag:.0f} days",
        color=PALETTE["red"],
        fontweight="bold",
        fontsize=9,
    )
    ax.set_xlabel("Trading days before the real-time label caught up")
    ax.set_ylabel("Episodes")
    ax.legend()
    ax.set_title("The live label is not late. This is the result I expected to be wrong.")
    caption(
        fig,
        "Measured from the first day of each hindsight-labelled stressed regime to "
        "the first day the expanding-window label agreed.",
    )
    print("  " + str(save(fig, FIGURES / "04-how-late.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "days": int(len(vix)),
                "from": str(vix.index[0].date()),
                "to": str(vix.index[-1].date()),
                "vix": {
                    "min": round(float(vix.min()), 2),
                    "median": round(float(vix.median()), 2),
                    "max": round(float(vix.max()), 2),
                },
                "persistence": round(persistence(after), 4),
                "transition_matrix": [[round(float(v), 4) for v in row] for row in matrix],
                **difference.summary(),
                "stress_episodes_any_length": len(lags),
                "stress_episodes_20_days_plus": len(major_lags),
                "median_days_late_all": float(np.median(lags)),
                "median_days_late_major": median_lag,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
