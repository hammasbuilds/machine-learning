"""Do the five customer segments survive being asked to repeat themselves?

uv run python projects/07-segmentation-stability/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "04-customer-value"))

from stability import (  # noqa: E402
    assess,
    fit_labels,
    inertia_curve,
    resample_agreement,
    reseed_agreement,
)
from value import rfm  # noqa: E402

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, caption, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
KS = range(2, 11)


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = rfm(online_retail())
    frame = frame[frame["monetary"] > 0]

    # Log-transform before scaling. RFM is heavily skewed and k-means minimises squared
    # distance, so on raw values a handful of whales define every centroid and the
    # "segments" become "one whale, and everyone else" repeated k times.
    features = pd.DataFrame(
        {
            "recency": np.log1p(frame["recency_days"]),
            "frequency": np.log1p(frame["frequency"]),
            "monetary": np.log1p(frame["monetary"]),
        }
    )
    X = StandardScaler().fit_transform(features.to_numpy())
    print(f"\n{len(X):,} customers, three features (log-scaled recency, frequency, monetary)\n")

    # --- the elbow plot, and what it cannot tell you -----------------------
    inertia = inertia_curve(X, KS)
    drops = -np.diff(inertia) / inertia[:-1]
    print("  elbow plot: inertia falls monotonically, as it must")
    for k, value, drop in zip(list(KS)[1:], inertia[1:], drops, strict=True):
        print(f"    k={k:<2} inertia {value:>9,.0f}   improvement over k={k - 1}: {drop:>5.1%}")

    # --- the interrogation -------------------------------------------------
    print(
        f"\n  {'k':<3}{'silhouette':>12}{'on shuffled':>13}{'excess':>9}"
        f"{'reseed ARI':>12}{'resample ARI':>14}   verdict"
    )
    verdicts = []
    for k in KS:
        verdict = assess(X, k, seed=k)
        verdicts.append(verdict)
        s = verdict.summary()
        print(
            f"  {k:<3}{s['silhouette']:>12.3f}{s['silhouette_shuffled']:>13.3f}"
            f"{s['excess_over_null']:>9.3f}{s['reseed_ari']:>12.3f}{s['resample_ari']:>14.3f}"
            f"   {s['verdict']}"
        )

    table = pd.DataFrame([v.summary() for v in verdicts])
    stable = table[table["verdict"] == "stable and beats null"]
    print(
        f"\n  values of k that are reproducible AND beat structure-free data: "
        f"{list(stable['k']) if len(stable) else 'none'}"
    )

    # --- 1. the elbow that isn't --------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    left.plot(list(KS), inertia, "o-", color=PALETTE["blue"], lw=2)
    left.set_xlabel("k")
    left.set_ylabel("Within-cluster sum of squares")
    left.set_title("The elbow plot")
    right.bar([str(k) for k in list(KS)[1:]], drops * 100, color=PALETTE["sky"])
    right.set_xlabel("k")
    right.set_ylabel("Improvement over k-1 (%)")
    right.set_title("Its first difference — no corner to find")
    right.grid(axis="y")
    fig.suptitle(
        "Inertia falls monotonically by construction, so it can never say 'no'",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        f"{len(X):,} customers from UCI Online Retail II. Adding a cluster always "
        "reduces within-cluster variance, whatever the data.",
    )
    print(
        "\n  "
        + str(save(fig, FIGURES / "01-the-elbow-that-isnt.png").relative_to(HERE.parent.parent))
    )

    # --- 2. against the null -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(
        table["k"], table["silhouette"], "o-", lw=2.2, color=PALETTE["blue"], label="real customers"
    )
    ax.plot(
        table["k"],
        table["silhouette_shuffled"],
        "o--",
        lw=2,
        color=PALETTE["orange"],
        label="same features, columns shuffled independently",
    )
    ax.fill_between(
        table["k"],
        table["silhouette_shuffled"],
        table["silhouette"],
        color=PALETTE["blue"],
        alpha=0.12,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.legend()
    ax.set_title("How much of the cluster quality is structure, and how much is skew")
    caption(
        fig,
        "Shuffling each column independently keeps every marginal distribution and "
        "destroys every relationship. The gap is what the relationships are worth.",
    )
    print(
        "  " + str(save(fig, FIGURES / "02-against-the-null.png").relative_to(HERE.parent.parent))
    )

    # --- 3. does it repeat? -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    reseed = [reseed_agreement(X, k, seeds=10) for k in KS]
    resample = [resample_agreement(X, k, rounds=10, seed=k) for k in KS]
    positions = np.arange(len(list(KS)))
    ax.boxplot(
        reseed,
        positions=positions - 0.17,
        widths=0.28,
        patch_artist=True,
        medianprops={"color": "black"},
        boxprops={"facecolor": PALETTE["sky"], "alpha": 0.85},
    )
    ax.boxplot(
        resample,
        positions=positions + 0.17,
        widths=0.28,
        patch_artist=True,
        medianprops={"color": "black"},
        boxprops={"facecolor": PALETTE["purple"], "alpha": 0.85},
    )
    ax.axhline(0.75, color=PALETTE["red"], ls="--", lw=1.6)
    ax.text(
        len(positions) - 0.5,
        0.765,
        "0.75 — 'substantially the same partition'",
        ha="right",
        fontsize=8.5,
        color=PALETTE["red"],
        fontweight="bold",
    )
    ax.set_xticks(positions, [str(k) for k in KS])
    ax.set_xlabel("k")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(0, 1.02)
    ax.plot([], [], color=PALETTE["sky"], lw=8, alpha=0.85, label="different random seed")
    ax.plot([], [], color=PALETTE["purple"], lw=8, alpha=0.85, label="80% resample")
    ax.legend(loc="lower left")
    ax.set_title("Run it again: how much of the partition survives?")
    caption(
        fig,
        "Adjusted Rand Index corrects for chance agreement, so a random pair of "
        "partitions scores 0 rather than something comfortably positive.",
    )
    print("  " + str(save(fig, FIGURES / "03-does-it-repeat.png").relative_to(HERE.parent.parent)))

    # --- 4. two runs of the same k, side by side ---------------------------
    k_show = 5
    a = fit_labels(X, k_show, seed=0)
    b = fit_labels(X, k_show, seed=7)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.6), sharex=True, sharey=True)
    for ax, labels, title in ((left, a, "seed 0"), (right, b, "seed 7")):
        ax.scatter(
            features["frequency"],
            features["monetary"],
            c=labels,
            cmap="Set2",
            s=5,
            alpha=0.6,
            linewidths=0,
        )
        ax.set_xlabel("log frequency")
        ax.set_title(title)
    left.set_ylabel("log monetary")
    from sklearn.metrics import adjusted_rand_score

    fig.suptitle(
        f"The same data, the same k={k_show}, two seeds — ARI {adjusted_rand_score(a, b):.2f}",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "A continuous cloud cut two different ways. k-means cannot report that "
        "there were no natural groups to find.",
    )
    print("  " + str(save(fig, FIGURES / "04-two-runs.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "customers": int(len(X)),
                "k_values_that_are_reproducible": list(map(int, stable["k"]))
                if len(stable)
                else [],
                "assessment": [v.summary() for v in verdicts],
                "ari_k5_two_seeds": round(float(adjusted_rand_score(a, b)), 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
