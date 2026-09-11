"""Customer value on 1.05M real invoice lines.

uv run python projects/04-customer-value/run.py
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

from value import (  # noqa: E402
    cohorts,
    concentration,
    gini,
    one_time_buyers,
    rfm,
    value_at_risk,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    money_axis,
    percent_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    ledger = online_retail()
    attributed = ledger.dropna(subset=["CustomerID"])
    print(
        f"\n{len(ledger):,} invoice lines, {ledger['InvoiceDate'].min().date()} to "
        f"{ledger['InvoiceDate'].max().date()}"
    )
    print(
        f"  {len(ledger) - len(attributed):,} lines ({1 - len(attributed) / len(ledger):.1%}) "
        f"carry no customer id and are excluded from everything per-customer"
    )
    print(
        f"  {int(ledger['is_return'].sum()):,} return lines, kept: a customer who bought "
        f"and sent it back is worth what is left\n"
    )

    frame = rfm(ledger)
    n = len(frame)

    # --- the number everybody reports, and the ones that matter -----------
    mean = frame["monetary"].mean()
    median = frame["monetary"].median()
    print(f"  {n:,} customers")
    print(f"    mean value    GBP {mean:>10,.0f}   <- the number that gets put in the deck")
    print(f"    median value  GBP {median:>10,.0f}   <- the typical customer")
    print(f"    the mean sits at the {(frame['monetary'] < mean).mean():.0%}th percentile\n")

    curve = concentration(frame["monetary"])
    deciles = {}
    for share in (0.01, 0.05, 0.10, 0.20, 0.50):
        idx = int(share * len(curve)) - 1
        deciles[share] = float(curve["revenue"].iloc[max(idx, 0)])
        print(f"    top {share:>4.0%} of customers  ->  {deciles[share]:>6.1%} of revenue")

    g = gini(frame["monetary"])
    once = one_time_buyers(frame)
    print(f"\n    Gini {g:.3f}   |   {once:.1%} of customers bought exactly once")

    # --- 1. the distribution ----------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    positive = frame.loc[frame["monetary"] > 0, "monetary"]
    left.hist(positive, bins=120, range=(0, 10000), color=PALETTE["blue"], alpha=0.85)
    left.axvline(mean, color=PALETTE["red"], lw=2)
    left.axvline(median, color=PALETTE["green"], lw=2)
    annotate(
        left, mean, left.get_ylim()[1] * 0.7, f"  mean\n  GBP {mean:,.0f}", color=PALETTE["red"]
    )
    annotate(
        left,
        median,
        left.get_ylim()[1] * 0.45,
        f"  median\n  GBP {median:,.0f}",
        dx=10,
        color=PALETTE["green"],
    )
    left.set_xlabel("Lifetime value (GBP, clipped at 10k for the view)")
    left.set_ylabel("Customers")
    left.set_title("Nobody is average")
    money_axis(left, "x", symbol="")

    right.plot(curve["customers"], curve["revenue"], color=PALETTE["orange"], lw=2.4)
    right.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1.2)
    for share in (0.05, 0.20):
        right.plot([share], [deciles[share]], "o", color=PALETTE["red"], ms=7)
        annotate(
            right,
            share,
            deciles[share],
            f"  top {share:.0%} = {deciles[share]:.0%}",
            color=PALETTE["red"],
        )
    right.set_xlabel("Share of customers (ranked by value)")
    right.set_ylabel("Share of revenue")
    right.set_title(f"Concentration — Gini {g:.2f}")
    percent_axis(right)
    percent_axis(right, "x")

    fig.suptitle(
        "A million invoice lines, 5,876 customers",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "Source: UCI Online Retail II (CC BY 4.0), 2009-12-01 to 2011-12-09. "
        "Returns netted off; unattributed lines excluded.",
    )
    print(
        "\n  "
        + str(save(fig, FIGURES / "01-nobody-is-average.png").relative_to(HERE.parent.parent))
    )

    # --- 2. cohort retention ----------------------------------------------
    group = cohorts(ledger)
    survival = group.average_survival()
    horizon = min(12, len(survival) - 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (_cohort, row) in enumerate(group.rates.iterrows()):
        series = row.dropna().iloc[: horizon + 1]
        ax.plot(
            series.index,
            series.to_numpy(),
            lw=1.1,
            alpha=0.45,
            color=PALETTE["sky"],
            label="individual cohorts" if i == 0 else None,
        )
    ax.plot(
        survival.index[: horizon + 1],
        survival.iloc[: horizon + 1].to_numpy(),
        lw=3,
        color=PALETTE["blue"],
        label="average across cohorts",
    )
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Share of the intake still buying")
    percent_axis(ax)
    ax.set_title(f"Month one retention is {survival.iloc[1]:.0%}. It never recovers.")
    ax.legend()
    caption(
        fig,
        f"{len(group.rates)} monthly cohorts. Cohorts weighted equally, not by size: "
        "a size-weighted average describes the biggest campaign, not retention.",
    )
    print(
        "  " + str(save(fig, FIGURES / "02-cohort-retention.png").relative_to(HERE.parent.parent))
    )

    # --- 3. the cohort heatmap --------------------------------------------
    view = group.rates.iloc[:, : horizon + 1]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    image = ax.imshow(view.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(view.shape[1]), [str(c) for c in view.columns], fontsize=8)
    ax.set_yticks(range(len(view)), [str(c) for c in view.index], fontsize=8)
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort")
    ax.set_title("Retention by intake month")
    ax.grid(visible=False)
    for y in range(view.shape[0]):
        for x in range(view.shape[1]):
            v = view.to_numpy()[y, x]
            if np.isfinite(v) and v > 0.01:
                ax.text(
                    x,
                    y,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if v > 0.28 else "#333333",
                )
    fig.colorbar(image, ax=ax, shrink=0.8, label="share of intake still buying")
    caption(
        fig,
        "The diagonal is an artefact of the window ending 2011-12-09: later cohorts "
        "simply have not had twelve months yet, and their blanks are not churn.",
    )
    print("  " + str(save(fig, FIGURES / "03-cohort-heatmap.png").relative_to(HERE.parent.parent)))

    # --- 4. who to phone --------------------------------------------------
    risk = value_at_risk(frame, dormant_days=180)
    top = risk.head(20)
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.barh(
        [str(int(c)) for c in top.index][::-1],
        top["monetary"].to_numpy()[::-1],
        color=PALETTE["purple"],
    )
    ax.set_xlabel("Lifetime value (GBP)")
    ax.set_ylabel("Customer id")
    money_axis(ax, "x", symbol="")
    ax.set_title(
        f"{len(risk):,} repeat customers dormant 180+ days, worth GBP {risk['monetary'].sum():,.0f}"
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(fig, "RFM is only useful when it ends in a list of names sorted by what is at stake.")
    print("  " + str(save(fig, FIGURES / "04-value-at-risk.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "invoice_lines": int(len(ledger)),
                "unattributed_share": round(1 - len(attributed) / len(ledger), 4),
                "customers": int(n),
                "mean_value": round(float(mean), 2),
                "median_value": round(float(median), 2),
                "mean_percentile": round(float((frame["monetary"] < mean).mean()), 4),
                "gini": round(g, 4),
                "one_time_buyer_share": round(once, 4),
                "revenue_share": {
                    f"top_{int(k * 100)}pct": round(v, 4) for k, v in deciles.items()
                },
                "month_1_retention": round(float(survival.iloc[1]), 4),
                "month_12_retention": round(float(survival.iloc[min(12, len(survival) - 1)]), 4),
                "dormant_repeat_customers": int(len(risk)),
                "value_dormant": round(float(risk["monetary"].sum()), 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
