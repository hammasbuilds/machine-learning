"""Does raising the price sell fewer units? It depends entirely on what you compare.

uv run python projects/10-price-elasticity/run.py
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

from elasticity import (  # noqa: E402
    naive_elasticity,
    per_product_elasticities,
    product_month_panel,
    within_product_and_month,
    within_product_elasticity,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, annotate, caption, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    panel = product_month_panel(online_retail(), min_months=6)
    print(f"\n{len(panel):,} product-months across {panel['StockCode'].nunique():,} products")
    print(f"  price range GBP {panel['price'].min():.2f} to GBP {panel['price'].max():,.2f}")
    print(f"  median units per product-month: {panel['units'].median():,.0f}\n")

    estimates = [
        naive_elasticity(panel),
        within_product_elasticity(panel),
        within_product_and_month(panel),
    ]

    print(f"  {'method':<34}{'elasticity':>12}{'std err':>10}{'t':>9}   sign")
    for e in estimates:
        s = e.summary()
        print(
            f"  {s['method']:<34}{s['elasticity']:>12.3f}{s['standard_error']:>10.3f}"
            f"{s['t']:>9.1f}   {'ok' if s['sign_is_sane'] else 'WRONG WAY'}"
        )

    print()
    for e in estimates:
        print(f"  {e.sentence()}")

    # --- 1. why the pooled estimate lies ----------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    sample = panel.sample(min(6000, len(panel)), random_state=0)
    left.scatter(
        sample["log_price"],
        sample["log_units"],
        s=5,
        alpha=0.25,
        color=PALETTE["blue"],
        linewidths=0,
    )
    line = np.linspace(panel["log_price"].min(), panel["log_price"].max(), 50)
    naive = estimates[0]
    left.plot(
        line,
        naive.elasticity * line
        + (panel["log_units"].mean() - naive.elasticity * panel["log_price"].mean()),
        color=PALETTE["red"],
        lw=2.6,
    )
    left.set_xlabel("log price")
    left.set_ylabel("log units sold")
    left.set_title(f"Pooled: elasticity {naive.elasticity:+.2f}")

    # The same picture, a handful of products at a time.
    busiest = panel.groupby("StockCode").size().sort_values(ascending=False).head(6).index
    for code, colour in zip(busiest, PALETTE.values(), strict=False):
        group = panel[panel["StockCode"] == code]
        right.plot(
            group["log_price"], group["log_units"], "o-", ms=4, lw=1.2, alpha=0.85, color=colour
        )
    right.set_xlabel("log price")
    right.set_ylabel("log units sold")
    right.set_title("Six products, each tracked against itself")

    fig.suptitle(
        "The pooled cloud is a catalogue. The lines are demand curves.",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        f"UCI Online Retail II: {len(panel):,} product-months, "
        f"{panel['StockCode'].nunique():,} products observed 6+ months.",
    )
    print(
        "\n  " + str(save(fig, FIGURES / "01-pooled-vs-within.png").relative_to(HERE.parent.parent))
    )

    # --- 2. the three estimates -------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    names = [e.method for e in estimates][::-1]
    values = [e.elasticity for e in estimates][::-1]
    errors = [1.96 * e.standard_error for e in estimates][::-1]
    colours = [PALETTE["green"] if v < 0 else PALETTE["red"] for v in values]
    ax.barh(names, values, xerr=errors, color=colours, capsize=4)
    ax.axvline(0, color="#444444", lw=2)
    for i, v in enumerate(values):
        ax.text(
            v + (0.03 if v > 0 else -0.03),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v > 0 else "right",
            fontweight="bold",
        )
    ax.set_xlabel("Estimated price elasticity of demand")
    ax.set_title("Same ledger, three comparisons, opposite conclusions")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        "Bars show 95% intervals. A positive elasticity means customers bought more "
        "when the price rose - which is a confounder speaking, not a customer.",
    )
    print("  " + str(save(fig, FIGURES / "02-three-estimates.png").relative_to(HERE.parent.parent)))

    # --- 3. the confounder, drawn -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.3))
    by_product = panel.groupby("StockCode").agg(
        price=("price", "median"), units=("units", "median")
    )
    ax.scatter(
        by_product["price"],
        by_product["units"],
        s=9,
        alpha=0.35,
        color=PALETTE["purple"],
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median price of the product (GBP, log scale)")
    ax.set_ylabel("Median units sold per month (log scale)")
    ax.set_title("Cheap products sell in hundreds. Expensive ones sell in ones.")
    caption(
        fig,
        "This is the relationship the pooled regression measures. It is real, and it "
        "is not a demand curve - it is what kind of thing each product is.",
    )
    print("  " + str(save(fig, FIGURES / "03-the-confounder.png").relative_to(HERE.parent.parent)))

    # --- 4. one number hides the distribution -----------------------------
    per_product = per_product_elasticities(panel, min_months=10)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.hist(per_product["elasticity"].clip(-8, 8), bins=70, color=PALETTE["sky"], alpha=0.9)
    ax.axvline(0, color="#444444", lw=2)
    median = float(per_product["elasticity"].median())
    ax.axvline(median, color=PALETTE["red"], lw=2)
    annotate(ax, median, ax.get_ylim()[1] * 0.85, f"  median {median:+.2f}", color=PALETTE["red"])
    share_negative = float((per_product["elasticity"] < 0).mean())
    ax.set_xlabel("Elasticity estimated for that product alone (clipped to ±8 for the view)")
    ax.set_ylabel("Products")
    ax.set_title(
        f"{len(per_product):,} products, one elasticity each — {share_negative:.0%} slope downward"
    )
    caption(
        fig,
        "Products with 10+ months of their own price history. A single "
        "catalogue-wide elasticity asserts every product behaves the same way.",
    )
    print("  " + str(save(fig, FIGURES / "04-per-product.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "product_months": int(len(panel)),
                "products": int(panel["StockCode"].nunique()),
                "estimates": [e.summary() for e in estimates],
                "per_product": {
                    "products_with_own_estimate": int(len(per_product)),
                    "median_elasticity": round(median, 3),
                    "share_downward_sloping": round(share_negative, 4),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
