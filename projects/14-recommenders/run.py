"""Can a recommender beat "show everyone the bestsellers"? 1M real transactions.

uv run python projects/14-recommenders/run.py
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

from recommend import (  # noqa: E402
    ALS,
    ItemKNN,
    Popularity,
    Random,
    build_interactions,
    evaluate,
    temporal_split,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
K = 10


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    ledger = online_retail()
    purchases = ledger[(~ledger["is_return"]) & (ledger["Quantity"] > 0)].dropna(
        subset=["CustomerID"]
    )
    purchases = purchases[["CustomerID", "StockCode", "InvoiceDate"]].drop_duplicates()
    print(f"\n{len(purchases):,} distinct (customer, product) purchases")

    # --- the split has to be temporal --------------------------------------
    train_raw, test_raw = temporal_split(purchases, date_col="InvoiceDate", quantile=0.8)
    print(f"  train: up to {train_raw['InvoiceDate'].max().date()}  ({len(train_raw):,})")
    print(f"  test : after that                ({len(test_raw):,})")

    train = build_interactions(train_raw, user_col="CustomerID", item_col="StockCode")
    print(
        f"\n  {train.n_users:,} customers x {train.n_items:,} products, "
        f"{train.density:.3%} of the matrix observed"
    )

    # Test interactions, restricted to users and items the model has actually seen.
    user_index = {u: i for i, u in enumerate(train.user_ids)}
    item_index = {v: i for i, v in enumerate(train.item_ids)}
    test_by_user: dict[int, set[int]] = {}
    for customer, product in zip(test_raw["CustomerID"], test_raw["StockCode"], strict=True):
        if customer in user_index and product in item_index:
            test_by_user.setdefault(user_index[customer], set()).add(item_index[product])
    print(f"  {len(test_by_user):,} customers have held-out purchases to be judged on")

    # --- the line-up -------------------------------------------------------
    models = [Random(seed=0), Popularity(), ItemKNN(neighbours=60), ALS(factors=32, iterations=12)]
    scores = []
    for model in models:
        print(f"\n  fitting {model.name} ...", flush=True)
        model.fit(train)
        result = evaluate(model, train, test_by_user, k=K)
        scores.append(result)
        s = result.summary()
        print(
            f"    precision@{K} {s['precision@k']:.4f}   recall@{K} {s['recall@k']:.4f}   "
            f"ndcg@{K} {s['ndcg@k']:.4f}"
        )

    table = pd.DataFrame([s.summary() for s in scores])
    print(
        f"\n  {'model':<26}{'prec@10':>10}{'recall@10':>11}{'ndcg@10':>10}"
        f"{'coverage':>11}{'novelty':>10}"
    )
    for _, r in table.iterrows():
        print(
            f"  {r['model']:<26}{r['precision@k']:>10.4f}{r['recall@k']:>11.4f}"
            f"{r['ndcg@k']:>10.4f}{r['catalogue_coverage']:>11.2%}{r['novelty']:>10.2f}"
        )

    popularity = next(s for s in scores if s.model == "Popularity")
    best = max((s for s in scores if s.model != "Popularity"), key=lambda s: s.ndcg)
    beat = best.ndcg > popularity.ndcg
    print("\n  === does anything beat the bestsellers? ===")
    if beat:
        print(
            f"    yes: {best.model} at ndcg {best.ndcg:.4f} vs popularity {popularity.ndcg:.4f} "
            f"(+{(best.ndcg / popularity.ndcg - 1):.1%})"
        )
    else:
        print(
            f"    no. best model {best.model} scores ndcg {best.ndcg:.4f}; "
            f"popularity scores {popularity.ndcg:.4f}"
        )
    print(
        f"    but popularity covers {popularity.coverage:.2%} of the catalogue "
        f"and {best.model} covers {best.coverage:.2%}."
    )
    print("    A recommender that shows everyone the same ten items is not a recommender.")

    # --- 1. accuracy, with the baseline drawn in --------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    order = table.sort_values("ndcg@k")
    colours = [PALETTE["green"] if v > popularity.ndcg else PALETTE["sky"] for v in order["ndcg@k"]]
    bars = ax.barh(order["model"], order["ndcg@k"], color=colours)
    ax.axvline(popularity.ndcg, color=PALETTE["red"], lw=2, ls="--")
    ax.text(
        popularity.ndcg * 1.03,
        -0.42,
        "the bestsellers",
        color=PALETTE["red"],
        fontsize=9,
        fontweight="bold",
    )
    for bar, v in zip(bars, order["ndcg@k"], strict=True):
        ax.text(
            v + max(order["ndcg@k"]) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_xlabel(f"NDCG@{K}")
    ax.set_title("Every recommender is measured against doing nothing clever")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        f"UCI Online Retail II. Temporal split at the 80th percentile of date; "
        f"{len(test_by_user):,} customers judged on held-out purchases.",
    )
    print(
        "\n  " + str(save(fig, FIGURES / "01-vs-the-baseline.png").relative_to(HERE.parent.parent))
    )

    # --- 2. accuracy is not the only axis ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, colour in zip(
        scores, (PALETTE["purple"], PALETTE["red"], PALETTE["blue"], PALETTE["green"]), strict=True
    ):
        ax.scatter(
            s.coverage,
            s.ndcg,
            s=260,
            color=colour,
            alpha=0.85,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            s.model,
            (s.coverage, s.ndcg),
            xytext=(9, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Share of the catalogue ever recommended")
    ax.set_ylabel(f"NDCG@{K}")
    percent_axis(ax, "x")
    ax.set_title("Top-left is a model that is accurate by being useless")
    caption(
        fig,
        "A recommender that only ever surfaces bestsellers recommends what customers "
        "would have found anyway, and never sells the long tail.",
    )
    print(
        "  "
        + str(save(fig, FIGURES / "02-accuracy-vs-coverage.png").relative_to(HERE.parent.parent))
    )

    # --- 3. what the split does -------------------------------------------
    random_split = purchases.sample(frac=1.0, random_state=0)
    cut = int(len(random_split) * 0.8)
    rnd_train = build_interactions(
        random_split.iloc[:cut], user_col="CustomerID", item_col="StockCode"
    )
    r_user = {u: i for i, u in enumerate(rnd_train.user_ids)}
    r_item = {v: i for i, v in enumerate(rnd_train.item_ids)}
    rnd_test: dict[int, set[int]] = {}
    for c, p in zip(
        random_split.iloc[cut:]["CustomerID"], random_split.iloc[cut:]["StockCode"], strict=True
    ):
        if c in r_user and p in r_item:
            rnd_test.setdefault(r_user[c], set()).add(r_item[p])

    knn_random = evaluate(ItemKNN(neighbours=60).fit(rnd_train), rnd_train, rnd_test, k=K)
    knn_temporal = next(s for s in scores if s.model.startswith("Item-item"))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    values = [knn_random.ndcg, knn_temporal.ndcg]
    bars = ax.bar(
        ["Random split\n(leaks the future)", "Temporal split\n(honest)"],
        values,
        color=[PALETTE["red"], PALETTE["green"]],
        width=0.5,
    )
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"  {v:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel(f"NDCG@{K}")
    inflation = knn_random.ndcg / max(knn_temporal.ndcg, 1e-9)
    ax.set_title(f"The same model, two splits — {inflation:.1f}x apart")
    caption(
        fig,
        "Recommending an item somebody bought in March, using the fact that they "
        "bought it in June, is a lookup rather than a recommendation.",
    )
    print("  " + str(save(fig, FIGURES / "03-the-split.png").relative_to(HERE.parent.parent)))

    # --- 4. the long tail --------------------------------------------------
    counts = np.bincount(train.items, minlength=train.n_items)
    ranked = np.sort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(
        np.arange(1, len(ranked) + 1),
        np.cumsum(ranked) / ranked.sum(),
        lw=2.4,
        color=PALETTE["orange"],
    )
    for share in (0.2, 0.5):
        idx = int(np.searchsorted(np.cumsum(ranked) / ranked.sum(), share)) + 1
        ax.plot([idx], [share], "o", color=PALETTE["red"], ms=7)
        ax.annotate(
            f"{idx:,} products = {share:.0%} of purchases",
            (idx, share),
            xytext=(10, -4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Products, ranked by popularity")
    ax.set_ylabel("Cumulative share of purchases")
    percent_axis(ax)
    ax.set_title(f"{train.n_items:,} products, and the head takes most of the volume")
    caption(
        fig,
        "This concentration is why the popularity baseline is hard to beat, and why "
        "beating it on accuracy alone is not the goal.",
    )
    print("  " + str(save(fig, FIGURES / "04-long-tail.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "purchases": int(len(purchases)),
                "users": train.n_users,
                "items": train.n_items,
                "density": round(train.density, 6),
                "test_users": len(test_by_user),
                "k": K,
                "scores": [s.summary() for s in scores],
                "best_beats_popularity": bool(beat),
                "split_inflation": round(float(inflation), 3),
                "random_split_ndcg": round(knn_random.ndcg, 4),
                "temporal_split_ndcg": round(knn_temporal.ndcg, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
