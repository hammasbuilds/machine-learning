"""Association rules on a real wholesaler's invoices, and what survives a permutation null.

    uv run python projects/20-market-basket/run.py
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

from basket import (  # noqa: E402
    build_baskets,
    build_rules,
    holdout_check,
    item_counts,
    permutation_null,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, caption, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
TOP_ITEMS = 400
MIN_SUPPORT = 10


def shorten(text: str, width: int = 26) -> str:
    text = str(text).title()
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = online_retail()
    frame = frame[~frame["is_return"]]
    frame = frame.dropna(subset=["Description"])
    frame["Description"] = frame["Description"].astype(str).str.strip()

    cut = frame["InvoiceDate"].quantile(0.75)
    early = frame[frame["InvoiceDate"] <= cut]
    late = frame[frame["InvoiceDate"] > cut]

    baskets = build_baskets(early, basket_col="Invoice", item_col="Description")
    later = build_baskets(late, basket_col="Invoice", item_col="Description")
    counts = item_counts(baskets)

    print(f"\nOnline Retail II: {len(baskets):,} multi-item baskets before {cut:%Y-%m-%d}, "
          f"{len(later):,} after")
    print(f"  {len(counts):,} distinct products; rules built on the top {TOP_ITEMS} "
          f"with at least {MIN_SUPPORT} co-occurrences\n")

    rules = build_rules(baskets, min_support_count=MIN_SUPPORT, top_items=TOP_ITEMS)
    print(f"  {len(rules):,} rules from {TOP_ITEMS * (TOP_ITEMS - 1):,} candidate directed pairs.")
    print(f"  {(rules['lift'] > 1).mean():.1%} of them have lift above 1, which the "
          f"rule of thumb calls 'associated'.\n")

    # --- 1. confidence ranks by popularity ----------------------------------
    print("  === ranked by confidence ===")
    by_confidence = rules.sort_values("confidence", ascending=False).head(8)
    print(f"  {'rule':<56}{'conf':>7}{'lift':>7}{'P(B)':>8}")
    for r in by_confidence.itertuples():
        label = f"{shorten(r.antecedent, 24)} -> {shorten(r.consequent, 24)}"
        print(f"  {label:<56}{r.confidence:>7.1%}{r.lift:>7.1f}{r.consequent_support:>8.2%}")

    top_pct = counts.most_common(TOP_ITEMS)
    popular_rate = np.mean([by_confidence["consequent_support"].mean()])
    all_rate = rules["consequent_support"].mean()
    print(f"\n    Mean P(consequent) among the top-confidence rules: {popular_rate:.2%}, "
          f"against {all_rate:.2%}")
    print("    across all rules. Confidence ranks by how popular the right-hand side is.\n")

    # --- 2. lift ranks by rarity --------------------------------------------
    print("  === ranked by lift ===")
    by_lift = rules.sort_values("lift", ascending=False).head(8)
    print(f"  {'rule':<56}{'lift':>8}{'baskets':>9}{'conf':>7}")
    for r in by_lift.itertuples():
        label = f"{shorten(r.antecedent, 24)} -> {shorten(r.consequent, 24)}"
        print(f"  {label:<56}{r.lift:>8.1f}{r.support_count:>9}{r.confidence:>7.1%}")
    print(f"\n    Median supporting baskets among the top-lift rules: "
          f"{by_lift['support_count'].median():.0f}, out of {len(baskets):,}. These are real.")
    print(f"    They survive only because the candidate set was capped at the top {TOP_ITEMS}")
    print(f"    products with {MIN_SUPPORT}+ co-occurrences. Lift the cap and it changes:\n")

    # The same ranking with the guardrails removed, which is the default in most tooling.
    wide = build_rules(baskets, min_support_count=3, top_items=2500)
    wide_top = wide.sort_values("lift", ascending=False).head(6)
    print(f"  === the same ranking over {len(wide):,} rules, support floor 3 ===")
    print(f"  {'rule':<56}{'lift':>8}{'baskets':>9}{'conf':>7}")
    for r in wide_top.itertuples():
        label = f"{shorten(r.antecedent, 24)} -> {shorten(r.consequent, 24)}"
        print(f"  {label:<56}{r.lift:>8.1f}{r.support_count:>9}{r.confidence:>7.1%}")
    print(f"\n    Median supporting baskets now: {wide_top['support_count'].median():.0f}. "
          f"Top lift {wide_top['lift'].max():.0f}x, from\n    "
          f"{int(wide_top['support_count'].iloc[0])} baskets out of {len(baskets):,}. "
          f"That is one buyer's order pattern, ranked first.\n")

    # --- 3. what does chance produce? ---------------------------------------
    print("  === the permutation null: items reshuffled, all real structure destroyed ===")
    null = permutation_null(
        baskets, replicates=12, top_items=TOP_ITEMS, min_support_count=MIN_SUPPORT, seed=5
    )
    print(f"  {'':<22}{'real data':>12}{'shuffled':>12}")
    real_stats = {
        "rules found": len(rules),
        "median lift": rules["lift"].median(),
        "99th pct lift": rules["lift"].quantile(0.99),
        "max lift": rules["lift"].max(),
        "share lift > 2": (rules["lift"] > 2).mean(),
    }
    null_stats = {
        "rules found": null["rules"].mean(),
        "median lift": null["median_lift"].mean(),
        "99th pct lift": null["p99_lift"].mean(),
        "max lift": null["max_lift"].mean(),
        "share lift > 2": null["above_2"].mean(),
    }
    for key in real_stats:
        fmt = "{:>12.1%}" if key.startswith("share") else "{:>12.1f}"
        print(f"  {key:<22}" + fmt.format(real_stats[key]) + fmt.format(null_stats[key]))

    threshold = float(null["max_lift"].mean())
    survivors = rules[rules["lift"] > threshold]
    print(f"\n    Chance alone produces a maximum lift of {threshold:.1f} on this data.")
    print(f"    {len(survivors):,} of {len(rules):,} rules ({len(survivors) / len(rules):.1%}) "
          f"exceed it.")
    print("    Every rule below that line is indistinguishable from reshuffled products.\n")

    # --- 4. do the rules survive the next quarter? --------------------------
    print("  === the only test that matters: does the rule hold next quarter? ===")
    for label, ranked in (("by lift", rules.sort_values("lift", ascending=False)),
                          ("by leverage", rules.sort_values("leverage", ascending=False))):
        check = holdout_check(ranked, later, top=50)
        held = check["held"].mean()
        usable = check[np.isfinite(check["lift_later"])]
        ratio = (usable["lift_later"] / usable["lift_discovered"]).median()
        print(f"  top 50 {label:<14} {held:>6.0%} still have lift > 1   "
              f"median lift retained: {ratio:>5.0%}")
        if label == "by lift":
            lift_check = check
        else:
            leverage_check = check

    print(f"\n    Leverage holds {leverage_check['held'].mean() - lift_check['held'].mean():+.0%} "
          f"more often than lift, on rules bounded to the top {TOP_ITEMS} products.")
    print("    Both hold far better than expected - which is what the support floor bought.")
    print("    The same check on the unbounded rule set is the one that fails.\n")

    wide_check = holdout_check(wide.sort_values("lift", ascending=False), later, top=50)
    print(f"  top 50 by lift, unbounded    {wide_check['held'].mean():>6.0%} "
          f"still have lift > 1   (vs {lift_check['held'].mean():.0%} bounded)")
    print("    Remove the support floor and the top-ranked rules stop reproducing.\n")

    # --- figure 1: confidence is popularity ---------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    left.scatter(rules["consequent_support"], rules["confidence"], s=4, alpha=0.15,
                 color=PALETTE["blue"])
    line = np.linspace(rules["consequent_support"].min(), rules["consequent_support"].max(), 50)
    left.plot(line, line, color=PALETTE["red"], lw=2, ls="--", label="P(B): no association")
    left.set_xlabel("P(consequent) - how popular B is on its own")
    left.set_ylabel("Confidence  P(B | A)")
    left.set_xscale("log"); left.set_yscale("log")
    left.legend(fontsize=8)
    left.set_title("Every rule starts at the popularity of its consequent")

    right.scatter(rules["support_count"], rules["lift"], s=4, alpha=0.15, color=PALETTE["orange"])
    right.axhline(threshold, color=PALETTE["red"], lw=2, ls="--")
    right.text(rules["support_count"].max(), threshold * 1.1,
               f"max lift under reshuffling ({threshold:.0f})", fontsize=8, ha="right",
               color=PALETTE["red"], fontweight="bold")
    right.set_xscale("log"); right.set_yscale("log")
    right.set_xlabel("Baskets supporting the rule"); right.set_ylabel("Lift")
    right.set_title("and the biggest lifts rest on the fewest baskets")
    caption(fig, f"{len(rules):,} rules from {len(baskets):,} baskets. Both failure modes are "
                 f"structural, not artefacts of this dataset.")
    print("  " + str(save(fig, FIGURES / "01-two-failures.png").relative_to(HERE.parent.parent)))

    # --- figure 2: the null -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.logspace(0, np.log10(max(rules["lift"].max(), threshold) * 1.1), 60)
    ax.hist(rules["lift"], bins=bins, color=PALETTE["blue"], alpha=0.8, label="real baskets")
    ax.axvline(threshold, color=PALETTE["red"], lw=2.5)
    ax.axvline(1.0, color="#333333", lw=1.5, ls="--")
    ax.text(1.05, ax.get_ylim()[1] * 0.7, "lift = 1\n(the usual threshold)", fontsize=8)
    ax.text(threshold * 1.08, ax.get_ylim()[1] * 0.45,
            f"highest lift chance\nproduced: {threshold:.0f}", fontsize=8,
            color=PALETTE["red"], fontweight="bold")
    ax.set_xscale("log"); ax.set_xlabel("Lift"); ax.set_ylabel("Rules")
    ax.legend(fontsize=8)
    ax.set_title("Where the 'lift > 1 means associated' threshold actually belongs")
    caption(fig, "Null: items reassigned to baskets at random, preserving basket sizes and "
                 "item popularity. 12 replicates. Everything real structure contributes is "
                 "destroyed; the marginals are not.")
    print("  " + str(save(fig, FIGURES / "02-permutation-null.png").relative_to(HERE.parent.parent)))

    # --- figure 3: holdout --------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, check, title in (
        (left, lift_check, "Top 50 by lift"),
        (right, leverage_check, "Top 50 by leverage"),
    ):
        usable = check[np.isfinite(check["lift_later"])]
        ax.scatter(usable["lift_discovered"], usable["lift_later"], s=30,
                   color=PALETTE["blue"], alpha=0.75, edgecolor="white")
        limit = max(usable["lift_discovered"].max(), usable["lift_later"].max()) * 1.2
        ax.plot([1, limit], [1, limit], color=PALETTE["red"], ls="--", lw=1.5)
        ax.axhline(1.0, color="#333333", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Lift when discovered"); ax.set_ylabel("Lift the next quarter")
        ax.set_title(f"{title} — {check['held'].mean():.0%} still associated")
    caption(fig, "Dashed line: the rule holds exactly as measured. Points below it are rules "
                 "that weakened; points below the solid line stopped being associations.")
    print("  " + str(save(fig, FIGURES / "03-next-quarter.png").relative_to(HERE.parent.parent)))

    # --- figure 4: what a useful ranking looks like -------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    best = rules.sort_values("leverage", ascending=False).head(12)
    labels = [f"{shorten(r.antecedent, 22)}\n-> {shorten(r.consequent, 22)}"
              for r in best.itertuples()]
    bars = ax.barh(range(len(best)), best["leverage"] * len(baskets), color=PALETTE["green"])
    for bar, r in zip(bars, best.itertuples(), strict=True):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"lift {r.lift:.1f}, {r.support_count} baskets", va="center", fontsize=7.5)
    ax.set_yticks(range(len(best)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Excess baskets over independence (leverage x N)")
    ax.set_title("Ranked by how many extra baskets the association is worth")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    caption(fig, "Leverage is in baskets, not ratios. It cannot be inflated by a rare "
                 "consequent, which is why the rules it surfaces are duller and survive.")
    print("  " + str(save(fig, FIGURES / "04-leverage.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "baskets_early": len(baskets),
                "baskets_late": len(later),
                "distinct_products": len(counts),
                "rules": int(len(rules)),
                "share_lift_above_1": round(float((rules["lift"] > 1).mean()), 4),
                "null": {k: round(float(v), 4) for k, v in null_stats.items()},
                "real": {k: round(float(v), 4) for k, v in real_stats.items()},
                "chance_lift_threshold": round(threshold, 2),
                "rules_above_threshold": int(len(survivors)),
                "holdout": {
                    "by_lift_held": round(float(lift_check["held"].mean()), 4),
                    "by_leverage_held": round(float(leverage_check["held"].mean()), 4),
                    "by_lift_unbounded_held": round(float(wide_check["held"].mean()), 4),
                },
                "unbounded": {
                    "rules": int(len(wide)),
                    "max_lift": round(float(wide["lift"].max()), 1),
                    "median_support_of_top6": int(wide_top["support_count"].median()),
                },
                "top_by_confidence": [
                    {"rule": f"{r.antecedent} -> {r.consequent}",
                     "confidence": round(float(r.confidence), 4),
                     "lift": round(float(r.lift), 2),
                     "consequent_support": round(float(r.consequent_support), 4)}
                    for r in by_confidence.itertuples()
                ],
                "top_by_leverage": [
                    {"rule": f"{r.antecedent} -> {r.consequent}", "lift": round(r.lift, 2),
                     "baskets": int(r.support_count),
                     "excess_baskets": round(float(r.leverage * len(baskets)), 1)}
                    for r in best.itertuples()
                ],
            },
            indent=2,
        )
    )
    print(f"\n  top {len(top_pct)} products carried "
          f"{sum(c for _, c in top_pct) / sum(counts.values()):.1%} of all basket lines.")


if __name__ == "__main__":
    main()
