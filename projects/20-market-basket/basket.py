"""Association rules, and the three ways a "strong rule" turns out to be nothing.

Market basket analysis produces rules of the form *{A} -> {B}*, scored by three numbers that
are almost always presented as a ranked table and almost never interrogated:

    support     how often A and B appear together at all
    confidence  P(B | A) - of the baskets containing A, how many contain B
    lift        confidence / P(B) - how much more likely B is, given A

Each of the three fails in its own way, and all three failures are visible in any real
transaction dataset:

**1. Confidence is popularity in disguise.** If B is in 30% of all baskets, then *any* rule
ending in B starts at 30% confidence before A contributes anything. Rank by confidence and
you rank by how popular the right-hand side is. The classic illustration: {anything} -> {milk}
is a fantastic rule in a supermarket, and it is not a rule, it is a fact about milk.

**2. Lift fixes that and breaks somewhere else.** Lift divides out the base rate, so popular
consequents stop winning. Now the winners are pairs seen in four baskets, where the ratio is
computed from a denominator of four. A lift of 300 on five co-occurrences is one customer's
habit and will not survive contact with next quarter.

**3. The search is enormous and nobody corrects for it.** A thousand products give half a
million candidate pairs. At any significance threshold, thousands of them clear it by chance.
This is the multiple-comparisons problem of project 01 wearing different clothes, and the
standard tooling has no notion of it at all.

The defences implemented here are the ones that work: a **permutation null** to find the lift
value that chance produces on this dataset, and an **out-of-period check** to see whether a
rule discovered in one quarter still holds in the next.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


def build_baskets(frame: pd.DataFrame, *, basket_col: str, item_col: str) -> list[frozenset]:
    """One set of distinct items per transaction.

    Sets, not lists: buying three of something is a quantity question, not an association
    question, and counting it twice inflates every co-occurrence that involves it.
    """
    grouped = frame.groupby(basket_col)[item_col].apply(lambda s: frozenset(s.unique()))
    return [b for b in grouped.tolist() if len(b) > 1]


def item_counts(baskets: list[frozenset]) -> Counter:
    counts: Counter = Counter()
    for basket in baskets:
        counts.update(basket)
    return counts


def pair_counts(baskets: list[frozenset], *, keep: set | None = None, max_basket: int = 60) -> Counter:
    """Co-occurrence counts for every pair, restricted to items worth counting.

    `max_basket` drops the handful of enormous baskets - wholesale orders of two hundred
    distinct lines - because one of them contributes 20,000 pairs and would dominate the
    counts on its own. That is a judgement call, so it is a parameter with a stated default
    rather than a silent filter.
    """
    counts: Counter = Counter()
    for basket in baskets:
        items = sorted(basket & keep) if keep is not None else sorted(basket)
        if len(items) < 2 or len(items) > max_basket:
            continue
        counts.update(combinations(items, 2))
    return counts


@dataclass(frozen=True)
class Rule:
    """A -> B, with every number needed to distrust it."""

    antecedent: str
    consequent: str
    support_count: int
    support: float
    confidence: float
    lift: float
    consequent_support: float
    leverage: float

    def summary(self) -> dict:
        return {
            "rule": f"{self.antecedent} -> {self.consequent}",
            "baskets": self.support_count,
            "support": round(self.support, 5),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 2),
        }


def build_rules(
    baskets: list[frozenset], *, min_support_count: int = 5, top_items: int = 400
) -> pd.DataFrame:
    """Every directed pair rule above a support floor.

    `top_items` restricts to the most-sold products. Not for speed alone: the rules involving
    items sold four times are the ones the lift ranking is about to be dominated by, and the
    honest thing is to bound the candidate set explicitly rather than let the ranking find
    them and present them as discoveries.
    """
    counts = item_counts(baskets)
    keep = {item for item, _ in counts.most_common(top_items)}
    pairs = pair_counts(baskets, keep=keep)
    n = len(baskets)

    rows = []
    for (a, b), together in pairs.items():
        if together < min_support_count:
            continue
        support = together / n
        for antecedent, consequent in ((a, b), (b, a)):
            p_consequent = counts[consequent] / n
            confidence = together / counts[antecedent]
            rows.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support_count": together,
                    "support": support,
                    "confidence": confidence,
                    "lift": confidence / p_consequent,
                    "consequent_support": p_consequent,
                    # Leverage: how many more baskets than independence predicts. In baskets,
                    # not ratios - the number a buyer would actually care about.
                    "leverage": support - (counts[antecedent] / n) * p_consequent,
                }
            )

    return pd.DataFrame(rows)


def permutation_null(
    baskets: list[frozenset], *, replicates: int = 20, top_items: int = 400,
    min_support_count: int = 5, seed: int = 0,
) -> pd.DataFrame:
    """Destroy every real association, keep the basket sizes, and see what lift still appears.

    Items are reassigned to baskets at random, preserving how many items each basket has and
    how often each item is sold. Any lift surviving this is manufactured by the marginals plus
    chance - which is the number the "lift > 1 means associated" rule of thumb is missing.
    """
    rng = np.random.default_rng(seed)
    pool = [item for basket in baskets for item in basket]
    sizes = [len(b) for b in baskets]

    rows = []
    for replicate in range(replicates):
        shuffled = rng.permutation(pool)
        position = 0
        fake = []
        for size in sizes:
            fake.append(frozenset(shuffled[position : position + size]))
            position += size

        rules = build_rules(fake, min_support_count=min_support_count, top_items=top_items)
        if rules.empty:
            continue
        rows.append(
            {
                "replicate": replicate,
                "rules": len(rules),
                "max_lift": float(rules["lift"].max()),
                "p99_lift": float(rules["lift"].quantile(0.99)),
                "median_lift": float(rules["lift"].median()),
                "above_2": float((rules["lift"] > 2).mean()),
            }
        )

    return pd.DataFrame(rows)


def holdout_check(rules: pd.DataFrame, later: list[frozenset], *, top: int = 50) -> pd.DataFrame:
    """Re-measure the top rules on a later period they were not discovered in.

    The only test that matters commercially. A rule is a claim about what will happen next
    time, and a claim that cannot be re-measured on the next quarter was a description of the
    past.
    """
    counts = item_counts(later)
    n = len(later)
    together: Counter = Counter()
    wanted = {
        frozenset((r.antecedent, r.consequent))
        for r in rules.head(top).itertuples()
    }

    for basket in later:
        for pair in wanted:
            if pair <= basket:
                together[pair] += 1

    rows = []
    for rule in rules.head(top).itertuples():
        pair = frozenset((rule.antecedent, rule.consequent))
        seen = together[pair]
        antecedent_count = counts.get(rule.antecedent, 0)
        consequent_rate = counts.get(rule.consequent, 0) / n if n else 0.0

        confidence = seen / antecedent_count if antecedent_count else float("nan")
        lift = confidence / consequent_rate if consequent_rate > 0 else float("nan")
        rows.append(
            {
                "antecedent": rule.antecedent,
                "consequent": rule.consequent,
                "lift_discovered": rule.lift,
                "lift_later": lift,
                "support_discovered": rule.support_count,
                "support_later": seen,
                "held": bool(np.isfinite(lift) and lift > 1.0),
            }
        )

    return pd.DataFrame(rows)
