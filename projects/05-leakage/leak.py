"""Target leakage, demonstrated on a column the dataset's own authors warn about.

UCI's Bank Marketing data records 45,211 phone calls offering a term deposit, and whether
the customer subscribed. It carries a feature called `duration` — how long the call lasted,
in seconds.

The dataset documentation says, in as many words:

    "this attribute highly affects the output target (e.g., if duration=0 then y='no').
    Yet, the duration is not known before a call is performed. Also, after the end of the
    call y is obviously known. Thus, this input should only be included for benchmark
    purposes and should be discarded if the intention is to have a realistic predictive
    model."

**Almost nobody discards it**, because leaving it in takes ROC-AUC from roughly 0.79 to
roughly 0.93, and 0.93 is the number that gets put in the notebook.

The reason it is not a bug but a *leak* is worth being precise about: `duration` is not
wrong, and it is not noise. It is a genuine, powerful, perfectly measured predictor. It is
simply **unavailable at the moment the prediction has to be made**. A model that uses it
scores brilliantly in cross-validation and cannot be deployed, because to know the call
lasted eleven minutes you must first have made an eleven-minute call — by which point the
customer has already said yes or no and the prediction is worthless.

That is the general shape of target leakage, and it is why the test for it is never
statistical. **It is a question about time:** at the instant this decision is made, which of
these columns would I actually have?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns recorded after or during the event being predicted. The dataset's own
# documentation flags the first; the others are outcomes of the same campaign.
LEAKY = ("duration",)


@dataclass(frozen=True)
class Audit:
    """One column's relationship to the target, and the question that settles it."""

    column: str
    solo_auc: float  # how well this column alone separates the classes
    available_at_decision_time: bool

    @property
    def suspicious(self) -> bool:
        """A single column doing most of the work is a prompt to ask when it is measured."""
        return self.solo_auc >= 0.70 or self.solo_auc <= 0.30

    def verdict(self) -> str:
        if not self.available_at_decision_time:
            return "LEAK - not knowable when the prediction is made"
        if self.suspicious:
            return "check - unusually strong for a single column"
        return "ok"


def single_column_auc(values: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC of one feature against the target, computed without a model.

    Fitting a classifier to test for leakage buries the signal under hyperparameters. A
    rank correlation asks the question directly: *does this column, alone, order the rows
    almost perfectly?* If it does, find out when it is recorded before celebrating.
    """
    values = np.asarray(values, dtype=float)
    y = np.asarray(y).astype(bool)

    finite = np.isfinite(values)
    values, y = values[finite], y[finite]
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return 0.5

    # Mann-Whitney U, which equals the AUC. Ties share their average rank.
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)

    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < len(values):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]

    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def audit_columns(
    frame: pd.DataFrame, y: np.ndarray, *, known_at_decision_time: set[str]
) -> pd.DataFrame:
    """Score every numeric column alone, and record whether it exists yet.

    The second column cannot be computed. It has to be supplied by somebody who knows how
    the data was collected, and that is the honest part: **leakage detection is a
    conversation with the data owner, not a statistic.** The automated half only tells you
    which conversations to have first.
    """
    rows = []
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        audit = Audit(
            column=column,
            solo_auc=single_column_auc(series.to_numpy(), y),
            available_at_decision_time=column in known_at_decision_time,
        )
        rows.append(
            {
                "column": audit.column,
                "solo_auc": audit.solo_auc,
                "available": audit.available_at_decision_time,
                "verdict": audit.verdict(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("solo_auc", key=lambda s: (s - 0.5).abs(), ascending=False)
        .reset_index(drop=True)
    )


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, *, k: float = 0.10) -> float:
    """How many times the base rate you capture by calling the top k% of the list.

    The metric a campaign manager actually uses, and the one where leakage does its real
    damage: a leaky model promises a lift it cannot deliver on a single future call.
    """
    y_true = np.asarray(y_true).astype(bool)
    cut = max(1, int(len(scores) * k))
    top = np.argsort(-np.asarray(scores, dtype=float))[:cut]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base else 0.0


def expected_campaign_value(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    k: float,
    value_per_subscription: float,
    cost_per_call: float,
) -> dict:
    """What calling the top k% of a ranked list is worth, in money.

    The only comparison that matters between the leaky and the honest model, because the
    leaky one's advantage evaporates exactly here — it cannot rank a list of people who
    have not been called yet.
    """
    y_true = np.asarray(y_true).astype(bool)
    cut = max(1, int(len(scores) * k))
    top = np.argsort(-np.asarray(scores, dtype=float))[:cut]

    subscriptions = int(y_true[top].sum())
    revenue = subscriptions * value_per_subscription
    cost = cut * cost_per_call
    return {
        "calls": cut,
        "subscriptions": subscriptions,
        "conversion": round(subscriptions / cut, 4),
        "profit": round(revenue - cost, 2),
        "lift": round(lift_at_k(y_true, scores, k=k), 3),
    }
