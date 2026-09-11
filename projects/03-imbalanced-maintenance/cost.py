"""Failure prediction where the two mistakes cost very different amounts.

10,000 machine cycles, 339 failures. **A model that predicts "no failure" every single
time scores 96.6% accuracy** and is worth nothing. That is the whole problem with
imbalanced classification in one number, and it is why three of the defaults have to go:

  accuracy      dominated by the majority class; reports the base rate as skill
  ROC-AUC       optimistic under imbalance, because the false-positive rate is divided by
                a huge number of true negatives and barely moves
  threshold 0.5  a decision nobody made on purpose, silently asserting that a missed
                failure and an unnecessary inspection cost the same

On this problem they do not. An unplanned stoppage costs a shift of production; an
unnecessary inspection costs an engineer an hour. The ratio is somewhere near 50:1, and
the threshold should move by roughly that much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def precision_recall_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    """Precision and recall at every threshold, plus the thresholds themselves.

    Written out rather than imported so the axis that matters is explicit: **precision is
    the share of alarms that were real**, and on a 3.4% base rate it is brutal. A model
    with 90% recall and 10% precision raises nine false alarms for every real failure, and
    maintenance teams stop answering the tenth.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    s_sorted = scores[order]

    true_positives = np.cumsum(y_sorted)
    predicted_positives = np.arange(1, len(y_sorted) + 1)
    positives = int(y_true.sum())

    precision = true_positives / predicted_positives
    recall = true_positives / positives if positives else np.zeros_like(precision)

    # Keep one point per distinct score, at the last index where that score appears.
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    return precision[distinct], recall[distinct], s_sorted[distinct]


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, the honest summary under imbalance.

    Its floor is the base rate, not 0.5. On this dataset a useless model scores 0.034, so
    "0.55 average precision" is a sixteenfold improvement over nothing — a statement
    ROC-AUC cannot make, because its floor is 0.5 whatever the imbalance.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    # Rectangular integration over recall, which is the standard AP definition.
    recall_with_zero = np.r_[0.0, recall]
    return float(np.sum(np.diff(recall_with_zero) * precision))


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-positive rate against true-positive rate."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    positives = max(int(y_true.sum()), 1)
    negatives = max(int((~y_true).sum()), 1)

    tpr = np.r_[0.0, np.cumsum(y_sorted) / positives]
    fpr = np.r_[0.0, np.cumsum(~y_sorted) / negatives]
    return fpr, tpr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal area. Used for ROC, where the curve is monotone in x."""
    return float(np.trapezoid(y, x))


# --- the decision ---------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A threshold chosen against a stated cost, with what it buys and what it costs."""

    threshold: float
    cost: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def recall(self) -> float:
        caught = self.true_positives + self.false_negatives
        return self.true_positives / caught if caught else 0.0

    @property
    def precision(self) -> float:
        alarms = self.true_positives + self.false_positives
        return self.true_positives / alarms if alarms else 0.0

    @property
    def accuracy(self) -> float:
        total = sum(
            (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives)
        )
        return (self.true_positives + self.true_negatives) / total if total else 0.0

    def summary(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "cost": round(self.cost, 2),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "accuracy": round(self.accuracy, 4),
            "missed_failures": self.false_negatives,
            "false_alarms": self.false_positives,
        }


def evaluate_at(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, *, cost_miss: float, cost_alarm: float
) -> Decision:
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(scores, dtype=float) >= threshold

    tp = int(np.sum(flagged & y_true))
    fp = int(np.sum(flagged & ~y_true))
    fn = int(np.sum(~flagged & y_true))
    tn = int(np.sum(~flagged & ~y_true))

    return Decision(threshold, fn * cost_miss + fp * cost_alarm, tp, fp, fn, tn)


def cost_curve(
    y_true: np.ndarray, scores: np.ndarray, *, cost_miss: float, cost_alarm: float, steps: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Total cost at every threshold. The shape is the argument.

    It is U-shaped: flag nothing and every failure is missed; flag everything and every
    cycle is inspected. The minimum sits wherever the two costs balance, and on a 50:1
    ratio that is nowhere near 0.5.
    """
    thresholds = np.linspace(0.0, 1.0, steps)
    costs = np.array(
        [
            evaluate_at(y_true, scores, t, cost_miss=cost_miss, cost_alarm=cost_alarm).cost
            for t in thresholds
        ]
    )
    return thresholds, costs


def best_threshold(
    y_true: np.ndarray, scores: np.ndarray, *, cost_miss: float, cost_alarm: float, steps: int = 400
) -> Decision:
    """The threshold that minimises expected cost, rather than the one that ships by default."""
    thresholds, costs = cost_curve(
        y_true, scores, cost_miss=cost_miss, cost_alarm=cost_alarm, steps=steps
    )
    return evaluate_at(
        y_true,
        scores,
        float(thresholds[int(np.argmin(costs))]),
        cost_miss=cost_miss,
        cost_alarm=cost_alarm,
    )
