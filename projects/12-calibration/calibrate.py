"""Turning a score into a probability you can act on.

A classifier outputs a number between 0 and 1. Almost nobody checks whether that number
means what it appears to mean.

    "the model says 0.70"

Does 70% of that group actually default? Usually not. Gradient boosting is systematically
**overconfident** — it pushes scores toward 0 and 1 because that is what minimising log
loss on finite data rewards. Random forests are usually the opposite, too timid, because
averaging trees pulls every prediction toward the middle.

**And AUC cannot see any of it.** AUC depends only on the *order* of the scores. Apply any
monotone transformation — square them, take logs, halve them all — and AUC is unchanged
while every probability is different. A model with 0.93 AUC can say 70% and be right 45% of
the time, and nothing in the usual report will tell you.

That matters the moment a number is used rather than a ranking:

    expected loss  = P(default) x exposure x loss given default
    expected value = P(convert) x margin  - cost of contact
    reserve        = P(claim)   x severity

Every one of those multiplies the probability by money. A ranking cannot be multiplied.

Two repairs, and the difference between them is the whole decision:

  Platt      fit a logistic curve to the scores. Two parameters, so it needs little data
             and can only apply a sigmoid-shaped correction.
  Isotonic   fit any non-decreasing step function. Far more flexible, and with a small
             calibration set it will happily fit the noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def reliability(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 12, strategy: str = "quantile"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted probability against observed frequency, in bins.

    `strategy="quantile"` puts an equal *number of samples* in each bin rather than equal
    width. On a skewed score distribution — which is every real classifier — uniform bins
    leave the top bin holding nine samples and the bottom holding ninety thousand, and the
    resulting plot is dominated by noise at exactly the end that matters.

    Returns (mean predicted, observed rate, count) per bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)

    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, len(edges) - 2)

    predicted, observed, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = index == b
        n = int(mask.sum())
        counts.append(n)
        predicted.append(float(probabilities[mask].mean()) if n else np.nan)
        observed.append(float(y_true[mask].mean()) if n else np.nan)

    return np.array(predicted), np.array(observed), np.array(counts)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 12
) -> float:
    """Average gap between claimed and actual probability, weighted by bin size.

    The single number for "how much can I trust this probability". 0 is perfect. A model
    with ECE 0.08 is, on average, eight percentage points out — which on a portfolio of a
    million loans is a large sum of money that no accuracy metric mentions.
    """
    predicted, observed, counts = reliability(y_true, probabilities, bins=bins)
    valid = counts > 0
    if not valid.any():
        return float("nan")
    return float(np.average(np.abs(predicted[valid] - observed[valid]), weights=counts[valid]))


def maximum_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 12
) -> float:
    """The worst bin. ECE can hide a catastrophic region behind a comfortable average."""
    predicted, observed, counts = reliability(y_true, probabilities, bins=bins)
    valid = counts > 0
    return (
        float(np.max(np.abs(predicted[valid] - observed[valid]))) if valid.any() else float("nan")
    )


def brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of a probability. Decomposes into calibration + refinement."""
    return float(np.mean((np.asarray(probabilities, float) - np.asarray(y_true, float)) ** 2))


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC. Included to demonstrate what it cannot see."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = int(y_true.sum()), int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique) < len(scores):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]

    return float((ranks[y_true].sum() - positives * (positives + 1) / 2) / (positives * negatives))


# --- the calibrators ------------------------------------------------------


class Platt:
    """Logistic regression on the score. Two parameters.

    Fitted on the log-odds of the score rather than the score itself, which is the standard
    form and lets the sigmoid actually stretch rather than just shift.
    """

    name = "Platt (sigmoid)"

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1e10, solver="lbfgs")

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> Platt:
        self._model.fit(self._logit(scores), np.asarray(y_true).astype(int))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._logit(scores))[:, 1]


class Isotonic:
    """A non-decreasing step function. Non-parametric and greedy.

    More flexible than Platt by a wide margin, which cuts both ways: given a small
    calibration set it will fit the noise in it, and the result looks excellent on the data
    it was fitted to and worse than doing nothing on new data. The experiment in `run.py`
    varies the calibration set size specifically to find where that turns over.
    """

    name = "Isotonic"

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> Isotonic:
        self._model.fit(np.asarray(scores, dtype=float), np.asarray(y_true, dtype=float))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(self._model.predict(np.asarray(scores, dtype=float)), 0.0, 1.0)


class Uncalibrated:
    """The model's raw output, so the comparison has a control."""

    name = "Uncalibrated"

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> Uncalibrated:  # noqa: ARG002
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float)


@dataclass(frozen=True)
class Report:
    """One calibrator, judged on held-out data."""

    method: str
    auc: float
    brier: float
    ece: float
    mce: float
    n_calibration: int

    def summary(self) -> dict:
        return {
            "method": self.method,
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 5),
            "ece": round(self.ece, 5),
            "max_error": round(self.mce, 5),
            "calibration_samples": self.n_calibration,
        }


def evaluate(
    calibrator,
    scores_fit: np.ndarray,
    y_fit: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    *,
    bins: int = 12,
) -> tuple[Report, np.ndarray]:
    """Fit a calibrator on one split, judge it on another.

    The two splits must be disjoint, and it is worth saying why rather than assuming it.
    Isotonic regression can reproduce any training set exactly if allowed enough steps;
    fitted and judged on the same data it reports an ECE near zero regardless of whether it
    has learned anything. Calibration needs its own held-out set exactly as a model does.
    """
    calibrated = calibrator.fit(scores_fit, y_fit).transform(scores_test)
    return (
        Report(
            method=calibrator.name,
            auc=auc(y_test, calibrated),
            brier=brier(y_test, calibrated),
            ece=expected_calibration_error(y_test, calibrated, bins=bins),
            mce=maximum_calibration_error(y_test, calibrated, bins=bins),
            n_calibration=len(scores_fit),
        ),
        calibrated,
    )


def money_at_stake(
    y_true: np.ndarray, probabilities: np.ndarray, *, exposure: float, loss_given_default: float
) -> dict:
    """What a miscalibrated probability costs, in currency.

    Expected loss is `P(default) x exposure x LGD`. If the probabilities are 8 points high
    on average, the provision is 8 points too large on every account in the book — and the
    error is invisible to AUC, accuracy, precision and recall alike.
    """
    predicted_loss = float(np.sum(probabilities) * exposure * loss_given_default)
    actual_loss = float(np.sum(y_true) * exposure * loss_given_default)
    return {
        "predicted_loss": round(predicted_loss, 2),
        "actual_loss": round(actual_loss, 2),
        "error": round(predicted_loss - actual_loss, 2),
        "error_pct": round((predicted_loss - actual_loss) / max(actual_loss, 1e-9), 4),
    }
