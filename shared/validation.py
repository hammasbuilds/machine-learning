"""Splitting and scoring, with the traps that make ordinary ML results wrong.

Nine of the ten projects in this repo depend on getting one of these three things right,
and all three are things a default `train_test_split` gets wrong:

  1. **Time.** Shuffling a time series lets the model learn from next week to predict
     last week. The score is excellent and the model is worthless.
  2. **Groups.** When one engine, one customer or one match appears in many rows,
     splitting by row puts the same entity on both sides. The model recognises the
     entity rather than learning the pattern.
  3. **Thresholds.** A classifier does not output a decision, it outputs a score.
     Choosing 0.5 is a decision nobody made on purpose, and on imbalanced data it is
     almost always the wrong one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np


def time_split(
    n: int, *, n_splits: int = 5, min_train: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window split: train on the past, test on the future, repeatedly.

    Unlike k-fold, each fold's training set is a prefix of the series, so no future
    information can reach the model. Unlike a single hold-out, it produces several
    estimates, which is the only way to see whether performance is stable or whether one
    lucky period is carrying the result.
    """
    if n_splits < 1:
        raise ValueError("need at least one split")
    fold = n // (n_splits + 1)
    if fold < 1:
        raise ValueError(f"{n} rows is too few for {n_splits} splits")

    for i in range(1, n_splits + 1):
        train_end = fold * i
        if train_end < min_train:
            continue
        test_end = min(fold * (i + 1), n)
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def purged_time_split(
    n: int, *, n_splits: int = 5, embargo: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding window with a gap between train and test.

    The gap matters whenever a label depends on a window of future data — a 5-day forward
    return, a 30-day remaining-life estimate. Without it the last training rows overlap
    the first test rows in *time*, even though they do not overlap in index, and that
    overlap leaks. Lopez de Prado calls the gap an embargo.
    """
    for train, test in time_split(n, n_splits=n_splits):
        if embargo:
            train = train[: max(0, len(train) - embargo)]
        if len(train):
            yield train, test


def group_split(
    groups: Sequence, *, test_fraction: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split so that no group appears on both sides.

    The failure this prevents: a turbofan dataset has ~200 cycles per engine. Split by
    row and the model sees engine 42 at cycle 100 in training and cycle 101 in test. It
    learns to recognise engine 42, scores beautifully, and cannot generalise to an engine
    it has never seen — which is the only situation that matters in production.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    cut = max(1, int(round(len(unique) * test_fraction)))
    held_out = set(shuffled[:cut].tolist())

    mask = np.array([g in held_out for g in groups])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


# --- scoring --------------------------------------------------------------


@dataclass(frozen=True)
class Threshold:
    """A decision rule chosen deliberately, with the cost it was chosen under."""

    value: float
    expected_cost: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def caught(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    def summary(self) -> dict:
        return {
            "threshold": round(self.value, 4),
            "expected_cost": round(self.expected_cost, 2),
            "recall": round(self.caught, 4),
            "false_alarms": self.false_positives,
        }


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray, *, cost_fn: float, cost_fp: float
) -> Threshold:
    """The threshold that minimises expected cost, not the one that maximises F1.

    On a fraud problem where a missed fraud costs $500 and a false alarm costs $5, the
    0.5 default is not a neutral choice — it silently asserts the two errors cost the
    same. They differ by a factor of a hundred, and the threshold should move by roughly
    that much.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)

    candidates = np.unique(np.round(scores, 4))
    best: Threshold | None = None

    for t in candidates:
        predicted = scores >= t
        tp = int(np.sum(predicted & y_true))
        fp = int(np.sum(predicted & ~y_true))
        fn = int(np.sum(~predicted & y_true))
        cost = fn * cost_fn + fp * cost_fp
        if best is None or cost < best.expected_cost:
            best = Threshold(float(t), cost, tp, fp, fn)

    assert best is not None
    return best


def brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of a probability. Lower is better; 0.25 is a coin flip.

    Reported throughout this repo alongside accuracy, because accuracy answers "was the
    call right" and Brier answers "was the confidence honest". A forecaster who says 90%
    and is right 60% of the time is accurate and badly calibrated, and only the second
    number shows it.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - y_true) ** 2))


def calibration_curve(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted probability against observed frequency, in bins.

    Returns (mean predicted, observed rate, count per bin). Perfect calibration is the
    diagonal. The count matters: a bin holding four samples tells you nothing, and
    plotting it at the same weight as a bin holding four thousand is how calibration
    plots mislead.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)

    predicted, observed, counts = [], [], []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        counts.append(n)
        predicted.append(float(probabilities[mask].mean()) if n else np.nan)
        observed.append(float(y_true[mask].mean()) if n else np.nan)

    return np.array(predicted), np.array(observed), np.array(counts)
