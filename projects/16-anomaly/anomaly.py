"""Anomaly detection, and the question it cannot answer.

Unsupervised anomaly detection has a structural problem that most write-ups quietly skip:

    **you cannot evaluate it without labels, and if you had labels you would not need it.**

Every paper that reports "AUC 0.94" for an unsupervised detector is evaluating on labels it
claims not to have. That is not dishonest — it is the only way to measure anything — but it
changes what the number means. It says *"on this dataset, where we know the answer, the
method found it."* It does not say the method will find the next one.

So this file is built to make the gap visible rather than hide it. AI4I has labels, they are
withheld from every detector, and used only at the end to ask a question the detectors never
see: **did the thing you flagged turn out to be the thing we cared about?**

Second problem, which follows from the first: **"anomalous" and "the failure mode I care
about" are different sets.** A machine running unusually cool is an anomaly. It is not a
failure. A detector optimising for statistical unusualness will find it, rank it highly, and
be correct and useless at the same time.

Three detectors, three definitions of "unusual":

  Isolation Forest   how few random splits does it take to isolate this point?
  LOF                is this point in a sparser neighbourhood than its neighbours are?
  Mahalanobis        how many standard deviations from the centre, accounting for covariance?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Isolation:
    """Random splits until a point is alone. Anomalies need fewer.

    The insight is that isolation is cheap for outliers and expensive for inliers, so the
    *average path length in a random tree* is itself an anomaly score. No distance metric,
    no density estimate, and it scales.
    """

    name = "Isolation Forest"

    def __init__(self, *, contamination: float = 0.05, seed: int = 0) -> None:
        self._model = IsolationForest(
            n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1
        )

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        # Negated so higher always means more anomalous, across all three detectors.
        return -self._model.fit(X).score_samples(X)


class LOF:
    """Local Outlier Factor: density relative to the neighbours' density.

    The only one of the three that handles *varying* density. A point in a sparse region is
    not anomalous if its neighbours are equally sparse — which is why a global distance
    threshold flags entire legitimate operating regimes and LOF does not.
    """

    name = "Local Outlier Factor"

    def __init__(self, *, neighbours: int = 35, contamination: float = 0.05) -> None:
        self._model = LocalOutlierFactor(
            n_neighbors=neighbours, contamination=contamination, novelty=False
        )

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        self._model.fit(X)
        return -self._model.negative_outlier_factor_


class Mahalanobis:
    """Distance from the centre, in units of the covariance.

    The classical answer, and it assumes one elliptical blob. Included because it is the
    baseline the other two have to beat, and because when it wins — which happens — the
    extra machinery was not needed.
    """

    name = "Mahalanobis distance"

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        return EmpiricalCovariance().fit(X).mahalanobis(X)


@dataclass(frozen=True)
class Agreement:
    """How much two detectors agree on what is unusual."""

    a: str
    b: str
    overlap: float  # share of the top-k flagged by both
    rank_correlation: float


def top_k_overlap(scores_a: np.ndarray, scores_b: np.ndarray, *, k: int) -> float:
    top_a = set(np.argsort(-np.asarray(scores_a))[:k].tolist())
    top_b = set(np.argsort(-np.asarray(scores_b))[:k].tolist())
    return len(top_a & top_b) / k


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(np.asarray(a)))
    rb = np.argsort(np.argsort(np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def agreement_matrix(scores: dict[str, np.ndarray], *, k: int) -> list[Agreement]:
    """Pairwise agreement between detectors.

    The number worth looking at before any label is touched. Three detectors that disagree
    completely are three different definitions of "unusual", and picking one of them is a
    decision about which definition matters — not a hyperparameter.
    """
    names = list(scores)
    return [
        Agreement(
            a=names[i],
            b=names[j],
            overlap=top_k_overlap(scores[names[i]], scores[names[j]], k=k),
            rank_correlation=spearman(scores[names[i]], scores[names[j]]),
        )
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]


def precision_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Of the k most anomalous points, how many were actually the thing we cared about?

    The only evaluation an operations team recognises: they can inspect k machines a week,
    and this is the share of those inspections that find something.
    """
    top = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(labels).astype(bool)[top].mean())


def recall_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    labels = np.asarray(labels).astype(bool)
    top = np.argsort(-np.asarray(scores))[:k]
    return float(labels[top].sum() / max(labels.sum(), 1))


def lift_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    base = np.asarray(labels).astype(bool).mean()
    return precision_at_k(scores, labels, k=k) / base if base > 0 else 0.0
