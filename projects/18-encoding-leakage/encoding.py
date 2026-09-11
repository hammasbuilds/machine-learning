"""Target encoding, and the leak that lives inside a single column.

High-cardinality categoricals - product codes, customer IDs, postcodes, merchant IDs - break
one-hot encoding. Four thousand stock codes become four thousand columns, most of which are
zero almost everywhere.

The standard answer is **target encoding**: replace each category with the mean of the target
for that category. One column instead of four thousand, and it usually works.

It also contains the most easily missed leak in applied machine learning:

    **the row you are encoding is inside the mean you are encoding it with.**

For a category that appears 500 times, one row in the average is a rounding error. For a
category that appears **once**, the encoded value *is* that row's target. The model does not
have to learn anything: the feature is the label, wearing a different name.

That is why target encoding a column of pure random noise produces a feature with enormous
apparent predictive power - and why the effect is strongest exactly where high-cardinality
encoding is most tempting, in the long tail of rare categories.

Three defences, in increasing order of correctness:

    train-only     compute the means on the training split only. Fixes the *test* score.
                   Does nothing about the training score, so the model still overfits.
    smoothing      shrink each category mean toward the global mean, weighted by how many
                   rows it has. Rare categories get pulled back to "we do not know".
    out-of-fold    encode each training row using the *other* folds' means, so no row ever
                   sees its own target. The only version that is actually correct.

This file implements all four, including the broken one, because the broken one is what most
tutorials show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def category_means(
    categories: np.ndarray, target: np.ndarray, *, prior: float, smoothing: float
) -> dict:
    """Per-category target mean, shrunk toward the global rate.

    The shrinkage is the empirical-Bayes weight: a category with `smoothing` observations is
    weighted half toward its own mean and half toward the prior. With `smoothing=0` this is
    the raw mean, which is what makes rare categories dangerous.
    """
    frame = pd.DataFrame({"category": categories, "target": target})
    grouped = frame.groupby("category")["target"].agg(["sum", "count"])
    weight = grouped["count"] / (grouped["count"] + smoothing)
    encoded = weight * (grouped["sum"] / grouped["count"]) + (1 - weight) * prior
    return encoded.to_dict()


@dataclass
class NaiveTargetEncoder:
    """Fit the means on **everything**, including the rows being encoded. The broken one.

    This is the version that appears in most tutorials and most notebooks. It is not subtly
    wrong - on a high-cardinality column it hands the model the answer.
    """

    name = "Naive (fit on all data)"
    smoothing: float = 0.0

    def fit_transform(
        self, train: np.ndarray, train_y: np.ndarray, test: np.ndarray, test_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # The leak: test targets go into the means the training rows are encoded with, and
        # each training row's own target goes into its own encoding.
        every = np.concatenate([train, test])
        every_y = np.concatenate([train_y, test_y])
        prior = float(every_y.mean())
        lookup = category_means(every, every_y, prior=prior, smoothing=self.smoothing)
        return (
            np.array([lookup.get(c, prior) for c in train]),
            np.array([lookup.get(c, prior) for c in test]),
        )


@dataclass
class TrainOnlyEncoder:
    """Means from the training split only. Fixes the test score; the model still overfits.

    The usual "fix", and it is half of one. No test target reaches the encoder, so the test
    score is no longer contaminated - but every training row is still encoded with a mean
    containing itself, so the model learns to trust a feature that will be weaker at serving
    time. The result is a model that is *worse*, not just measured more honestly.
    """

    name = "Train-only"
    smoothing: float = 0.0

    def fit_transform(
        self, train: np.ndarray, train_y: np.ndarray, test: np.ndarray, test_y: np.ndarray  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        prior = float(train_y.mean())
        lookup = category_means(train, train_y, prior=prior, smoothing=self.smoothing)
        return (
            np.array([lookup.get(c, prior) for c in train]),
            np.array([lookup.get(c, prior) for c in test]),
        )


@dataclass
class SmoothedEncoder(TrainOnlyEncoder):
    """Train-only, plus shrinkage toward the global rate.

    Rare categories are pulled back toward "we do not know", which is the correct thing to
    say about a stock code seen twice. It reduces the leak substantially without removing it.
    """

    name = "Train-only + smoothing"
    smoothing: float = 50.0


@dataclass
class OutOfFoldEncoder:
    """Encode each training row from the *other* folds. The version that is actually correct.

    No row ever contributes to its own encoded value. The training feature now has the same
    statistical properties as the serving feature - computed from data that does not include
    the row being scored - which is the only condition under which the model's estimate of
    how much to trust it is right.
    """

    name = "Out-of-fold"

    def __init__(self, *, folds: int = 5, smoothing: float = 20.0, seed: int = 0) -> None:
        self.folds = folds
        self.smoothing = smoothing
        self.seed = seed
        self.name = f"Out-of-fold ({folds} folds)"

    def fit_transform(
        self, train: np.ndarray, train_y: np.ndarray, test: np.ndarray, test_y: np.ndarray  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        prior = float(train_y.mean())
        encoded_train = np.full(len(train), prior)

        splitter = KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        for inner_fit, inner_encode in splitter.split(train):
            lookup = category_means(
                train[inner_fit], train_y[inner_fit], prior=prior, smoothing=self.smoothing
            )
            encoded_train[inner_encode] = [
                lookup.get(c, prior) for c in train[inner_encode]
            ]

        # Test rows are encoded from the full training set, which is correct: at serving time
        # all the training data is available and none of the test row's target is.
        full = category_means(train, train_y, prior=prior, smoothing=self.smoothing)
        return encoded_train, np.array([full.get(c, prior) for c in test])


ENCODERS = [NaiveTargetEncoder(), TrainOnlyEncoder(), SmoothedEncoder(), OutOfFoldEncoder()]


def cardinality_profile(categories: np.ndarray) -> pd.DataFrame:
    """How many categories appear once, twice, rarely - the surface the leak acts on.

    A column where 60% of categories appear fewer than five times is a column where target
    encoding is mostly copying labels.
    """
    counts = pd.Series(categories).value_counts()
    buckets = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 10**9)]
    rows = []
    for low, high in buckets:
        mask = (counts >= low) & (counts <= high)
        label = f"{low}" if low == high else (f"{low}+" if high > 10**8 else f"{low}-{high}")
        rows.append(
            {
                "appearances": label,
                "categories": int(mask.sum()),
                "rows": int(counts[mask].sum()),
                "share_of_rows": float(counts[mask].sum() / counts.sum()),
            }
        )
    return pd.DataFrame(rows)
