"""Regularisation paths, and why "the model selected these features" is not a finding.

L1 regularisation is sold as two things at once: a way to control overfitting, and a way to
discover which variables matter. It is reliable at the first and **unreliable at the second**,
and the gap between those two claims is where most feature-importance stories go wrong.

The mechanism is not subtle. When two features are correlated, the lasso penalty is close to
indifferent between them - putting the weight on A, on B, or splitting it all cost about the
same. So it picks one, by a margin that is tiny compared to what either of them means.

The obvious diagnostic is to bootstrap and see whether the choice moves. On the data here it
mostly does not - and **that is the trap**, not the reassurance it looks like. The tie is
broken by a property of the full dataset, so every resample breaks it the same way. A feature
selected in 100% of resamples can still be a coin that landed once and was never re-flipped.

    **Reproducible is not the same as meaningful.**

Three things this file measures that a single fit cannot show:

  1. **Selection stability.** Refit on bootstrap resamples and count how often each feature
     survives. A feature selected in 51% of resamples was selected by a coin - but a feature
     at 100% has only been shown to be *consistently* chosen, which is a weaker claim than it
     sounds when its near-duplicate sits at 1%.
  2. **Sign stability.** A coefficient that changes sign across resamples is not evidence
     about direction, however large it is in the one fit you ran.
  3. **Scaling.** The penalty is applied to coefficients, and coefficients are in the units
     of their features. Forget to standardise and you have penalised variables in proportion
     to the arbitrary units they were recorded in.

Ridge, lasso and elastic net are all fitted, and the point of the comparison is that their
*predictive* performance is indistinguishable while their *selections* are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Fit:
    """One fitted model: its coefficients and what it scored."""

    name: str
    coefficients: np.ndarray
    columns: list[str]
    train_auc: float
    test_auc: float

    @property
    def selected(self) -> np.ndarray:
        """Which coefficients survived. The threshold matters and so it is explicit."""
        return np.abs(self.coefficients) > 1e-6

    @property
    def n_selected(self) -> int:
        return int(self.selected.sum())


def fit_penalised(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    columns: list[str],
    *,
    name: str,
    penalty: str,
    C: float,
    l1_ratio: float | None = None,
) -> Fit:
    """Logistic regression with an explicit penalty. saga handles all three."""
    model = LogisticRegression(
        penalty=penalty, C=C, l1_ratio=l1_ratio, solver="saga", max_iter=5000, tol=1e-4
    ).fit(X_train, y_train)
    return Fit(
        name=name,
        coefficients=model.coef_.ravel(),
        columns=columns,
        train_auc=float(roc_auc_score(y_train, model.decision_function(X_train))),
        test_auc=float(roc_auc_score(y_test, model.decision_function(X_test))),
    )


def regularisation_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    columns: list[str],
    *,
    strengths: np.ndarray,
    penalty: str = "l1",
) -> pd.DataFrame:
    """Every coefficient at every penalty strength, plus the score at each one.

    The path is the useful object, not the single fit. It shows the order in which features
    drop out - and whether the test score cares, which it usually does not until the very end.
    """
    rows = []
    for C in strengths:
        fit = fit_penalised(
            X_train,
            y_train,
            X_test,
            y_test,
            columns,
            name=f"C={C:g}",
            penalty=penalty,
            C=C,
        )
        for column, coefficient in zip(columns, fit.coefficients, strict=True):
            rows.append(
                {
                    "C": float(C),
                    "column": column,
                    "coefficient": float(coefficient),
                    "n_selected": fit.n_selected,
                    "train_auc": fit.train_auc,
                    "test_auc": fit.test_auc,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_selection(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    *,
    C: float,
    replicates: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Refit the lasso on resampled data and record what survives each time.

    This is Meinshausen & Buhlmann's stability selection, used here as a diagnostic rather
    than a selection method. The number that matters is not which features were chosen once,
    but **how often** - a feature at 50% was chosen by noise, and a feature at 99% was not.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    selections = np.zeros((replicates, len(columns)), dtype=bool)
    signs = np.zeros((replicates, len(columns)))

    for replicate in range(replicates):
        index = rng.integers(0, n, size=n)
        # A resample with one class missing cannot be fitted; redraw rather than skip, so
        # every replicate contributes and the count stays honest.
        while len(np.unique(y[index])) < 2:
            index = rng.integers(0, n, size=n)

        model = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=3000, tol=1e-3).fit(
            X[index], y[index]
        )
        coefficients = model.coef_.ravel()
        selections[replicate] = np.abs(coefficients) > 1e-6
        signs[replicate] = np.sign(coefficients)

    frequency = selections.mean(axis=0)
    # Of the replicates where the feature survived, how often did it point the same way as
    # the majority? Below 100% the "direction" of the effect is not a finding.
    consistency = []
    for j in range(len(columns)):
        kept = signs[selections[:, j], j]
        consistency.append(
            float(max((kept > 0).mean(), (kept < 0).mean())) if len(kept) else float("nan")
        )

    return pd.DataFrame(
        {
            "column": columns,
            "selected_share": frequency,
            "sign_consistency": consistency,
            "coin_flip": np.abs(frequency - 0.5) < 0.25,
        }
    ).sort_values("selected_share", ascending=False)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap between two selected sets. 1.0 means identical, 0.0 means disjoint."""
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 1.0


def pairwise_stability(
    X: np.ndarray, y: np.ndarray, *, C: float, replicates: int = 40, seed: int = 1
) -> np.ndarray:
    """Jaccard overlap between the feature sets chosen on different resamples.

    The single number to quote when someone says "the model told us these five variables
    matter": if two runs on resampled data agree on half the set, the set is not a result.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    sets = []

    for _ in range(replicates):
        index = rng.integers(0, n, size=n)
        while len(np.unique(y[index])) < 2:
            index = rng.integers(0, n, size=n)
        model = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=3000, tol=1e-3).fit(
            X[index], y[index]
        )
        sets.append(np.abs(model.coef_.ravel()) > 1e-6)

    return np.array(
        [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    )
