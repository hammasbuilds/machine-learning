"""Customer segments, and whether they exist.

Every marketing deck has five customer segments with names like "Loyal Champions" and
"At-Risk Bargain Hunters". They come out of k-means, which will return five segments from
any data you give it, including data with no group structure whatsoever.

**k-means always succeeds.** Ask for five clusters and you get five clusters. The algorithm
has no way to report "there are no natural groups here, I have simply cut a continuous
cloud into five pieces" — and on customer RFM data, a continuous cloud is usually what it
has.

This module asks the question the elbow plot cannot:

    **run it again and do you get the same answer?**

Three checks, in increasing order of how much they hurt:

  reseed      same data, different random initialisation. If the segments move, the
              segments were an artefact of where the centroids happened to start.
  resample    80% of the customers, drawn again. If the segments move, they describe this
              sample rather than the business.
  null        the same features, independently shuffled column by column. This destroys
              every relationship between variables while keeping each variable's
              distribution exactly. **Whatever silhouette score survives that is the score
              structure-free data earns**, and a real result has to beat it.

Agreement is measured with the Adjusted Rand Index, which compares two partitions of the
same points and is corrected so that random agreement scores zero rather than something
comfortably positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def fit_labels(X: np.ndarray, k: int, *, seed: int) -> np.ndarray:
    """One k-means fit. `n_init=1` on purpose.

    scikit-learn defaults to ten restarts and keeps the best, which hides exactly the
    instability being measured here. The question is not "what is the best partition this
    algorithm can find" — it is "what does a practitioner get when they run it", and a
    practitioner who runs it twice with different seeds should be shown both answers.
    """
    model = KMeans(n_clusters=k, n_init=1, random_state=seed, max_iter=300)
    return model.fit_predict(X)


def reseed_agreement(X: np.ndarray, k: int, *, seeds: int = 12) -> np.ndarray:
    """Pairwise Adjusted Rand Index across runs that differ only in initialisation."""
    partitions = [fit_labels(X, k, seed=s) for s in range(seeds)]
    scores = [
        adjusted_rand_score(partitions[i], partitions[j])
        for i in range(len(partitions))
        for j in range(i + 1, len(partitions))
    ]
    return np.array(scores)


def resample_agreement(
    X: np.ndarray, k: int, *, rounds: int = 12, fraction: float = 0.8, seed: int = 0
) -> np.ndarray:
    """Agreement between the full-data partition and partitions from subsamples.

    Compared on the overlap only: a customer left out of a subsample has no label in that
    run, and scoring them would mean inventing one.
    """
    rng = np.random.default_rng(seed)
    reference = fit_labels(X, k, seed=0)
    n = len(X)

    scores = []
    for r in range(rounds):
        take = rng.choice(n, size=int(n * fraction), replace=False)
        subset = fit_labels(X[take], k, seed=r)
        scores.append(adjusted_rand_score(reference[take], subset))
    return np.array(scores)


def shuffled_columns(X: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Each column permuted independently: same marginals, no joint structure.

    The right null for clustering. Comparing against uniform random points is too easy a
    bar, because real features are skewed and skew alone creates apparent clumps. This
    keeps every marginal distribution intact and destroys only the relationships.
    """
    rng = np.random.default_rng(seed)
    out = np.array(X, dtype=float, copy=True)
    for column in range(out.shape[1]):
        out[:, column] = rng.permutation(out[:, column])
    return out


@dataclass(frozen=True)
class Verdict:
    """What a given k is worth once it has been asked to repeat itself."""

    k: int
    silhouette: float
    silhouette_on_shuffled: float
    reseed_ari: float
    resample_ari: float

    @property
    def beats_null(self) -> bool:
        return self.silhouette > self.silhouette_on_shuffled

    @property
    def stable(self) -> bool:
        """0.75 is the conventional bar for "substantially the same partition"."""
        return self.reseed_ari >= 0.75 and self.resample_ari >= 0.75

    def summary(self) -> dict:
        return {
            "k": self.k,
            "silhouette": round(self.silhouette, 4),
            "silhouette_shuffled": round(self.silhouette_on_shuffled, 4),
            "excess_over_null": round(self.silhouette - self.silhouette_on_shuffled, 4),
            "reseed_ari": round(self.reseed_ari, 4),
            "resample_ari": round(self.resample_ari, 4),
            "verdict": "stable and beats null"
            if (self.stable and self.beats_null)
            else "not reproducible",
        }


def assess(X: np.ndarray, k: int, *, sample_for_silhouette: int = 4000, seed: int = 0) -> Verdict:
    """Fit, then interrogate."""
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(X), sample_for_silhouette, replace=False)
        if len(X) > sample_for_silhouette
        else np.arange(len(X))
    )

    labels = fit_labels(X, k, seed=0)
    null = shuffled_columns(X, seed=seed)
    null_labels = fit_labels(null, k, seed=0)

    return Verdict(
        k=k,
        silhouette=float(silhouette_score(X[idx], labels[idx])),
        silhouette_on_shuffled=float(silhouette_score(null[idx], null_labels[idx])),
        reseed_ari=float(np.median(reseed_agreement(X, k))),
        resample_ari=float(np.median(resample_agreement(X, k))),
    )


def inertia_curve(X: np.ndarray, ks: range, *, seed: int = 0) -> np.ndarray:
    """Within-cluster sum of squares for each k — the elbow plot.

    Included to be argued with. It falls monotonically by construction, so it can never say
    "no", and on smooth data it has no elbow to find.
    """
    return np.array([KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X).inertia_ for k in ks])
