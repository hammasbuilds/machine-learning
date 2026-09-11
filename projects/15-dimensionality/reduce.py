"""PCA, SVD, NMF, t-SNE, UMAP — what each one preserves, and what each one destroys.

Five methods that all "reduce dimensions" and answer completely different questions. They
get used interchangeably, and three of the resulting mistakes are so common they are worth
naming before any code:

**1. PCA on unscaled data measures your units.** Variance is not scale-free. Put income in
rupees next to age in years and the first component is income, not because income matters
more but because rupees are small. Change to lakhs and the answer changes. Scaling is not a
preprocessing nicety here; without it PCA is reporting your choice of units back to you.

**2. t-SNE cluster distances are meaningless.** It preserves *neighbourhoods*, explicitly at
the cost of global structure. Two clusters drawn far apart may be adjacent in the data and
two drawn adjacent may be unrelated. The cluster *sizes* are meaningless too — t-SNE
expands sparse regions and compresses dense ones by design. Every one of those is read off
the plot anyway.

**3. Explained variance is not information.** A component can carry 40% of the variance and
nothing you care about, particularly when one feature is noisy and large.

What each actually optimises:

  PCA    orthogonal directions of maximum variance. Linear, reversible, deterministic.
  SVD    the same decomposition without centring. On text this is LSA.
  NMF    parts that add up, never subtract. Interpretable *because* it cannot cancel.
  t-SNE  local neighbourhoods, at the explicit cost of global geometry.
  UMAP   local neighbourhoods with more global structure retained, and faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Decomposition:
    """A fitted linear reduction, with the pieces needed to judge it."""

    name: str
    components: np.ndarray  # (k, n_features)
    embedding: np.ndarray  # (n_samples, k)
    explained_variance_ratio: np.ndarray
    mean: np.ndarray | None

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def components_needed(self, share: float = 0.9) -> int:
        """How many components to retain `share` of the variance."""
        reached = np.flatnonzero(self.cumulative_variance >= share)
        return int(reached[0]) + 1 if len(reached) else len(self.explained_variance_ratio)

    def reconstruct(self) -> np.ndarray:
        out = self.embedding @ self.components
        return out + self.mean if self.mean is not None else out

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared error of putting it back together.

        The measure that lets linear methods be compared honestly. t-SNE and UMAP have no
        inverse and therefore no reconstruction error, which is itself the point: they are
        visualisations, not compressions, and cannot be used as a preprocessing step.
        """
        return float(np.mean((np.asarray(X, dtype=float) - self.reconstruct()) ** 2))


def pca(X: np.ndarray, k: int, *, scale: bool = True) -> Decomposition:
    """Principal components, via SVD of the centred matrix.

    `scale=False` exists to demonstrate the first mistake rather than to be used.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centred = X - mean

    if scale:
        sd = centred.std(axis=0)
        sd[sd == 0] = 1.0
        centred = centred / sd
        mean_used = mean  # kept for reporting; reconstruction below stays in scaled space
    else:
        mean_used = mean

    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    variance = S**2 / (len(X) - 1)

    return Decomposition(
        name="PCA" + ("" if scale else " (unscaled)"),
        components=Vt[:k],
        embedding=(U[:, :k] * S[:k]),
        explained_variance_ratio=variance[:k] / variance.sum(),
        mean=mean_used if not scale else None,
    )


def truncated_svd(X: np.ndarray, k: int) -> Decomposition:
    """SVD without centring. The right choice for sparse count data.

    Centring a sparse matrix destroys its sparsity — every zero becomes `-mean` — which on a
    term-document matrix turns a few million non-zeros into a few billion. LSA is exactly
    this applied to text.
    """
    X = np.asarray(X, dtype=float)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    variance = S**2
    return Decomposition(
        name="Truncated SVD",
        components=Vt[:k],
        embedding=U[:, :k] * S[:k],
        explained_variance_ratio=variance[:k] / variance.sum(),
        mean=None,
    )


def nmf(X: np.ndarray, k: int, *, iterations: int = 300, seed: int = 0) -> Decomposition:
    """Non-negative matrix factorisation by multiplicative updates (Lee & Seung).

    Everything is non-negative, so components can only **add**. That is the entire source of
    its interpretability: a PCA component saying "more of A, less of B" has no meaning for
    a basket of goods, while an NMF component is a set of things that co-occur — a theme.

    The cost is that it is non-convex, so the answer depends on the seed, and there is no
    natural ordering of components by importance.
    """
    X = np.clip(np.asarray(X, dtype=float), 0, None)
    rng = np.random.default_rng(seed)
    scale = np.sqrt(X.mean() / k) if X.mean() > 0 else 1.0

    W = np.abs(rng.normal(scale, scale * 0.1, (X.shape[0], k)))
    H = np.abs(rng.normal(scale, scale * 0.1, (k, X.shape[1])))
    eps = 1e-10

    for _ in range(iterations):
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        W *= (X @ H.T) / (W @ H @ H.T + eps)

    # No natural ordering, so order by how much of the reconstruction each part explains.
    weight = np.array([np.sum(np.outer(W[:, j], H[j]) ** 2) for j in range(k)])
    order = np.argsort(-weight)

    return Decomposition(
        name="NMF",
        components=H[order],
        embedding=W[:, order],
        explained_variance_ratio=weight[order] / max(weight.sum(), eps),
        mean=None,
    )


# --- neighbourhood preservation -------------------------------------------


def trustworthiness(X: np.ndarray, embedding: np.ndarray, *, k: int = 12) -> float:
    """Do points that are neighbours in the embedding correspond to neighbours in the data?

    The measure that lets t-SNE and UMAP be judged at all, since neither has a
    reconstruction error. 1.0 means every embedded neighbour was a true neighbour; lower
    means the picture has invented adjacencies that are not in the data.

    It says nothing about whether the *distances between clusters* survived — and nothing
    can, because for t-SNE they did not.
    """
    X = np.asarray(X, dtype=float)
    embedding = np.asarray(embedding, dtype=float)
    n = len(X)
    k = min(k, n - 2)

    def rank_matrix(data: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return np.argsort(np.argsort(distances, axis=1), axis=1)

    original_ranks = rank_matrix(X)
    embedded_neighbours = np.argsort(
        np.linalg.norm(embedding[:, None, :] - embedding[None, :, :], axis=2)
        + np.diag(np.full(n, np.inf)),
        axis=1,
    )[:, :k]

    penalty = 0.0
    for i in range(n):
        for j in embedded_neighbours[i]:
            r = original_ranks[i, j]
            if r >= k:
                penalty += r - k

    return float(1 - (2 / (n * k * (2 * n - 3 * k - 1))) * penalty)


def neighbourhood_overlap(X: np.ndarray, embedding: np.ndarray, *, k: int = 12) -> float:
    """Share of each point's k nearest neighbours that survive the projection.

    Simpler than trustworthiness and easier to explain: if 0.62, then roughly six of every
    ten nearest neighbours in the original space are still nearest neighbours in the picture.
    """

    def neighbours(data: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]

    a, b = neighbours(np.asarray(X, dtype=float)), neighbours(np.asarray(embedding, dtype=float))
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))


def distance_correlation(
    X: np.ndarray, embedding: np.ndarray, *, sample: int = 600, seed: int = 0
) -> float:
    """Correlation between pairwise distances before and after.

    **The number that exposes t-SNE.** A method preserving global geometry scores high; t-SNE
    scores low by construction, because it was never trying to. Reporting it next to
    trustworthiness shows the trade rather than asserting it.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    a = np.linalg.norm(X[idx][:, None] - X[idx][None, :], axis=2)
    b = np.linalg.norm(embedding[idx][:, None] - embedding[idx][None, :], axis=2)

    triangle = np.triu_indices(len(idx), k=1)
    return float(np.corrcoef(a[triangle], b[triangle])[0, 1])
