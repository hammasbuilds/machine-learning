"""Recommenders, and the baseline that beats most of them.

Every recommender paper opens with a matrix factorisation and closes with an improvement
over some other matrix factorisation. The comparison nobody wants to run is:

    **recommend the most popular items to everybody.**

It requires no model, no training, no embeddings, and on offline metrics it is startlingly
hard to beat. Any recommender that cannot clear it is a maintenance cost with a dashboard.

Three more things this file is careful about, each of which quietly breaks offline
evaluation:

**1. Implicit feedback is not a rating.** A purchase is evidence of interest. A *non*-purchase
is not evidence of disinterest — it is mostly evidence that the customer never saw the item.
Treating unobserved cells as zeros trains the model to predict "no" for 99.9% of a catalogue
it was never shown.

**2. The split must be temporal.** Recommending an item a customer bought in March using the
fact that they bought it in June is not a recommendation, it is a lookup. Random splits on
interaction data leak the future into the past and every offline metric inflates.

**3. Popularity is a confound, not a feature.** Popular items are bought by everyone, so
co-occurrence statistics are dominated by them. Two obscure items bought by the same three
people are far stronger evidence than two bestsellers bought by the same three thousand —
and raw counts say the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Interactions:
    """A user-item matrix in coordinate form, plus the vocabularies to read it."""

    users: np.ndarray  # integer user index per interaction
    items: np.ndarray  # integer item index per interaction
    user_ids: list
    item_ids: list

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def __len__(self) -> int:
        return len(self.users)

    @property
    def density(self) -> float:
        """Share of the matrix actually observed. Usually a fraction of a percent."""
        return len(self) / max(self.n_users * self.n_items, 1)

    def to_matrix(self) -> np.ndarray:
        """Dense binary matrix. Fine at this scale; a real catalogue needs scipy.sparse."""
        matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        matrix[self.users, self.items] = 1.0
        return matrix

    def by_user(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for u, i in zip(self.users, self.items, strict=True):
            out.setdefault(int(u), set()).add(int(i))
        return out


def build_interactions(
    frame: pd.DataFrame, *, user_col: str, item_col: str, min_user: int = 5, min_item: int = 10
) -> Interactions:
    """Index users and items, dropping the long tail that cannot be evaluated.

    A customer with two purchases cannot have a held-out test item *and* a training history.
    An item bought once cannot be recommended to anyone but its buyer. Both are dropped —
    and the thresholds are stated because they change the numbers: a catalogue filtered to
    popular items makes every recommender look better, including the popularity baseline.
    """
    counts_user = frame[user_col].value_counts()
    counts_item = frame[item_col].value_counts()
    kept = frame[
        frame[user_col].isin(counts_user[counts_user >= min_user].index)
        & frame[item_col].isin(counts_item[counts_item >= min_item].index)
    ]

    user_ids = sorted(kept[user_col].unique())
    item_ids = sorted(kept[item_col].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {v: i for i, v in enumerate(item_ids)}

    return Interactions(
        users=kept[user_col].map(user_index).to_numpy(dtype=int),
        items=kept[item_col].map(item_index).to_numpy(dtype=int),
        user_ids=user_ids,
        item_ids=item_ids,
    )


def temporal_split(
    frame: pd.DataFrame, *, date_col: str, quantile: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before a cut-off date trains; everything after is held out.

    One date for the whole dataset, not a per-user split. A per-user "leave the last item
    out" split still lets the model see other users' futures, which on co-occurrence methods
    is a real leak — item A and item B look related because *next month* they were bought
    together.
    """
    cut = frame[date_col].quantile(quantile)
    return frame[frame[date_col] <= cut], frame[frame[date_col] > cut]


# --- the recommenders -----------------------------------------------------


class Popularity:
    """Recommend the most-bought items to everyone. The baseline that must be beaten."""

    name = "Popularity"

    def fit(self, data: Interactions) -> Popularity:
        counts = np.bincount(data.items, minlength=data.n_items)
        self._ranked = np.argsort(-counts)
        return self

    def recommend(self, user: int, k: int, seen: set[int]) -> list[int]:  # noqa: ARG002
        return [int(i) for i in self._ranked if int(i) not in seen][:k]


class ItemKNN:
    """Item-item collaborative filtering on cosine similarity.

    Cosine, not raw co-occurrence, and the normalisation is the whole method. Two items
    bought by the same three people out of five buyers each is far stronger evidence than
    two bestsellers sharing three thousand buyers out of a hundred thousand — raw counts
    rank the second pair higher, cosine ranks the first.
    """

    name = "Item-item CF (cosine)"

    def __init__(self, *, neighbours: int = 50) -> None:
        self.neighbours = neighbours

    def fit(self, data: Interactions) -> ItemKNN:
        matrix = data.to_matrix()
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = 1.0
        normalised = matrix / norms

        self._similarity = normalised.T @ normalised
        np.fill_diagonal(self._similarity, 0.0)  # an item is not its own recommendation

        # Keep only the strongest neighbours per item. Dense similarity rows are mostly
        # noise, and trimming them is what makes item-item work rather than average.
        if self.neighbours < data.n_items:
            cut = np.partition(self._similarity, -self.neighbours, axis=1)[:, -self.neighbours]
            self._similarity[self._similarity < cut[:, None]] = 0.0
        return self

    def recommend(self, user: int, k: int, seen: set[int]) -> list[int]:  # noqa: ARG002
        if not seen:
            return []
        scores = self._similarity[list(seen)].sum(axis=0)
        scores[list(seen)] = -np.inf
        return [int(i) for i in np.argsort(-scores)[:k]]


class ALS:
    """Implicit-feedback matrix factorisation, alternating least squares.

    Hu, Koren & Volinsky (2008). The idea that makes it work on purchases rather than
    ratings: every cell gets a target of 1 or 0, but with a **confidence** that scales with
    how much evidence there is. Unobserved cells are included with target 0 and low
    confidence — "probably not interested, but we never asked" — rather than excluded or
    treated as certain negatives.
    """

    name = "ALS matrix factorisation"

    def __init__(
        self,
        *,
        factors: int = 32,
        regularisation: float = 0.05,
        alpha: float = 20.0,
        iterations: int = 12,
        seed: int = 0,
    ) -> None:
        self.factors = factors
        self.regularisation = regularisation
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, data: Interactions) -> ALS:
        rng = np.random.default_rng(self.seed)
        preference = data.to_matrix()
        confidence = 1.0 + self.alpha * preference

        self._user_factors = rng.normal(0, 0.01, (data.n_users, self.factors))
        self._item_factors = rng.normal(0, 0.01, (data.n_items, self.factors))
        eye = self.regularisation * np.eye(self.factors)

        for _ in range(self.iterations):
            self._user_factors = self._solve(self._item_factors, preference, confidence, eye)
            self._item_factors = self._solve(self._user_factors, preference.T, confidence.T, eye)
        return self

    def _solve(
        self, fixed: np.ndarray, preference: np.ndarray, confidence: np.ndarray, eye: np.ndarray
    ) -> np.ndarray:
        # The Hu-Koren-Volinsky trick: precompute Y'Y once, then each row only needs the
        # correction for its own non-zero confidences.
        gram = fixed.T @ fixed
        out = np.zeros((preference.shape[0], fixed.shape[1]))

        for row in range(preference.shape[0]):
            c = confidence[row]
            nonzero = np.flatnonzero(preference[row])
            if len(nonzero) == 0:
                continue
            Y = fixed[nonzero]
            weighted = gram + (Y.T * (c[nonzero] - 1.0)) @ Y + eye
            target = (Y.T * c[nonzero]) @ preference[row][nonzero]
            out[row] = np.linalg.solve(weighted, target)
        return out

    def recommend(self, user: int, k: int, seen: set[int]) -> list[int]:
        scores = self._item_factors @ self._user_factors[user]
        scores[list(seen)] = -np.inf
        return [int(i) for i in np.argsort(-scores)[:k]]


class Random:
    """Uniformly random items. The floor, so every other number has a scale."""

    name = "Random"

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def fit(self, data: Interactions) -> Random:
        self._n = data.n_items
        return self

    def recommend(self, user: int, k: int, seen: set[int]) -> list[int]:  # noqa: ARG002
        candidates = [i for i in self._rng.permutation(self._n) if int(i) not in seen]
        return [int(i) for i in candidates[:k]]


# --- evaluation -----------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    """Accuracy, and the two things accuracy hides."""

    model: str
    precision: float
    recall: float
    ndcg: float
    coverage: float  # share of the catalogue that ever gets recommended
    novelty: float  # mean unpopularity of what is recommended
    users_evaluated: int

    def summary(self) -> dict:
        return {
            "model": self.model,
            "precision@k": round(self.precision, 4),
            "recall@k": round(self.recall, 4),
            "ndcg@k": round(self.ndcg, 4),
            "catalogue_coverage": round(self.coverage, 4),
            "novelty": round(self.novelty, 3),
            "users": self.users_evaluated,
        }


def discounted_gain(recommended: list[int], relevant: set[int]) -> float:
    """NDCG: credit for hits, discounted by how far down the list they appear.

    Precision treats position 1 and position 10 identically. Nobody scrolls to position 10.
    """
    gain = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(recommended) if item in relevant)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), len(recommended))))
    return gain / ideal if ideal > 0 else 0.0


def evaluate(
    model, train: Interactions, test_by_user: dict[int, set[int]], *, k: int = 10
) -> Scores:
    """Score a fitted model on held-out interactions.

    `coverage` and `novelty` are reported next to accuracy deliberately. A recommender that
    shows everyone the same ten bestsellers can score respectably on precision while being
    commercially useless — it recommends what customers would have found anyway and never
    surfaces the long tail that the catalogue exists to sell.
    """
    seen_by_user = train.by_user()
    counts = np.bincount(train.items, minlength=train.n_items).astype(float)
    popularity = counts / max(counts.sum(), 1)

    precisions, recalls, gains, recommended_items = [], [], [], set()
    novelty_terms = []

    for user, relevant in test_by_user.items():
        if user >= train.n_users or not relevant:
            continue
        seen = seen_by_user.get(user, set())
        recommended = model.recommend(user, k, seen)
        if not recommended:
            continue

        hits = len(set(recommended) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
        gains.append(discounted_gain(recommended, relevant))
        recommended_items.update(recommended)
        # Self-information: rare items score high, bestsellers score near zero.
        novelty_terms.append(
            float(np.mean([-np.log2(max(popularity[i], 1e-12)) for i in recommended]))
        )

    return Scores(
        model=model.name,
        precision=float(np.mean(precisions)) if precisions else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(gains)) if gains else 0.0,
        coverage=len(recommended_items) / max(train.n_items, 1),
        novelty=float(np.mean(novelty_terms)) if novelty_terms else 0.0,
        users_evaluated=len(precisions),
    )
