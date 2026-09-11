"""Tests for projects 14-16: recommenders, dimensionality reduction, anomaly detection.

The theme across all three is that the headline metric is not the thing being claimed, so
most of these assert a *gap* between what a number says and what it means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for project in ("14-recommenders", "15-dimensionality", "16-anomaly"):
    sys.path.insert(0, str(ROOT / "projects" / project))

from anomaly import (  # noqa: E402
    LOF,
    Isolation,
    Mahalanobis,
    agreement_matrix,
    lift_at_k,
    precision_at_k,
    recall_at_k,
    spearman,
    top_k_overlap,
)
from recommend import (  # noqa: E402
    ALS,
    ItemKNN,
    Popularity,
    Random,
    build_interactions,
    discounted_gain,
    evaluate,
    temporal_split,
)
from reduce import (  # noqa: E402
    distance_correlation,
    nmf,
    pca,
    truncated_svd,
    trustworthiness,
)

# --- 14 · recommenders -----------------------------------------------------


def _toy_interactions(n_users: int = 60, n_items: int = 60, seed: int = 0) -> pd.DataFrame:
    """Users in two taste groups, each buying mostly from its own half of the catalogue."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        group = user % 2
        for _ in range(12):
            item = rng.integers(0, n_items // 2) + (group * n_items // 2)
            rows.append({"user": user, "item": int(item)})
    return pd.DataFrame(rows)


def _held_out(data, seed: int = 0) -> dict:
    """One held-out item per user, drawn from that user's own taste group.

    Built this way deliberately: a held-out set drawn uniformly from the catalogue is
    unrelated to what the user likes, and against it the *random* recommender wins. That is
    not a result about recommenders, it is a broken evaluation - and it is the first version
    of this test.
    """
    rng = np.random.default_rng(seed)
    half = data.n_items // 2
    return {
        user: {int(rng.integers(0, half) + (user % 2) * half)} for user in range(data.n_users)
    }


def test_popularity_is_a_real_baseline_not_a_formality():
    """The comparison every recommender paper skips: does the model beat 'show bestsellers'?"""
    frame = _toy_interactions()
    data = build_interactions(frame, user_col="user", item_col="item", min_user=2, min_item=2)
    held_out = _held_out(data)

    popularity = evaluate(Popularity().fit(data), data, held_out, k=5)
    random = evaluate(Random(seed=1).fit(data), data, held_out, k=5)

    assert popularity.precision >= random.precision
    # Popularity shows everyone the same list, so its catalogue coverage is tiny.
    assert popularity.coverage < random.coverage


def test_coverage_and_novelty_separate_two_models_that_score_the_same():
    """Accuracy alone cannot distinguish a useful recommender from a bestseller list."""
    frame = _toy_interactions(seed=2)
    data = build_interactions(frame, user_col="user", item_col="item", min_user=2, min_item=2)
    held_out = _held_out(data, seed=2)

    popularity = evaluate(Popularity().fit(data), data, held_out, k=10)
    knn = evaluate(ItemKNN(neighbours=10).fit(data), data, held_out, k=10)

    # Item-item recommends different things to different users; popularity cannot.
    assert knn.coverage > popularity.coverage
    assert knn.novelty > popularity.novelty


def test_item_knn_learns_the_taste_groups():
    """Cosine similarity should recover the block structure the data was built with."""
    frame = _toy_interactions(seed=3)
    data = build_interactions(frame, user_col="user", item_col="item", min_user=2, min_item=2)
    model = ItemKNN(neighbours=8).fit(data)

    half = data.n_items // 2
    # A user who has only bought from the first half should be recommended from it.
    recommended = model.recommend(0, k=5, seen={0, 1, 2})
    assert sum(i < half for i in recommended) >= 4


def test_als_runs_and_produces_distinct_recommendations_per_user():
    frame = _toy_interactions(seed=4)
    data = build_interactions(frame, user_col="user", item_col="item", min_user=2, min_item=2)
    model = ALS(factors=8, iterations=4, seed=0).fit(data)

    first = model.recommend(0, k=5, seen={0})
    second = model.recommend(1, k=5, seen={0})
    assert len(first) == 5
    assert first != second  # two users in different taste groups


def test_temporal_split_never_puts_a_later_row_in_training():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100, freq="D")})
    train, test = temporal_split(frame, date_col="date", quantile=0.8)
    assert train["date"].max() <= test["date"].min()
    assert len(train) + len(test) == len(frame)


def test_ndcg_rewards_position_and_precision_does_not():
    """The whole reason NDCG exists: nobody scrolls to position ten."""
    relevant = {7}
    assert discounted_gain([7, 1, 2, 3], relevant) > discounted_gain([1, 2, 3, 7], relevant)
    assert discounted_gain([1, 2, 3, 4], relevant) == 0.0


# --- 15 · dimensionality ---------------------------------------------------


def test_pca_without_scaling_follows_the_units():
    """The mistake that makes PCA a report on which column was recorded in larger numbers."""
    rng = np.random.default_rng(5)
    X = np.column_stack(
        [
            rng.normal(0, 1, 500),  # a variable in units of 1
            rng.normal(0, 1, 500) * 1000,  # the same variable, recorded in millimetres
            rng.normal(0, 1, 500),
        ]
    )
    unscaled = pca(X, k=3, scale=False)
    scaled = pca(X, k=3, scale=True)

    # Unscaled: the first component is almost entirely the large-unit column.
    assert unscaled.explained_variance_ratio[0] > 0.95
    # Scaled: all three contribute roughly equally, as they should.
    assert scaled.explained_variance_ratio[0] < 0.5


def test_pca_reconstruction_improves_monotonically_with_components():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 8))
    errors = [pca(X, k=k, scale=False).reconstruction_error(X) for k in (2, 4, 6, 8)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] == pytest.approx(0.0, abs=1e-8)


def test_nmf_produces_non_negative_factors():
    """The constraint is the entire point: parts that add, never subtract."""
    rng = np.random.default_rng(7)
    X = np.abs(rng.normal(size=(120, 10)))
    result = nmf(X, k=4, iterations=120, seed=0)
    assert (result.components >= 0).all()
    assert (result.embedding >= 0).all()


def test_truncated_svd_matches_pca_on_centred_data():
    """They are the same decomposition; the difference is only the centring step."""
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 6))
    centred = X - X.mean(axis=0)
    assert truncated_svd(centred, k=3).explained_variance_ratio == pytest.approx(
        pca(X, k=3, scale=False).explained_variance_ratio, abs=1e-8
    )


def test_trustworthiness_is_high_for_an_embedding_that_keeps_neighbours():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(200, 10))
    # An embedding that is just the first two PCA axes keeps local structure reasonably.
    good = pca(X, k=2).embedding
    scrambled = rng.permutation(good)
    assert trustworthiness(X, good, k=8) > trustworthiness(X, scrambled, k=8)


def test_distance_correlation_catches_a_relationship_pearson_misses():
    """Zero linear correlation, obvious dependence. The reason the metric exists."""
    rng = np.random.default_rng(10)
    x = rng.uniform(-1, 1, size=400)
    y = x**2
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.15
    assert distance_correlation(x[:, None], y[:, None]) > 0.3


# --- 16 · anomaly ----------------------------------------------------------


def _scattered(seed: int = 0, n: int = 1200, outliers: int = 40) -> tuple:
    """Outliers placed in random directions, so each one is genuinely alone."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    direction = rng.normal(size=(outliers, 4))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    X[:outliers] = direction * rng.uniform(6, 9, size=(outliers, 1))
    labels = np.zeros(n, dtype=bool)
    labels[:outliers] = True
    return X, labels


def test_every_detector_beats_random_inspection():
    """Precision at k against the base rate is the only evaluation an operations team uses."""
    X, labels = _scattered()
    for detector in (Isolation(contamination=0.05), LOF(neighbours=20), Mahalanobis()):
        assert lift_at_k(detector.fit_score(X), labels, k=40) > 3.0


def test_lof_is_blind_to_a_cluster_of_outliers():
    """The failure mode that follows directly from what LOF measures, and it is total.

    Shift forty points together and they become each other's neighbours. Their density
    relative to their neighbours is then perfectly normal, so LOF scores them as ordinary -
    at k=40, it finds **none** of them. Isolation Forest and Mahalanobis, which measure
    against the global distribution rather than the local one, find them immediately.

    This is not a tuning problem. It is what "local" means.
    """
    rng = np.random.default_rng(13)
    X = rng.normal(size=(1200, 4))
    X[:40] += 6.0  # one tight cluster, far away
    labels = np.zeros(1200, dtype=bool)
    labels[:40] = True

    assert precision_at_k(LOF(neighbours=20).fit_score(X), labels, k=40) == 0.0
    assert precision_at_k(Mahalanobis().fit_score(X), labels, k=40) > 0.9
    assert precision_at_k(Isolation(contamination=0.05).fit_score(X), labels, k=40) > 0.9


def test_detectors_disagree_on_what_is_unusual():
    """Three definitions of 'unusual'. Choosing one is a decision, not a hyperparameter."""
    rng = np.random.default_rng(11)
    # Two groups of genuinely different density - the case that separates local from global.
    X = np.vstack([rng.normal(size=(600, 3)), rng.normal(scale=0.15, size=(200, 3)) + 4.0])
    scores = {
        d.name: d.fit_score(X) for d in (Isolation(contamination=0.05), LOF(), Mahalanobis())
    }
    overlaps = [a.overlap for a in agreement_matrix(scores, k=80)]
    assert min(overlaps) < 0.6  # they do not pick the same 80 points


def test_precision_and_recall_at_k_are_consistent():
    scores = np.arange(100, dtype=float)
    labels = np.zeros(100, dtype=bool)
    labels[90:] = True  # the ten highest-scoring points are the real ones
    assert precision_at_k(scores, labels, k=10) == 1.0
    assert recall_at_k(scores, labels, k=10) == 1.0
    assert precision_at_k(scores, labels, k=20) == 0.5
    assert recall_at_k(scores, labels, k=20) == 1.0


def test_lift_is_one_for_a_useless_detector():
    rng = np.random.default_rng(12)
    labels = rng.binomial(1, 0.1, size=5000).astype(bool)
    assert lift_at_k(rng.normal(size=5000), labels, k=500) == pytest.approx(1.0, abs=0.35)


def test_top_k_overlap_and_spearman_agree_at_the_extremes():
    a = np.arange(200, dtype=float)
    assert top_k_overlap(a, a, k=20) == 1.0
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert top_k_overlap(a, -a, k=20) == 0.0
