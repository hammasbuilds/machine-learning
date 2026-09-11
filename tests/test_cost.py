"""Tests for imbalanced-classification scoring and cost-based thresholds.

The ones that matter: average precision has to floor at the base rate (ROC-AUC does not),
and the chosen threshold has to follow the cost ratio rather than sitting at 0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "projects" / "03-imbalanced-maintenance")
)

from cost import (  # noqa: E402
    auc,
    average_precision,
    best_threshold,
    cost_curve,
    evaluate_at,
    precision_recall_curve,
    roc_curve,
)


def _imbalanced(n: int = 2000, rate: float = 0.03, seed: int = 0):
    """Labels plus a score that is informative but far from perfect."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < rate
    scores = np.clip(rng.normal(np.where(y, 0.6, 0.25), 0.18), 0, 1)
    return y, scores


# --- the floors -----------------------------------------------------------


def test_roc_auc_of_a_coin_flip_is_one_half_whatever_the_imbalance():
    """ROC-AUC has the same floor at every base rate. That is its whole appeal.

    Rates are kept at 5% and above so the estimate is stable - see the test below for what
    happens underneath that.
    """
    rng = np.random.default_rng(1)
    for rate in (0.5, 0.2, 0.05):
        y = rng.random(20000) < rate
        fpr, tpr = roc_curve(y, rng.random(20000))
        assert auc(fpr, tpr) == pytest.approx(0.5, abs=0.03)


def test_roc_auc_gets_noisy_when_positives_are_scarce():
    """Written after this failed at a 0.5% rate, and the failure was the point.

    With ~20 positives in 4,000 rows, a random scorer swings between AUC 0.34 and 0.61
        across seeds - a spread wide enough that one draw looks like a broken model and another
        looks like a working one, with nothing separating them but luck. The standard error of
        an AUC estimate is governed by the number of *positives*, not the number of rows, so a
        huge imbalanced dataset can carry the same uncertainty as a tiny balanced one.

        The practical consequence: an AUC quoted without the positive count is not a result.
    """
    spread = []
    for seed in range(40):
        local = np.random.default_rng(seed)
        y = local.random(4000) < 0.005
        fpr, tpr = roc_curve(y, local.random(4000))
        spread.append(auc(fpr, tpr))

    scarce = np.array(spread)
    assert scarce.mean() == pytest.approx(0.5, abs=0.05)  # unbiased ...
    assert scarce.std() > 0.05  # ... and wide enough to mislead
    assert scarce.max() > 0.60  # measured: 0.609
    assert scarce.min() < 0.40  # measured: 0.338


def test_average_precision_floors_at_the_base_rate_not_one_half():
    """The property that makes AP the honest summary under imbalance."""
    rng = np.random.default_rng(2)
    for rate in (0.5, 0.05):
        y = rng.random(4000) < rate
        ap = average_precision(y, rng.random(4000))
        assert ap == pytest.approx(rate, abs=0.03)


def test_a_perfect_ranking_scores_one_on_both():
    y = np.array([False] * 90 + [True] * 10)
    scores = np.r_[np.linspace(0.0, 0.4, 90), np.linspace(0.6, 1.0, 10)]
    fpr, tpr = roc_curve(y, scores)
    assert auc(fpr, tpr) == pytest.approx(1.0, abs=1e-6)
    assert average_precision(y, scores) == pytest.approx(1.0, abs=1e-6)


def test_roc_moves_less_than_precision_when_false_alarms_are_added():
    """Why the two curves disagree: the false-positive rate has a huge denominator."""
    y = np.r_[np.ones(30, bool), np.zeros(2000, bool)]
    good = np.r_[np.full(30, 0.9), np.full(2000, 0.1)]

    noisy = good.copy()
    noisy[30:130] = 0.95  # a hundred confident false alarms

    fpr_a, tpr_a = roc_curve(y, good)
    fpr_b, tpr_b = roc_curve(y, noisy)
    roc_drop = auc(fpr_a, tpr_a) - auc(fpr_b, tpr_b)
    ap_drop = average_precision(y, good) - average_precision(y, noisy)

    assert ap_drop > roc_drop * 5


# --- curves ---------------------------------------------------------------


def test_precision_recall_curve_is_ordered_and_bounded():
    y, scores = _imbalanced()
    precision, recall, thresholds = precision_recall_curve(y, scores)
    assert np.all((precision >= 0) & (precision <= 1))
    assert np.all(np.diff(recall) >= -1e-12)  # recall never falls as the threshold drops
    assert np.all(np.diff(thresholds) <= 1e-12)  # thresholds descend


def test_roc_curve_starts_at_the_origin_and_ends_at_one():
    y, scores = _imbalanced()
    fpr, tpr = roc_curve(y, scores)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)


# --- the decision ---------------------------------------------------------


def test_flagging_nothing_costs_every_missed_positive():
    y = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    decision = evaluate_at(y, np.zeros(100), 0.5, cost_miss=500.0, cost_alarm=5.0)
    assert decision.false_negatives == 10
    assert decision.false_positives == 0
    assert decision.cost == 5000.0
    assert decision.recall == 0.0
    assert decision.accuracy == 0.9  # and still 90% accurate


def test_flagging_everything_costs_every_negative():
    y = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    decision = evaluate_at(y, np.ones(100), 0.5, cost_miss=500.0, cost_alarm=5.0)
    assert decision.false_negatives == 0
    assert decision.false_positives == 90
    assert decision.cost == 450.0


def test_an_expensive_miss_pushes_the_threshold_down():
    """The central claim: the cost ratio, not the data, decides where the threshold goes."""
    y, scores = _imbalanced(seed=3)
    cheap_miss = best_threshold(y, scores, cost_miss=10.0, cost_alarm=10.0)
    dear_miss = best_threshold(y, scores, cost_miss=5000.0, cost_alarm=100.0)
    assert dear_miss.threshold < cheap_miss.threshold
    assert dear_miss.recall > cheap_miss.recall


def test_the_chosen_threshold_beats_the_default_on_cost():
    y, scores = _imbalanced(seed=4)
    default = evaluate_at(y, scores, 0.5, cost_miss=5000.0, cost_alarm=100.0)
    chosen = best_threshold(y, scores, cost_miss=5000.0, cost_alarm=100.0)
    assert chosen.cost <= default.cost


def test_optimising_cost_can_make_accuracy_worse():
    """Recorded as a test because it is the finding people refuse to believe."""
    y, scores = _imbalanced(seed=5)
    default = evaluate_at(y, scores, 0.5, cost_miss=5000.0, cost_alarm=100.0)
    chosen = best_threshold(y, scores, cost_miss=5000.0, cost_alarm=100.0)
    assert chosen.cost < default.cost
    assert chosen.accuracy <= default.accuracy


def test_the_cost_curve_is_u_shaped():
    y, scores = _imbalanced(seed=6)
    thresholds, costs = cost_curve(y, scores, cost_miss=5000.0, cost_alarm=100.0, steps=120)
    trough = int(np.argmin(costs))
    assert 0 < trough < len(costs) - 1  # the minimum is interior, not at an end
    assert costs[0] > costs[trough] < costs[-1]


def test_summary_is_json_safe():
    import json

    y, scores = _imbalanced(seed=7)
    chosen = best_threshold(y, scores, cost_miss=5000.0, cost_alarm=100.0)
    assert "missed_failures" in json.loads(json.dumps(chosen.summary()))
