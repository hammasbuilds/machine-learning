"""Tests for Elo, the odds conversion, and the scoring rules.

The three that earn their place: overround removal (silently biases every comparison),
Brier separating forecasters accuracy cannot, and home advantage actually moving the
expectation in the right direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "projects" / "02-sports-calibration")
)

from elo import (  # noqa: E402
    AWAY,
    DRAW,
    HOME,
    Elo,
    accuracy,
    implied_probabilities,
    log_loss,
    multiclass_brier,
    overround,
    reliability,
    three_way_probabilities,
)

# --- Elo ------------------------------------------------------------------


def test_equal_teams_on_neutral_ground_are_a_coin_flip():
    model = Elo(home_advantage=0.0)
    assert model.expected_home_score("a", "b") == pytest.approx(0.5)


def test_home_advantage_favours_the_home_side():
    model = Elo(home_advantage=65.0)
    assert model.expected_home_score("a", "b") > 0.5


def test_the_same_fixture_reverses_only_on_neutral_ground():
    """Written wrong first time, and the failure was the informative part.

    I asserted P(a beats b at a) == 1 - P(b beats a at b). That is false whenever home
    advantage exists, because *both* sides receive it when they are at home: with equal
    teams both fixtures return 0.59, not 0.59 and 0.41. The symmetry is a property of the
    rating gap, and it only shows up once the venue effect is removed.
    """
    neutral = Elo(home_advantage=0.0)
    neutral.ratings = {"a": 1600.0, "b": 1450.0}
    assert neutral.expected_home_score("a", "b") == pytest.approx(
        1.0 - neutral.expected_home_score("b", "a")
    )

    with_venue = Elo(home_advantage=65.0)
    assert with_venue.expected_home_score("a", "b") == pytest.approx(
        with_venue.expected_home_score("b", "a")
    )  # equal teams, so each is favoured by the same amount at home


def test_a_win_raises_the_winner_and_lowers_the_loser_equally():
    model = Elo(k=20.0, home_advantage=0.0)
    model.update("a", "b", 2, 0)
    assert model.rating("a") > 1500 > model.rating("b")
    assert (model.rating("a") - 1500) == pytest.approx(1500 - model.rating("b"))


def test_beating_a_stronger_team_moves_the_rating_more():
    weak_beats_strong = Elo(k=20.0, home_advantage=0.0)
    weak_beats_strong.ratings = {"a": 1500.0, "b": 1900.0}
    weak_beats_strong.update("a", "b", 1, 0)
    underdog_gain = weak_beats_strong.rating("a") - 1500.0

    expected_win = Elo(k=20.0, home_advantage=0.0)
    expected_win.ratings = {"a": 1900.0, "b": 1500.0}
    expected_win.update("a", "b", 1, 0)
    favourite_gain = expected_win.rating("a") - 1900.0

    assert underdog_gain > favourite_gain


def test_margin_of_victory_has_diminishing_returns():
    """A 5-0 beats a 1-0, but not by five times - the second goal says more than the fifth."""
    gains = []
    for margin in (1, 2, 5):
        model = Elo(k=20.0, home_advantage=0.0, mov_factor=0.35)
        model.update("a", "b", margin, 0)
        gains.append(model.rating("a") - 1500.0)

    assert gains[0] < gains[1] < gains[2]
    assert gains[2] < 5 * gains[0]


def test_regression_pulls_toward_the_mean_without_crossing_it():
    model = Elo()
    model.ratings = {"strong": 1900.0, "weak": 1100.0}
    model.regress_to_mean(0.25)
    assert 1500 < model.rating("strong") < 1900
    assert 1100 < model.rating("weak") < 1500


# --- odds -----------------------------------------------------------------


def test_implied_probabilities_sum_to_one_after_the_margin_is_removed():
    odds = np.array([[2.0, 3.4, 4.0], [1.2, 7.0, 15.0]])
    p = implied_probabilities(odds)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_overround_detects_the_bookmakers_margin():
    """A fair book sums to 1.00. A real one sums to more, and that difference is the vig."""
    fair = np.array([[2.0, 4.0, 4.0]])  # 0.5 + 0.25 + 0.25
    assert overround(fair)[0] == pytest.approx(1.0)

    real = np.array([[1.95, 3.9, 3.9]])
    assert overround(real)[0] > 1.0


def test_removing_the_margin_lowers_every_probability():
    odds = np.array([[1.95, 3.9, 3.9]])
    raw = 1.0 / odds
    fair = implied_probabilities(odds)
    assert np.all(fair < raw)


def test_shorter_odds_mean_higher_probability():
    p = implied_probabilities(np.array([[1.2, 7.0, 15.0]]))[0]
    assert p[0] > p[1] > p[2]


# --- scoring --------------------------------------------------------------


def test_perfect_forecasts_score_zero_brier():
    outcomes = np.array([HOME, DRAW, AWAY])
    perfect = np.eye(3)[outcomes]
    assert multiclass_brier(perfect, outcomes) == pytest.approx(0.0)


def test_a_confident_mistake_is_the_worst_possible_brier():
    outcomes = np.array([HOME])
    assert multiclass_brier(np.array([[0.0, 0.0, 1.0]]), outcomes) == pytest.approx(2.0)


def test_brier_separates_forecasters_that_accuracy_ties():
    """The point of the project, as a test.

    Both forecasters call every match a home win, so both score identical accuracy. One is
    certain and one is honest about the base rate, and only Brier notices.
    """
    outcomes = np.array([HOME, HOME, DRAW, AWAY, HOME])
    certain = np.tile([1.0, 0.0, 0.0], (5, 1))
    honest = np.tile([0.45, 0.25, 0.30], (5, 1))

    assert accuracy(certain, outcomes) == accuracy(honest, outcomes)
    assert multiclass_brier(honest, outcomes) < multiclass_brier(certain, outcomes)


def test_log_loss_punishes_confident_errors_harder_than_brier():
    outcomes = np.array([HOME])
    slightly_wrong = np.array([[0.34, 0.33, 0.33]])
    badly_wrong = np.array([[0.01, 0.01, 0.98]])

    brier_ratio = multiclass_brier(badly_wrong, outcomes) / multiclass_brier(
        slightly_wrong, outcomes
    )
    loss_ratio = log_loss(badly_wrong, outcomes) / log_loss(slightly_wrong, outcomes)
    assert loss_ratio > brier_ratio


def test_three_way_probabilities_are_a_distribution():
    for expectation in (0.05, 0.3, 0.5, 0.7, 0.95):
        p = three_way_probabilities(expectation, draw_rate=0.25)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p > 0)


def test_a_stronger_expectation_shifts_probability_from_away_to_home():
    weak = three_way_probabilities(0.3, draw_rate=0.25)
    strong = three_way_probabilities(0.7, draw_rate=0.25)
    assert strong[HOME] > weak[HOME]
    assert strong[AWAY] < weak[AWAY]
    assert strong[DRAW] == pytest.approx(weak[DRAW])


def test_reliability_keeps_the_count_per_bin():
    outcomes = np.array([HOME, DRAW, AWAY] * 20)
    p = np.tile([0.45, 0.25, 0.30], (60, 1))
    curve = reliability(p, outcomes, bins=10)
    assert curve["count"].sum() == 60 * 3
    assert curve["count"].max() > 0
