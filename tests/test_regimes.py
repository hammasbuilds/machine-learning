"""Tests for regime labelling.

The one that matters most: `realtime_labels` must never consult a future value. That is
the whole distinction the project rests on, and it is easy to break by an off-by-one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "projects" / "06-volatility-regimes")
)

from regimes import (  # noqa: E402
    CALM,
    NORMAL,
    STRESSED,
    compare,
    hindsight_labels,
    label_by_quantile,
    persistence,
    realtime_labels,
    runs,
    transition_matrix,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("1990-01-01", periods=len(values), freq="D"))


# --- labelling ------------------------------------------------------------


def test_values_fall_into_the_three_regimes():
    labels = label_by_quantile(np.array([1.0, 5.0, 9.0]), low=2.0, high=8.0)
    assert list(labels) == [CALM, NORMAL, STRESSED]


def test_the_cut_points_are_inclusive_on_both_sides():
    labels = label_by_quantile(np.array([2.0, 8.0]), low=2.0, high=8.0)
    assert list(labels) == [CALM, STRESSED]


def test_hindsight_labels_use_the_whole_series():
    """A late spike changes how early days are labelled - which is the point, and the flaw."""
    calm_only = _series([10.0] * 100)
    with_crisis = _series([10.0] * 100 + [80.0] * 100)

    assert set(hindsight_labels(calm_only)) <= {CALM, NORMAL, STRESSED}
    # The same early values, now judged against a history containing a crisis.
    assert hindsight_labels(with_crisis)[50] != STRESSED


# --- the property the project depends on ----------------------------------


def test_realtime_labels_never_look_forward():
    """The load-bearing test.

    Two series identical up to day 700 and wildly different afterwards must receive
    identical real-time labels for every day up to 700. If a future value can reach a past
    label, the whole comparison is meaningless.
    """
    rng = np.random.default_rng(0)
    shared = rng.uniform(10, 25, 700)

    quiet = _series(np.r_[shared, rng.uniform(10, 12, 300)])
    violent = _series(np.r_[shared, rng.uniform(70, 90, 300)])

    a = realtime_labels(quiet, warmup=500)
    b = realtime_labels(violent, warmup=500)
    assert np.array_equal(a[:700], b[:700])


def test_the_warmup_period_is_refused_not_guessed():
    series = _series(np.linspace(10, 40, 800))
    labels = realtime_labels(series, warmup=500)
    assert np.all(labels[:500] == -1)
    assert np.all(labels[500:] >= 0)


def test_a_rising_series_ends_stressed_in_real_time():
    series = _series(np.r_[np.full(600, 12.0), np.linspace(12, 60, 200)])
    labels = realtime_labels(series, warmup=500)
    assert labels[-1] == STRESSED


# --- structure ------------------------------------------------------------


def test_transition_rows_sum_to_one():
    labels = np.array([0, 0, 1, 1, 2, 2, 1, 0])
    matrix = transition_matrix(labels)
    for row in matrix:
        assert row.sum() == pytest.approx(1.0) or row.sum() == 0.0


def test_a_constant_regime_is_perfectly_persistent():
    labels = np.zeros(100, dtype=int)
    assert persistence(labels) == pytest.approx(1.0)
    assert transition_matrix(labels)[0, 0] == pytest.approx(1.0)


def test_an_alternating_series_has_no_persistence():
    labels = np.tile([0, 2], 50)
    assert persistence(labels) == pytest.approx(0.0)


def test_warmup_days_are_excluded_from_persistence():
    labels = np.r_[np.full(10, -1), np.zeros(10, dtype=int)]
    assert persistence(labels) == pytest.approx(1.0)


def test_runs_find_contiguous_stretches():
    labels = np.array([0, 0, 0, 1, 1, 2])
    frame = runs(labels)
    assert list(frame["regime"]) == [0, 1, 2]
    assert list(frame["length"]) == [3, 2, 1]


def test_runs_skip_the_warmup():
    labels = np.r_[np.full(5, -1), np.zeros(4, dtype=int)]
    frame = runs(labels)
    assert len(frame) == 1
    assert int(frame.iloc[0]["length"]) == 4


# --- comparison -----------------------------------------------------------


def test_identical_labellings_do_not_disagree():
    labels = np.array([0, 0, 1, 1, 2, 2] * 20)
    difference = compare(labels, labels)
    assert difference.days_disagreeing == 0
    assert difference.rate == 0.0


def test_disagreement_rate_counts_only_comparable_days():
    hindsight = np.zeros(100, dtype=int)
    realtime = np.r_[np.full(50, -1), np.ones(50, dtype=int)]
    difference = compare(hindsight, realtime)
    assert difference.days_compared == 50
    assert difference.days_disagreeing == 50
    assert difference.rate == pytest.approx(1.0)


def test_concentration_is_one_when_disagreement_is_uniform():
    """If the two labellings differ everywhere, turning points are not special."""
    hindsight = np.tile([0, 0, 0, 0, 2, 2, 2, 2], 40)
    realtime = 2 - hindsight
    difference = compare(hindsight, realtime, window=2)
    assert difference.concentration == pytest.approx(1.0, abs=0.05)
