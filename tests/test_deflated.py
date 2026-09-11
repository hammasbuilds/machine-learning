"""Tests for the Deflated Sharpe Ratio.

The important ones are the two that would have caught the units bug, and the one that
asserts the whole method works: a strategy with a real edge must survive deflation, and a
strategy found by searching noise must not. A correction that rejects everything is not a
correction, it is a refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "projects" / "01-backtest-overfitting")
)

from deflated import (  # noqa: E402
    annualise,
    deflated_sharpe,
    expected_max_sharpe,
    minimum_track_record_length,
    sharpe,
    sharpe_per_period,
)


def _returns(*, mean: float, sd: float, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


# --- units ----------------------------------------------------------------


def test_annualisation_is_root_periods():
    assert annualise(0.039, periods_per_year=252) == pytest.approx(0.619, abs=1e-3)


def test_sharpe_is_the_annualised_form_of_sharpe_per_period():
    r = _returns(mean=0.0004, sd=0.01, n=1000)
    assert sharpe(r) == pytest.approx(annualise(sharpe_per_period(r)), rel=1e-12)


def test_track_record_length_uses_per_period_units():
    """The bug that shipped: an annualised Sharpe here gives an answer 252x too small.

    A daily Sharpe of 0.039 (annualised 0.62) needs a few years of data. Feeding 0.62 in
    directly returns about a fortnight, which looks like an answer rather than an error.
    """
    per_period = minimum_track_record_length(0.039)
    annualised_by_mistake = minimum_track_record_length(0.62)

    assert 1000 < per_period < 3000  # years, in trading days
    assert annualised_by_mistake < 30  # the wrong answer, asserted so it stays visible
    assert per_period / annualised_by_mistake > 100


def test_track_record_is_infinite_when_there_is_no_edge():
    assert minimum_track_record_length(0.0) == float("inf")
    assert minimum_track_record_length(-0.01) == float("inf")


# --- the expected maximum -------------------------------------------------


def test_more_trials_raise_the_bar():
    variance = 0.0004
    bars = [expected_max_sharpe(n, variance) for n in (2, 10, 100, 1000, 10000)]
    assert bars == sorted(bars)
    assert bars[-1] > 2 * bars[0]


def test_the_bar_grows_slowly():
    """sqrt(2 ln N): a hundredfold increase in trials should not double the bar twice."""
    variance = 0.0004
    assert expected_max_sharpe(10000, variance) < 2.5 * expected_max_sharpe(100, variance)


def test_a_single_trial_needs_no_deflation():
    assert expected_max_sharpe(1, 0.0004) == 0.0


def test_zero_variance_means_no_search_effect():
    assert expected_max_sharpe(1000, 0.0) == 0.0


# --- the verdict ----------------------------------------------------------


def test_a_genuine_edge_survives_deflation():
    """The control the method needs: it must not reject something real.

    A daily Sharpe near 0.19 (annualised ~3) over eight years is an enormous, obvious edge.
    Ten trials cannot explain it, and the correction must say so.
    """
    strong = _returns(mean=0.0019, sd=0.01, n=2000, seed=1)
    verdict = deflated_sharpe(strong, n_trials=10)
    assert verdict.survives
    assert verdict.probability_real > 0.95


def test_the_best_of_many_noise_strategies_does_not_survive():
    """Search 500 zero-edge series, keep the best, and the correction must reject it."""
    rng = np.random.default_rng(2)
    trials = [rng.normal(0.0, 0.01, 1500) for _ in range(500)]
    sharpes = np.array([sharpe(t) for t in trials])
    best = trials[int(np.argmax(sharpes))]

    verdict = deflated_sharpe(best, n_trials=len(trials), all_trial_sharpes=sharpes)
    assert not verdict.survives
    assert verdict.benchmark_sharpe > 0


def test_the_same_returns_are_judged_worse_the_more_you_searched():
    """The central claim, isolated: identical returns, different N, different verdict."""
    r = _returns(mean=0.0009, sd=0.01, n=1500, seed=3)
    few = deflated_sharpe(r, n_trials=3)
    many = deflated_sharpe(r, n_trials=50_000)

    assert few.observed_sharpe == pytest.approx(many.observed_sharpe)
    assert many.benchmark_sharpe > few.benchmark_sharpe
    assert many.probability_real < few.probability_real


def test_verdict_reports_annualised_but_keeps_the_per_period_figure():
    r = _returns(mean=0.0004, sd=0.01, n=800, seed=4)
    verdict = deflated_sharpe(r, n_trials=5)
    assert verdict.observed_sharpe == pytest.approx(annualise(verdict.per_period_sharpe))


def test_too_few_observations_is_an_error_not_a_number():
    with pytest.raises(ValueError):
        deflated_sharpe(np.array([0.01, 0.02]), n_trials=10)


def test_summary_is_json_safe():
    import json

    verdict = deflated_sharpe(_returns(mean=0.0005, sd=0.01, n=900, seed=5), n_trials=20)
    assert json.loads(json.dumps(verdict.summary()))["trials"] == 20
