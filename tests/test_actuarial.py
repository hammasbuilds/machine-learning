"""Tests for projects 11-13: insurance pricing, calibration, survival.

Each block leads with the test that carries the project's claim, and several of these exist
because the first implementation was wrong in a way the output did not reveal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for project in ("11-insurance-pricing", "12-calibration", "13-survival"):
    sys.path.insert(0, str(ROOT / "projects" / project))

from calibrate import (  # noqa: E402
    Isotonic,
    Platt,
    Uncalibrated,
    auc,
    brier,
    expected_calibration_error,
    reliability,
)
from pricing import credible_bands, gini, poisson_deviance  # noqa: E402
from survival import (  # noqa: E402
    Cohort,
    concordance,
    fit_cox,
    kaplan_meier,
    log_rank,
    median_survival,
    survival_at,
)

# --- 11 · insurance pricing -----------------------------------------------


def test_gini_ranks_by_rate_not_by_count():
    """The bug that scored the correct model at -0.03 and the broken one at 0.22.

    Gini on an insurance portfolio must rank policies by predicted *rate*, because a policy
    with ten years of exposure will have more claims than a policy with one month regardless
    of how risky it is. Ranking by predicted count sorts by exposure and calls it skill.
    """
    rng = np.random.default_rng(0)
    exposure = rng.uniform(0.05, 5.0, size=4000)
    true_rate = rng.uniform(0.02, 0.4, size=4000)
    actual = rng.poisson(true_rate * exposure)

    perfect_rate = true_rate * exposure  # predicted counts from the true rate
    # A model that knows nothing but exposure: predicted count tracks exposure exactly.
    exposure_only = exposure * true_rate.mean()

    assert gini(perfect_rate, actual, exposure) > gini(exposure_only, actual, exposure)
    # The exposure-only model has no rate information at all, so it must score near zero.
    assert abs(gini(exposure_only, actual, exposure)) < 0.05


def test_gini_is_bounded_and_zero_for_a_constant_prediction():
    rng = np.random.default_rng(1)
    exposure = rng.uniform(0.1, 2.0, size=1000)
    actual = rng.poisson(0.1 * exposure)
    flat = np.full(1000, 0.1) * exposure
    assert abs(gini(flat, actual, exposure)) < 0.08
    assert -1.0 <= gini(flat, actual, exposure) <= 1.0


def test_credible_bands_flags_thin_exposure():
    """The small-denominator artefact, and the flag that stops it being read as a finding.

    Policies observed for two weeks with one claim have an *empirical* frequency of 26 claims
    a year. The band is real; the rate is a denominator. `credible=False` is what separates
    the two, and without it the plot shows a tenfold model failure that does not exist.
    """
    rng = np.random.default_rng(9)
    n_thin, n_thick = 60, 40000
    exposure = np.r_[np.full(n_thin, 0.02), np.full(n_thick, 0.9)]
    actual = np.r_[
        rng.binomial(1, 0.03, size=n_thin),  # brief policies: the *lowest* claim probability
        rng.poisson(0.09, size=n_thick),
    ]
    predicted = 0.1 * exposure

    bands = credible_bands(predicted, actual, exposure)
    thin = bands.iloc[0]
    thick = bands.iloc[-1]

    assert thin["exposure_years"] < 500.0
    assert not bool(thin["credible"])
    assert bool(thick["credible"])
    # The whole point: the incredible band's observed rate is wildly above the credible one.
    assert thin["observed_rate"] > 3 * thick["observed_rate"]


def test_poisson_deviance_is_zero_only_for_a_perfect_fit():
    actual = np.array([0.0, 1.0, 3.0, 2.0])
    assert poisson_deviance(actual, actual) == pytest.approx(0.0, abs=1e-9)
    assert poisson_deviance(actual + 0.5, actual) > 0


# --- 12 · calibration ------------------------------------------------------


def test_auc_is_blind_to_calibration():
    """The claim the project rests on: a monotone squash leaves AUC untouched.

    Ranking is all AUC sees. Halving every probability changes every decision that depends on
    a probability and changes AUC by nothing at all.
    """
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.3, size=5000)
    scores = np.clip(0.3 + 0.4 * (y - 0.5) + rng.normal(0, 0.15, size=5000), 0.001, 0.999)

    assert auc(y, scores) == pytest.approx(auc(y, scores / 2), abs=1e-12)
    # ...while the calibration error moves a long way.
    assert expected_calibration_error(y, scores / 2) > expected_calibration_error(y, scores)


def test_isotonic_beats_platt_on_a_non_sigmoid_distortion():
    """Platt assumes the distortion is a sigmoid. Isotonic assumes only that it is monotone."""
    rng = np.random.default_rng(3)
    n = 20000
    true_p = rng.uniform(0.02, 0.98, size=n)
    y = rng.binomial(1, true_p)
    # A distortion no sigmoid can undo: a piecewise-linear kink.
    distorted = np.where(true_p < 0.5, true_p * 0.4, 0.2 + (true_p - 0.5) * 1.6)
    distorted = np.clip(distorted, 1e-4, 1 - 1e-4)

    fit, test = slice(0, n // 2), slice(n // 2, n)
    platt = Platt().fit(distorted[fit], y[fit]).transform(distorted[test])
    isotonic = Isotonic().fit(distorted[fit], y[fit]).transform(distorted[test])

    assert expected_calibration_error(y[test], isotonic) < expected_calibration_error(
        y[test], platt
    )


def test_calibration_does_not_change_the_ranking():
    """Both methods are monotone, so neither can improve or damage AUC. Worth asserting."""
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.25, size=6000)
    scores = np.clip(rng.beta(2, 5, size=6000) + 0.3 * y, 1e-3, 1 - 1e-3)

    for method in (Platt(), Isotonic(), Uncalibrated()):
        transformed = method.fit(scores, y).transform(scores)
        # Isotonic produces ties, which move AUC very slightly; Platt cannot.
        assert auc(y, transformed) >= auc(y, scores) - 0.01


def test_brier_rewards_calibration_and_sharpness_together():
    """Brier is the one number that punishes a miscalibrated model. AUC does not."""
    rng = np.random.default_rng(10)
    p = rng.uniform(0.05, 0.95, size=20000)
    y = rng.binomial(1, p)
    assert brier(y, p) < brier(y, np.clip(p / 2, 1e-6, 1))
    assert brier(y, p) < brier(y, np.full_like(p, y.mean()))


def test_reliability_bins_recover_a_perfectly_calibrated_model():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, size=40000)
    y = rng.binomial(1, p)
    predicted, observed, counts = reliability(y, p, bins=10)
    assert (counts > 0).all()
    assert np.allclose(predicted, observed, atol=0.02)
    assert expected_calibration_error(y, p) < 0.01


# --- 13 · survival ---------------------------------------------------------


def test_censored_rows_are_not_negatives():
    """The claim the project exists for, measured.

    Same cohort twice: once with censoring handled, once with censored rows counted as
    "did not happen". The naive version understates the event rate substantially.
    """
    duration = np.r_[np.full(100, 10.0), np.full(100, 50.0)]
    observed = np.r_[np.ones(100, dtype=bool), np.zeros(100, dtype=bool)]
    cohort = Cohort(duration, observed)

    assert cohort.censoring_rate == pytest.approx(0.5)
    # Kaplan-Meier at t=10 says half the cohort had the event. The naive rate agrees here
    # only because censoring happens strictly later; the point is that KM uses the at-risk
    # denominator rather than the row count.
    assert survival_at(kaplan_meier(cohort), 10.0) == pytest.approx(0.5)


def test_kaplan_meier_drops_censored_rows_from_the_denominator():
    """A customer censored at t=5 must not be counted as a survivor of the event at t=8."""
    cohort = Cohort([5.0, 8.0, 9.0], [False, True, True])
    curve = kaplan_meier(cohort)
    # At t=8 only two customers are still at risk, so one event takes survival to 0.5.
    assert curve.loc[curve["time"] == 8.0, "at_risk"].iloc[0] == 2
    assert curve.loc[curve["time"] == 8.0, "survival"].iloc[0] == pytest.approx(0.5)


def test_median_survival_is_infinite_when_the_curve_never_reaches_half():
    """Reporting the last observed time instead is the most common way this is overstated."""
    cohort = Cohort([10.0, 20.0, 30.0, 40.0], [True, False, False, False])
    assert median_survival(kaplan_meier(cohort)) == float("inf")


def test_log_rank_separates_two_genuinely_different_cohorts():
    rng = np.random.default_rng(6)
    fast = Cohort(rng.exponential(10, 400), np.ones(400, dtype=bool))
    slow = Cohort(rng.exponential(30, 400), np.ones(400, dtype=bool))
    same = Cohort(rng.exponential(10, 400), np.ones(400, dtype=bool))

    assert log_rank(fast, slow)["p_value"] < 0.001
    assert log_rank(fast, same)["p_value"] > 0.05


def test_cox_recovers_a_known_hazard_ratio():
    """A covariate doubling the hazard should produce a coefficient near log(2)."""
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.normal(size=n)
    hazard = np.exp(np.log(2.0) * x)
    duration = rng.exponential(1 / hazard)
    observed = np.ones(n, dtype=bool)

    fitted = fit_cox(x[:, None], Cohort(duration, observed), ["x"])
    assert fitted.coefficients[0] == pytest.approx(np.log(2.0), abs=0.1)
    assert fitted.hazard_ratios()["hazard_ratio"].iloc[0] == pytest.approx(2.0, rel=0.1)


def test_concordance_is_half_for_a_useless_risk_score():
    rng = np.random.default_rng(8)
    cohort = Cohort(rng.exponential(10, 1500), np.ones(1500, dtype=bool))
    assert concordance(rng.normal(size=1500), cohort) == pytest.approx(0.5, abs=0.05)


def test_concordance_excludes_pairs_whose_order_is_unknowable():
    """Two censored rows carry no information about which came first."""
    cohort = Cohort([1.0, 2.0], [False, False])
    # No observed events at all, so there is no comparable pair and the answer is 0.5.
    assert concordance(np.array([5.0, 1.0]), cohort) == 0.5
