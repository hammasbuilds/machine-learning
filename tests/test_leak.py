"""Tests for the leakage audit.

The two that matter: a perfect leak has to be caught by the single-column AUC, and a
column being *strong* must never on its own be treated as a column being *leaky* — that
judgement needs someone who knows when the data was recorded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "projects" / "05-leakage"))

from leak import (  # noqa: E402
    Audit,
    audit_columns,
    expected_campaign_value,
    lift_at_k,
    single_column_auc,
)

# --- single-column AUC ----------------------------------------------------


def test_a_perfect_leak_scores_one():
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    perfect = np.r_[np.zeros(50), np.ones(50)]
    assert single_column_auc(perfect, y) == pytest.approx(1.0)


def test_a_perfectly_inverted_column_scores_zero():
    """Still a leak. Direction is irrelevant; an AUC of 0.02 is as damning as 0.98."""
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    inverted = np.r_[np.ones(50), np.zeros(50)]
    assert single_column_auc(inverted, y) == pytest.approx(0.0)


def test_an_unrelated_column_scores_one_half():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.3
    assert single_column_auc(rng.random(4000), y) == pytest.approx(0.5, abs=0.03)


def test_a_constant_column_is_uninformative():
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    assert single_column_auc(np.ones(100), y) == pytest.approx(0.5)


def test_ties_share_their_rank():
    """Without tie handling, a column with three distinct values scores differently
    depending on the order rows happen to arrive in."""
    y = np.array([False, False, True, True])
    values = np.array([1.0, 2.0, 2.0, 3.0])
    first = single_column_auc(values, y)
    order = np.array([3, 1, 2, 0])
    assert single_column_auc(values[order], y[order]) == pytest.approx(first)


def test_missing_values_are_skipped_not_imputed():
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    values = np.r_[np.zeros(50), np.ones(50)]
    values[0] = np.nan
    assert single_column_auc(values, y) == pytest.approx(1.0)


def test_a_single_class_cannot_be_separated():
    assert single_column_auc(np.arange(10.0), np.ones(10, bool)) == 0.5


# --- the verdict ----------------------------------------------------------


def test_a_column_absent_at_decision_time_is_a_leak_however_weak():
    """Availability decides it, not strength. A useless column you will not have is still
    a column you will not have."""
    weak_but_absent = Audit(column="duration", solo_auc=0.52, available_at_decision_time=False)
    assert weak_but_absent.verdict().startswith("LEAK")


def test_a_strong_column_that_you_will_have_is_only_flagged_for_a_look():
    """The distinction the whole audit rests on: strong is not the same as leaky.

    Plenty of legitimate features are powerful. Age predicts mortality; income predicts
    default. Flagging them as leaks would make the audit useless.
    """
    strong_and_present = Audit(column="balance", solo_auc=0.82, available_at_decision_time=True)
    assert strong_and_present.verdict() == "check - unusually strong for a single column"
    assert not strong_and_present.verdict().startswith("LEAK")


def test_an_ordinary_available_column_passes():
    ordinary = Audit(column="age", solo_auc=0.56, available_at_decision_time=True)
    assert ordinary.verdict() == "ok"


def test_suspicion_is_symmetric_around_one_half():
    assert Audit("a", 0.78, True).suspicious
    assert Audit("b", 0.22, True).suspicious
    assert not Audit("c", 0.55, True).suspicious


def test_audit_sorts_the_most_extreme_column_first():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(200, bool), np.ones(200, bool)]
    frame = pd.DataFrame(
        {
            "noise": rng.random(400),
            "leak": np.r_[rng.normal(0, 1, 200), rng.normal(6, 1, 200)],
            "weak": np.r_[rng.normal(0, 1, 200), rng.normal(0.3, 1, 200)],
        }
    )
    report = audit_columns(frame, y, known_at_decision_time={"noise", "weak"})
    assert report.iloc[0]["column"] == "leak"
    assert report.iloc[0]["verdict"].startswith("LEAK")


def test_non_numeric_columns_are_skipped():
    y = np.r_[np.zeros(5, bool), np.ones(5, bool)]
    frame = pd.DataFrame({"n": np.arange(10.0), "text": list("abcdefghij")})
    report = audit_columns(frame, y, known_at_decision_time={"n"})
    assert list(report["column"]) == ["n"]


# --- what it costs --------------------------------------------------------


def test_lift_of_a_random_ranking_is_about_one():
    rng = np.random.default_rng(2)
    y = rng.random(5000) < 0.12
    assert lift_at_k(y, rng.random(5000), k=0.1) == pytest.approx(1.0, abs=0.35)


def test_a_perfect_ranking_lifts_by_the_inverse_base_rate():
    """Calling the top 10% of a perfectly ranked list, with a 10% base rate, catches
    everyone - a tenfold lift, which is the ceiling."""
    y = np.r_[np.ones(100, bool), np.zeros(900, bool)]
    perfect = np.r_[np.ones(100), np.zeros(900)]
    assert lift_at_k(y, perfect, k=0.1) == pytest.approx(10.0, abs=0.1)


def test_campaign_value_counts_every_call_as_a_cost():
    y = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    perfect = np.r_[np.ones(10), np.zeros(90)]
    result = expected_campaign_value(
        y, perfect, k=0.1, value_per_subscription=80.0, cost_per_call=2.0
    )
    assert result["calls"] == 10
    assert result["subscriptions"] == 10
    assert result["profit"] == pytest.approx(10 * 80.0 - 10 * 2.0)


def test_a_worthless_ranking_still_pays_for_the_calls():
    y = np.r_[np.zeros(90, bool), np.ones(10, bool)]
    backwards = np.r_[np.ones(90), np.zeros(10)]
    result = expected_campaign_value(
        y, backwards, k=0.1, value_per_subscription=80.0, cost_per_call=2.0
    )
    assert result["subscriptions"] == 0
    assert result["profit"] < 0
