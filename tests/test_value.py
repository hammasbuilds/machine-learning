"""Tests for RFM, concentration and cohort retention.

The ones that earn their place: returns must reduce a customer's value rather than being
dropped, invoices must be counted instead of line items, and the cohort triangle must not
be mistaken for churn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "projects" / "04-customer-value"))

from value import (  # noqa: E402
    cohorts,
    concentration,
    gini,
    one_time_buyers,
    rfm,
    value_at_risk,
)


def _ledger(rows) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["Invoice", "CustomerID", "InvoiceDate", "Quantity", "Price"]
    )
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    frame["Invoice"] = frame["Invoice"].astype(str)
    frame["is_return"] = frame["Invoice"].str.startswith("C")
    frame["revenue"] = frame["Quantity"] * frame["Price"]
    return frame


# --- RFM ------------------------------------------------------------------


def test_a_return_reduces_the_customers_value():
    """The rule that decides whether this analysis is honest.

    Drop the negative rows and this customer is worth 500. Keep them and she is worth 20,
    which is the amount the business actually received.
    """
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 100, 5.0),  # +500
            ("C1002", 1.0, "2011-02-05", -96, 5.0),  # -480
        ]
    )
    frame = rfm(ledger)
    assert frame.loc[1.0, "monetary"] == pytest.approx(20.0)
    assert frame.loc[1.0, "returned"] == pytest.approx(480.0)


def test_frequency_counts_invoices_not_line_items():
    """A basket of forty products is one purchase, not forty."""
    ledger = _ledger([("1001", 1.0, "2011-01-05", 1, 10.0) for _ in range(40)])
    assert rfm(ledger).loc[1.0, "frequency"] == 1


def test_two_separate_orders_count_twice():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 10.0),
            ("1002", 1.0, "2011-03-05", 1, 10.0),
        ]
    )
    assert rfm(ledger).loc[1.0, "frequency"] == 2


def test_recency_is_never_zero_for_the_most_recent_buyer():
    """Defaulting `as_of` to the last transaction date gives recency 0, and something
    downstream eventually divides by it."""
    ledger = _ledger([("1001", 1.0, "2011-12-09", 1, 10.0)])
    assert rfm(ledger).loc[1.0, "recency_days"] >= 1


def test_customers_without_an_id_are_excluded():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 10.0),
            ("1002", None, "2011-01-06", 1, 10.0),
        ]
    )
    frame = rfm(ledger)
    assert len(frame) == 1
    assert 1.0 in frame.index


def test_average_order_value_divides_by_orders():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 100.0),
            ("1002", 1.0, "2011-02-05", 1, 300.0),
        ]
    )
    assert rfm(ledger).loc[1.0, "avg_order_value"] == pytest.approx(200.0)


# --- concentration --------------------------------------------------------


def test_equal_customers_give_a_gini_of_zero():
    values = pd.Series([100.0] * 50)
    assert gini(values) == pytest.approx(0.0, abs=0.02)


def test_one_customer_owning_everything_pushes_gini_toward_one():
    values = pd.Series([1.0] * 99 + [100000.0])
    assert gini(values) > 0.9


def test_concentration_is_a_cumulative_share_ending_at_one():
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    curve = concentration(values)
    assert curve["revenue"].iloc[-1] == pytest.approx(1.0)
    assert curve["customers"].iloc[-1] == pytest.approx(1.0)
    assert curve["revenue"].is_monotonic_increasing


def test_the_richest_customer_comes_first():
    values = pd.Series([1.0, 99.0])
    curve = concentration(values)
    assert curve["revenue"].iloc[0] == pytest.approx(0.99)


def test_one_time_buyers_are_counted_as_a_share():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 10.0),
            ("1002", 2.0, "2011-01-05", 1, 10.0),
            ("1003", 2.0, "2011-02-05", 1, 10.0),
        ]
    )
    assert one_time_buyers(rfm(ledger)) == pytest.approx(0.5)


# --- cohorts --------------------------------------------------------------


def test_a_cohort_starts_at_one_hundred_percent():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 10.0),
            ("1002", 2.0, "2011-01-06", 1, 10.0),
        ]
    )
    assert cohorts(ledger).rates[0].iloc[0] == pytest.approx(1.0)


def test_retention_falls_when_half_the_intake_does_not_return():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 10.0),
            ("1002", 2.0, "2011-01-06", 1, 10.0),
            ("1003", 1.0, "2011-02-05", 1, 10.0),  # only customer 1 comes back
        ]
    )
    rates = cohorts(ledger).rates
    assert rates[1].iloc[0] == pytest.approx(0.5)


def test_a_customer_is_assigned_to_the_month_of_their_first_purchase():
    ledger = _ledger(
        [
            ("1001", 1.0, "2011-01-05", 1, 10.0),
            ("1002", 1.0, "2011-06-05", 1, 10.0),
        ]
    )
    group = cohorts(ledger)
    assert str(group.rates.index[0]) == "2011-01"
    assert group.counts.loc[group.counts.index[0], 5] == 1  # five months later


def test_missing_later_months_are_absent_not_zero():
    """The cohort triangle is a window artefact, not churn, and the two must not be
    conflated: a blank means 'not observed yet', a zero means 'nobody came back'."""
    ledger = _ledger([("1001", 1.0, "2011-12-01", 1, 10.0)])
    rates = cohorts(ledger).rates
    assert 0 in rates.columns
    assert 6 not in rates.columns  # never observed, so never fabricated


# --- operational ----------------------------------------------------------


def test_value_at_risk_finds_dormant_repeat_buyers_only():
    ledger = _ledger(
        [
            ("1001", 1.0, "2010-01-05", 1, 500.0),  # repeat, long dormant
            ("1002", 1.0, "2010-02-05", 1, 500.0),
            ("1003", 2.0, "2011-12-01", 1, 500.0),  # repeat, recent
            ("1004", 2.0, "2011-12-05", 1, 500.0),
            ("1005", 3.0, "2010-01-05", 1, 500.0),  # dormant but bought only once
        ]
    )
    at_risk = value_at_risk(rfm(ledger), dormant_days=180)
    assert list(at_risk.index) == [1.0]


def test_value_at_risk_is_sorted_by_what_is_at_stake():
    """Note the anchor row: dormancy is measured against the end of the data, so a ledger
    that stops in February has nobody dormant in it. Getting this wrong in the test was a
    reminder that `as_of` is a property of the dataset, not of the calendar."""
    ledger = _ledger(
        [
            ("1001", 1.0, "2010-01-05", 1, 10.0),
            ("1002", 1.0, "2010-02-05", 1, 10.0),
            ("1003", 2.0, "2010-01-05", 1, 900.0),
            ("1004", 2.0, "2010-02-05", 1, 900.0),
            ("1005", 9.0, "2011-12-09", 1, 5.0),  # anchors as_of to the end of the window
        ]
    )
    at_risk = value_at_risk(rfm(ledger), dormant_days=180)
    assert list(at_risk.index) == [2.0, 1.0]
