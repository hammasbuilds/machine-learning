"""Customer value on a real transaction ledger, where the average is the wrong number.

**Mean customer value is close to meaningless on retail data**, and this project exists to
show why with a million real invoice lines rather than an assertion.

The distribution of spend across customers is not bell-shaped. It is a long tail: most
customers buy once and disappear, a few buy every month for two years. Reporting the mean
describes a customer who does not exist, and any plan built on it — acquisition budget,
support staffing, discount depth — is built on a fiction.

Three things this file computes instead:

  RFM             recency, frequency, monetary value per customer. Old, unglamorous, and
                  still the strongest cheap signal for who will buy again.
  concentration   what share of revenue the top n% of customers actually produce.
  cohort survival which month a customer arrived, and how many of that intake were still
                  buying k months later. The only honest way to see retention.

Returns are kept throughout. A customer who bought £500 and sent £480 back is worth £20,
and any pipeline that drops rows with negative quantities will tell you they are worth
£500 and recommend finding more like them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rfm(transactions: pd.DataFrame, *, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency, frequency and monetary value per customer.

    `as_of` is the date the snapshot is taken. It defaults to the day after the last
    transaction in the data, which matters: using the last transaction date itself gives
    the most recent buyer a recency of zero and quietly divides by it later.
    """
    ledger = transactions.dropna(subset=["CustomerID"]).copy()
    as_of = as_of or (ledger["InvoiceDate"].max() + pd.Timedelta(days=1))

    grouped = ledger.groupby("CustomerID")
    frame = pd.DataFrame(
        {
            "recency_days": (as_of - grouped["InvoiceDate"].max()).dt.days,
            "tenure_days": (as_of - grouped["InvoiceDate"].min()).dt.days,
            # Distinct invoices, not line items. A basket of forty products is one purchase,
            # and counting lines makes bulk buyers look like frequent buyers.
            "frequency": grouped["Invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "returned": grouped.apply(
                lambda g: -g.loc[g["is_return"], "revenue"].sum(), include_groups=False
            ),
        }
    )
    frame["avg_order_value"] = frame["monetary"] / frame["frequency"].clip(lower=1)
    frame["return_rate"] = (frame["returned"] / frame["monetary"].abs().clip(lower=0.01)).clip(0, 1)
    return frame.sort_values("monetary", ascending=False)


def concentration(values: pd.Series) -> pd.DataFrame:
    """Cumulative share of revenue against cumulative share of customers.

    The Lorenz curve, under its retail name. Reported at decile boundaries because "the top
    10% produce x%" is a sentence somebody can act on, and a Gini coefficient is not.
    """
    ordered = values.sort_values(ascending=False).to_numpy()
    ordered = ordered[ordered > 0]  # net-negative customers break a cumulative share
    total = ordered.sum()

    share_customers = np.arange(1, len(ordered) + 1) / len(ordered)
    share_revenue = np.cumsum(ordered) / total
    return pd.DataFrame({"customers": share_customers, "revenue": share_revenue})


def gini(values: pd.Series) -> float:
    """0 = every customer worth the same, 1 = one customer is the entire business."""
    ordered = np.sort(values[values > 0].to_numpy())
    n = len(ordered)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * ordered)) / (n * np.sum(ordered)) - (n + 1) / n)


@dataclass(frozen=True)
class Cohorts:
    """Monthly intakes and what became of them."""

    counts: pd.DataFrame  # cohort month x months since, customers still active
    rates: pd.DataFrame  # the same, as a share of the intake
    revenue: pd.DataFrame  # revenue per cohort per month

    def average_survival(self) -> pd.Series:
        """Retention by month-since-first-purchase, averaged across cohorts.

        Cohorts are weighted equally rather than by size. A weighted average is dominated by
        whichever month happened to run the biggest campaign, which tells you about the
        campaign rather than about retention.
        """
        return self.rates.mean(axis=0, skipna=True)


def cohorts(transactions: pd.DataFrame) -> Cohorts:
    """Group customers by the month of their first purchase and follow them forward."""
    ledger = transactions.dropna(subset=["CustomerID"]).copy()
    ledger["month"] = ledger["InvoiceDate"].dt.to_period("M")

    first = ledger.groupby("CustomerID")["month"].min().rename("cohort")
    ledger = ledger.join(first, on="CustomerID")
    ledger["months_since"] = (ledger["month"].dt.year * 12 + ledger["month"].dt.month) - (
        ledger["cohort"].dt.year * 12 + ledger["cohort"].dt.month
    )

    counts = (
        ledger.groupby(["cohort", "months_since"])["CustomerID"].nunique().unstack(fill_value=0)
    )
    intake = counts[0].replace(0, np.nan)
    rates = counts.div(intake, axis=0)

    revenue = ledger.groupby(["cohort", "months_since"])["revenue"].sum().unstack(fill_value=0.0)
    return Cohorts(counts=counts, rates=rates, revenue=revenue)


def one_time_buyers(frame: pd.DataFrame) -> float:
    """Share of customers who placed exactly one order. Usually the largest single group."""
    return float((frame["frequency"] == 1).mean())


def value_at_risk(frame: pd.DataFrame, *, dormant_days: int = 180) -> pd.DataFrame:
    """Customers who were valuable and have gone quiet.

    The join that makes RFM operational. High monetary value plus high recency is not a
    churn statistic — it is a named list of people worth phoning, sorted by how much is at
    stake.
    """
    at_risk = frame[(frame["recency_days"] >= dormant_days) & (frame["frequency"] >= 2)]
    return at_risk.sort_values("monetary", ascending=False)
