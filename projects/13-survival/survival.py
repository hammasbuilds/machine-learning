"""Survival analysis: *when*, not *whether* — and the customers you must not call negatives.

The churn question is almost always asked wrongly. "Will this customer churn?" invites a
classifier, a cut-off date, and a label. It also invites the error that makes the whole
exercise worthless:

    **A customer who has not churned yet is not a negative.**

She is **censored**. All you know is that she had not gone by the time you stopped looking.
Label her 0 and you have told the model that eighteen months of loyalty and eighteen days
of it are the same outcome, and every estimate that follows is biased toward optimism —
because the people most likely to churn soon are the ones still being counted as retained.

Survival analysis handles this directly. Every customer contributes exactly what is known
about them:

    observed    bought again after 34 days      -> an event at t=34
    censored    no second purchase, 200 days    -> survived *at least* 200 days

Three tools, in increasing order of what they ask of you:

  Kaplan-Meier   the survival curve itself. No model, no assumptions about shape.
  log-rank       is the gap between two curves bigger than chance?
  Cox            which covariates move the hazard, and by how much. Semi-parametric:
                 it never states the baseline hazard, only how features multiply it.

This file implements all three directly rather than importing lifelines, because the
censoring logic *is* the lesson and burying it in a dependency hides it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Cohort:
    """Durations and whether each one ended in the event or in the window closing."""

    duration: np.ndarray  # time until the event, or until observation stopped
    observed: np.ndarray  # True = the event happened, False = censored

    def __post_init__(self) -> None:
        self.duration = np.asarray(self.duration, dtype=float)
        self.observed = np.asarray(self.observed, dtype=bool)

    def __len__(self) -> int:
        return len(self.duration)

    @property
    def censoring_rate(self) -> float:
        """Share of rows where the event had not happened when we stopped looking.

        Usually large. In this retail data it is around half, which is exactly why treating
        censored rows as negatives would be a catastrophe rather than a rounding error.
        """
        return float(1 - self.observed.mean())

    def naive_event_rate(self) -> float:
        """What a classifier would report, by counting censored rows as negatives.

        Included as the thing to be compared against, not as a method.
        """
        return float(self.observed.mean())


def kaplan_meier(cohort: Cohort) -> pd.DataFrame:
    """The survival curve, estimated without assuming any shape.

    At each time an event occurs, the estimator asks a small question: *of everyone still
    at risk right now, what fraction made it through?* Multiplying those conditional
    survivals gives S(t).

    The censoring handling is the whole method, and it is almost invisible: a censored
    customer stays in the at-risk denominator right up to the moment she leaves, then
    quietly drops out. She is never counted as an event and never counted as a survivor
    past the point where she stopped being observed.
    """
    times = np.sort(np.unique(cohort.duration[cohort.observed]))
    rows = []
    survival = 1.0
    variance_sum = 0.0  # Greenwood's formula, accumulated

    for t in times:
        at_risk = int(np.sum(cohort.duration >= t))
        events = int(np.sum((cohort.duration == t) & cohort.observed))
        if at_risk == 0:
            continue

        survival *= 1 - events / at_risk
        if at_risk > events:
            variance_sum += events / (at_risk * (at_risk - events))

        standard_error = survival * np.sqrt(variance_sum)
        rows.append(
            {
                "time": float(t),
                "at_risk": at_risk,
                "events": events,
                "survival": survival,
                "lower": max(0.0, survival - 1.96 * standard_error),
                "upper": min(1.0, survival + 1.96 * standard_error),
            }
        )

    return pd.DataFrame(rows)


def median_survival(curve: pd.DataFrame) -> float:
    """The time by which half the cohort has had the event.

    Returns inf when the curve never reaches 0.5 — which happens, and is the honest answer.
    Reporting the last observed time instead would claim knowledge the data does not have,
    and it is the most common way a survival result is overstated.
    """
    below = curve[curve["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float("inf")


def survival_at(curve: pd.DataFrame, t: float) -> float:
    """S(t): the share still event-free at time t."""
    before = curve[curve["time"] <= t]
    return float(before["survival"].iloc[-1]) if len(before) else 1.0


def log_rank(a: Cohort, b: Cohort) -> dict:
    """Is the gap between two survival curves bigger than chance?

    Compares observed events in group A against what would be expected if both groups
    shared one hazard. Uses every event time, which is why it beats comparing two medians:
    two curves can cross, share a median, and be completely different.
    """
    duration = np.r_[a.duration, b.duration]
    observed = np.r_[a.observed, b.observed]
    group = np.r_[np.zeros(len(a), dtype=bool), np.ones(len(b), dtype=bool)]

    observed_a = expected_a = variance = 0.0
    for t in np.sort(np.unique(duration[observed])):
        at_risk = duration >= t
        n = int(at_risk.sum())
        n_a = int((at_risk & ~group).sum())
        events = int(((duration == t) & observed).sum())
        events_a = int(((duration == t) & observed & ~group).sum())

        if n < 2 or events == 0:
            continue

        observed_a += events_a
        expected_a += events * n_a / n
        variance += (events * (n_a / n) * (1 - n_a / n) * (n - events)) / (n - 1)

    statistic = (observed_a - expected_a) ** 2 / variance if variance > 0 else 0.0
    return {
        "observed_a": observed_a,
        "expected_a": round(expected_a, 2),
        "chi2": round(float(statistic), 4),
        "p_value": round(float(1 - stats.chi2.cdf(statistic, df=1)), 6),
    }


# --- Cox proportional hazards ---------------------------------------------


@dataclass
class Cox:
    """Semi-parametric hazard model, fitted by Newton-Raphson on the partial likelihood.

    The trick that makes Cox work: the baseline hazard cancels out of the partial
    likelihood entirely. The model never has to say *what* the underlying risk of churning
    at week six is — only that this customer's risk is 1.4 times that one's, whatever it is.

    `exp(coefficient)` is a **hazard ratio**: 1.4 means 40% more likely to have the event in
    any given interval, among those who have survived to it.
    """

    columns: list[str]
    coefficients: np.ndarray
    log_likelihood: float
    standard_errors: np.ndarray
    n_observations: int
    n_events: int

    def hazard_ratios(self) -> pd.DataFrame:
        z = np.divide(
            self.coefficients,
            self.standard_errors,
            out=np.zeros_like(self.coefficients),
            where=self.standard_errors > 0,
        )
        return pd.DataFrame(
            {
                "covariate": self.columns,
                "coefficient": self.coefficients,
                "hazard_ratio": np.exp(self.coefficients),
                "lower": np.exp(self.coefficients - 1.96 * self.standard_errors),
                "upper": np.exp(self.coefficients + 1.96 * self.standard_errors),
                "z": z,
                "p_value": 2 * (1 - stats.norm.cdf(np.abs(z))),
            }
        ).sort_values("hazard_ratio", ascending=False)

    def risk_score(self, X: np.ndarray) -> np.ndarray:
        """Relative hazard for each row. Higher means the event comes sooner."""
        return np.exp(np.asarray(X, dtype=float) @ self.coefficients)


def fit_cox(
    X: np.ndarray, cohort: Cohort, columns: list[str], *, iterations: int = 40, tol: float = 1e-7
) -> Cox:
    """Maximise the Breslow partial likelihood by Newton-Raphson.

    Breslow's approximation handles ties — several customers having the event on the same
    day, which in daily retail data is constant rather than rare.
    """
    X = np.asarray(X, dtype=float)
    order = np.argsort(-cohort.duration)  # descending, so the risk set accumulates
    Xs, durations, events = X[order], cohort.duration[order], cohort.observed[order]

    beta = np.zeros(X.shape[1])
    log_likelihood = 0.0

    for _ in range(iterations):
        eta = np.clip(Xs @ beta, -30, 30)
        weights = np.exp(eta)

        # Risk sets: everyone with duration >= t. Descending order makes these cumulative.
        risk_sum = np.cumsum(weights)
        risk_x = np.cumsum(weights[:, None] * Xs, axis=0)
        risk_xx = np.cumsum(weights[:, None, None] * (Xs[:, :, None] * Xs[:, None, :]), axis=0)

        gradient = np.zeros_like(beta)
        hessian = np.zeros((len(beta), len(beta)))
        log_likelihood = 0.0

        for i in np.flatnonzero(events):
            # Ties: the risk set is everyone whose duration is >= this one.
            last = int(np.searchsorted(-durations, -durations[i], side="right")) - 1
            total = risk_sum[last]
            if total <= 0:
                continue
            mean = risk_x[last] / total
            log_likelihood += eta[i] - np.log(total)
            gradient += Xs[i] - mean
            hessian -= risk_xx[last] / total - np.outer(mean, mean)

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    try:
        covariance = np.linalg.inv(-hessian)
        errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        errors = np.full_like(beta, np.nan)

    return Cox(
        columns=columns,
        coefficients=beta,
        log_likelihood=float(log_likelihood),
        standard_errors=errors,
        n_observations=len(cohort),
        n_events=int(cohort.observed.sum()),
    )


def concordance(risk: np.ndarray, cohort: Cohort) -> float:
    """Harrell's C-index: of all comparable pairs, how many did the model order correctly?

    The survival analogue of AUC, and 0.5 is again chance. A pair is *comparable* only when
    the ordering is actually knowable — if one customer is censored at day 30 and another
    has the event at day 90, we know who came first; if both are censored we do not, and
    that pair is excluded rather than guessed at.
    """
    risk = np.asarray(risk, dtype=float)
    concordant = discordant = tied = 0

    for i in np.flatnonzero(cohort.observed):
        later = cohort.duration > cohort.duration[i]
        if not later.any():
            continue
        concordant += int(np.sum(risk[later] < risk[i]))
        discordant += int(np.sum(risk[later] > risk[i]))
        tied += int(np.sum(risk[later] == risk[i]))

    total = concordant + discordant + tied
    return float((concordant + 0.5 * tied) / total) if total else 0.5
