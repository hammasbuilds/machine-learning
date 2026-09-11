"""Overlapping windows: how to get a t-statistic of 8 from 70 independent observations.

The classic finding in long-run finance is that valuation predicts returns: when the
cyclically-adjusted price/earnings ratio is high, the next ten years are poor. The
regression has an R-squared around 0.3 and a t-statistic that looks overwhelming.

**The t-statistic is wrong, and the reason is arithmetic rather than economics.**

Monthly data from 1881 to 2026 gives about 1,600 observations of "CAPE today, return over
the next ten years". But the ten-year window starting in January 1990 shares 119 of its 120
months with the window starting in February 1990. Those are not two observations. They are
one observation reported twice, with a one-month shuffle.

In 145 years there are about **fourteen** genuinely non-overlapping ten-year periods.
Fourteen. Every standard error computed as though there were 1,600 is too small by roughly
sqrt(120), and every t-statistic is too large by the same factor.

This is the same error as [the CLCuV pseudo-replication](../../README.md) and the same
error as counting one wire story republished by twelve outlets as twelve sources: **the
unit of observation is not the row.**

Three corrections, in increasing order of honesty:

  naive        ordinary least squares standard errors. Wrong, and the number usually quoted.
  Newey-West   an autocorrelation-consistent estimator. Better, still generous.
  disjoint     throw away every overlapping window and use the handful that remain. Brutal,
               and the only one whose assumptions are actually met.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def forward_return(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Annualised return over the next `horizon` periods. NaN where the window runs out."""
    prices = np.asarray(prices, dtype=float)
    out = np.full(len(prices), np.nan)
    years = horizon / 12.0

    ahead = prices[horizon:]
    here = prices[: len(prices) - horizon]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[: len(prices) - horizon] = (ahead / here) ** (1.0 / years) - 1.0
    return out


def effective_sample_size(n_observations: int, horizon: int) -> float:
    """How many independent windows a series of this length actually contains.

    The whole argument in one line: consecutive overlapping windows share all but one of
    their periods, so `n` rows of a `horizon`-period window carry about `n / horizon`
    independent pieces of information.
    """
    return n_observations / horizon if horizon > 0 else float(n_observations)


def newey_west_se(x: np.ndarray, y: np.ndarray, *, lags: int) -> tuple[float, float]:
    """Slope and a heteroskedasticity- and autocorrelation-consistent standard error.

    Newey-West widens the standard error to account for residuals that are correlated with
    their own recent past — which overlapping windows guarantee, because neighbouring
    residuals are built from mostly the same data. The usual choice of `lags` is the
    horizon minus one, since that is exactly how far the overlap reaches.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    design = np.c_[np.ones(n), x]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients

    # Meat of the sandwich: contemporaneous term plus Bartlett-weighted autocovariances.
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)

    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ meat @ bread
    return float(coefficients[1]), float(math.sqrt(max(covariance[1, 1], 0.0)))


@dataclass(frozen=True)
class Regression:
    """One estimate of the same relationship, under one set of assumptions."""

    method: str
    slope: float
    standard_error: float
    n_used: int
    n_independent: float

    @property
    def t_statistic(self) -> float:
        return self.slope / self.standard_error if self.standard_error else 0.0

    @property
    def p_value(self) -> float:
        degrees = max(self.n_independent - 2, 1)
        return float(2 * (1 - stats.t.cdf(abs(self.t_statistic), df=degrees)))

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> dict:
        return {
            "method": self.method,
            "slope": round(self.slope, 5),
            "standard_error": round(self.standard_error, 5),
            "t": round(self.t_statistic, 2),
            "p": round(self.p_value, 4),
            "rows_used": self.n_used,
            "independent_observations": round(self.n_independent, 1),
            "significant_at_5pct": self.significant,
        }


def ordinary_least_squares(
    x: np.ndarray, y: np.ndarray, *, method: str, n_independent: float | None = None
) -> Regression:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    design = np.c_[np.ones(n), x]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients

    sigma_squared = float(residuals @ residuals) / max(n - 2, 1)
    covariance = sigma_squared * np.linalg.inv(design.T @ design)

    return Regression(
        method=method,
        slope=float(coefficients[1]),
        standard_error=float(math.sqrt(max(covariance[1, 1], 0.0))),
        n_used=n,
        n_independent=float(n_independent if n_independent is not None else n),
    )


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.c_[np.ones(len(x)), x]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    total = y - y.mean()
    return float(1.0 - (residuals @ residuals) / (total @ total))


def disjoint_indices(n: int, horizon: int, *, offset: int = 0) -> np.ndarray:
    """Every `horizon`-th row, so no two selected windows share a single period."""
    return np.arange(offset, n, horizon)
