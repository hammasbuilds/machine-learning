"""Classical time series, and the one decision that dominates every other.

Before ARIMA, before Holt-Winters, before any model at all, there is a question that decides
what all of them will say:

    **is this series stationary, and if not, what do you do about it?**

Get it wrong in one direction and you fit a model to a trend that will not repeat. Get it
wrong in the other and you difference away the signal you were trying to find. And the two
standard tests for it - ADF and KPSS - **routinely disagree**, because they test opposite
null hypotheses and neither is powerful at the sample sizes people actually have.

The consequence nobody demonstrates, because it makes the rest of the analysis look fragile:

**Spurious regression.** Two completely independent random walks, regressed on each other in
levels, produce a large R^2 and a t-statistic that clears any significance threshold you
like. Granger & Newbold (1974). The relationship is not weak, not marginal - it is *absent*,
and the regression reports it as overwhelming. Differencing destroys it instantly.

This file implements the forecasters against a single interface so the comparison is
like-for-like, and the point of the comparison is that on a unit-root series **they all
collapse toward the same forecast**: the last value.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class StationarityVerdict:
    """What the two tests said, and whether they agreed.

    The nulls are opposite, which is the whole point and is almost never stated:

        ADF   null = "has a unit root"  -> a small p-value means *stationary*
        KPSS  null = "is stationary"    -> a small p-value means *not stationary*

    So the confirmatory result is both tests rejecting, in opposite directions. Anything else
    is a series neither test can classify, and that case is common.
    """

    series: str
    adf_p: float
    kpss_p: float
    adf_says_stationary: bool
    kpss_says_stationary: bool

    @property
    def agree(self) -> bool:
        return self.adf_says_stationary == self.kpss_says_stationary

    @property
    def verdict(self) -> str:
        if not self.agree:
            return "undecided"
        return "stationary" if self.adf_says_stationary else "non-stationary"


def stationarity_verdict(
    values: np.ndarray, name: str, *, alpha: float = 0.05
) -> StationarityVerdict:
    """Run both tests and report the disagreement rather than picking a favourite."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(values, autolag="AIC")[1])

    # KPSS clips its p-value when the statistic falls outside the interpolation table. That
    # clipping is the honest answer - the test cannot resolve further - so it is kept.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_p = float(kpss(values, regression="c", nlags="auto")[1])

    return StationarityVerdict(
        series=name,
        adf_p=adf_p,
        kpss_p=kpss_p,
        adf_says_stationary=adf_p < alpha,
        kpss_says_stationary=kpss_p > alpha,
    )


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, its t-statistic and R^2. Written out because the t-statistic is the exhibit."""
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = len(x) - 2
    sigma2 = float(residual @ residual) / dof
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    t = float(beta[1] / np.sqrt(covariance[1, 1]))
    total = float(((y - y.mean()) ** 2).sum())
    return {
        "slope": float(beta[1]),
        "t_statistic": t,
        "abs_t": abs(t),
        "r_squared": 1 - float(residual @ residual) / total if total > 0 else 0.0,
        "significant": abs(t) > 1.96,
    }


def spurious_regression(n: int, *, trials: int, seed: int = 0) -> pd.DataFrame:
    """Regress independent random walks on each other, in levels and in differences.

    Two series with no connection whatsoever. In levels the t-statistics are enormous and
    R^2 is often above 0.5; in differences both collapse to what chance predicts. Nothing
    about the data changed - only whether the regression was run on a stationary quantity.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for trial in range(trials):
        x = np.cumsum(rng.normal(size=n))
        y = np.cumsum(rng.normal(size=n))
        rows.append({"trial": trial, "basis": "levels", **_ols(x, y)})
        rows.append({"trial": trial, "basis": "differences", **_ols(np.diff(x), np.diff(y))})

    return pd.DataFrame(rows)


# --- forecasters ----------------------------------------------------------


class RandomWalk:
    """Tomorrow equals today. The benchmark every financial forecast has to beat."""

    name = "Random walk (last value)"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(history[-1], horizon)


class Drift:
    """A random walk that continues the average slope so far.

    The minimal way to use a trend, and the honest version of "the market goes up": it
    extrapolates the realised drift rather than a fitted one.
    """

    name = "Drift"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if len(history) < 2:
            return np.repeat(history[-1], horizon)
        slope = (history[-1] - history[0]) / (len(history) - 1)
        return history[-1] + slope * np.arange(1, horizon + 1)


class Mean:
    """The average of a trailing window. Included because on a *stationary* series it wins."""

    def __init__(self, window: int = 24) -> None:
        self.window = window
        self.name = f"Trailing mean ({window})"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(float(np.mean(history[-self.window :])), horizon)


class Arima:
    """Box-Jenkins with a fixed order, refitted on every fold.

    The order is fixed rather than searched, deliberately. Searching (p,d,q) inside a rolling
    backtest and reporting the best fold-by-fold result is the time-series version of the
    overfitting in project 01 - the search has to be inside the fold or the score is a
    fiction, and once it is inside the fold the chosen order changes every time.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = order
        self.name = f"ARIMA{order}"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(history, order=self.order).fit()
            return np.asarray(fitted.forecast(horizon), dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            # Non-convergence on a fold is a result, not something to hide: fall back to the
            # benchmark and let the score show the model had nothing to add.
            return np.repeat(history[-1], horizon)


class Holt:
    """Exponential smoothing with a damped trend.

    Damped rather than additive because an undamped trend extrapolated over a long horizon is
    the single most reliable way to produce an absurd forecast.
    """

    name = "Holt (damped trend)"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ExponentialSmoothing(history, trend="add", damped_trend=True).fit()
            return np.asarray(fitted.forecast(horizon), dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            return np.repeat(history[-1], horizon)


# --- evaluation -----------------------------------------------------------


def mase(actual: np.ndarray, predicted: np.ndarray, history: np.ndarray) -> float:
    """Error scaled by the in-sample one-step naive error.

    1.0 means "no better than predicting the last value". The scaling makes it comparable
    across series and immune to the level problem that makes MAPE useless near zero.
    """
    scale = float(np.mean(np.abs(np.diff(history))))
    if scale <= 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))) / scale)


def rolling_backtest(
    series: np.ndarray, models: list, *, horizon: int, folds: int, minimum: int
) -> pd.DataFrame:
    """Expanding-window origin evaluation: fit on the past, score on the next `horizon`.

    Expanding rather than sliding, because on 150 years of monthly data the question is not
    "does a short window help" but "does more history help at all" - and on a unit-root
    series the answer is no, which is worth showing rather than assuming away.
    """
    series = np.asarray(series, dtype=float)
    origins = np.linspace(minimum, len(series) - horizon, folds, dtype=int)
    rows = []

    for fold, origin in enumerate(origins):
        history = series[:origin]
        actual = series[origin : origin + horizon]
        for model in models:
            predicted = model.forecast(history, horizon)
            rows.append(
                {
                    "fold": fold,
                    "origin": int(origin),
                    "model": model.name,
                    "mase": mase(actual, predicted, history),
                    "mae": float(np.mean(np.abs(actual - predicted))),
                    "bias": float(np.mean(predicted - actual)),
                }
            )

    return pd.DataFrame(rows)
