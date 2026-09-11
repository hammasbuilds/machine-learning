"""The Deflated Sharpe Ratio: how much of a backtest result is skill and how much is search.

The problem in one sentence: **if you try a thousand strategies and keep the best one,
its Sharpe ratio measures how many things you tried, not how good the strategy is.**

This is not a subtle effect. Draw a thousand strategies that have *no edge whatsoever* —
pure noise, expected return exactly zero — and the best of them will show an annualised
Sharpe around 2.5 on ten years of daily data. Publish that one and it looks like a career.

Bailey and Lopez de Prado's correction (2014) asks a sharper question than "is this Sharpe
significantly above zero?". It asks:

    given that I ran N trials on T observations, and given how skewed and fat-tailed the
    returns are, what is the probability this Sharpe exceeds what the best of N pure-noise
    strategies would have produced anyway?

Three inputs the naive t-test ignores, all of which matter:

  N   the number of trials. The one everybody omits, because admitting it is embarrassing.
  gamma3  skewness. Negative skew - small steady gains, occasional disasters - inflates
          Sharpe, which is why selling options backtests beautifully until it does not.
  gamma4  kurtosis. Fat tails mean the Sharpe estimate itself is noisier than it looks.

Reference: Bailey, D. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of
Portfolio Management 40(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_per_period(returns: np.ndarray) -> float:
    """Sharpe ratio in the units the data arrives in. Risk-free rate taken as zero.

    **Every formula below operates on this, not on the annualised figure**, and the
    distinction is not pedantry - it was a live bug in this file. Feeding an annualised
    Sharpe of 0.62 into the track-record formula returns "15 days"; feeding the daily
    Sharpe of 0.039 that it actually represents returns roughly seven years. The second
    answer is the true one. The first is wrong by the annualisation factor squared, and
    it errs in the direction that makes an unproven strategy look proven.

    Zero for the risk-free rate is a simplification, and defensible: every strategy in
    the comparison is measured the same way, so a constant cannot change which one wins.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(returns.mean() / sd)


def annualise(per_period_sharpe: float, *, periods_per_year: int = 252) -> float:
    """Scale a per-period Sharpe for human reading. Display only - never fed back in."""
    return per_period_sharpe * math.sqrt(periods_per_year)


def sharpe(returns: np.ndarray, *, periods_per_year: int = 252) -> float:
    """Annualised Sharpe, for reporting. The search ranks on this; the maths does not."""
    return annualise(sharpe_per_period(returns), periods_per_year=periods_per_year)


def expected_max_sharpe(n_trials: int, variance_of_sharpe: float) -> float:
    """The Sharpe you should expect from the *best* of N strategies that have no edge.

    This is the number that makes the whole method work, and it is the number nobody
    computes. It follows from the expected maximum of N draws from a normal distribution,
    which grows like sqrt(2 ln N) — slowly, but relentlessly. Ten trials, a thousand
    trials and a million trials are meaningfully different bars to clear.
    """
    if n_trials < 2 or variance_of_sharpe <= 0:
        return 0.0

    n = float(n_trials)
    # Expected maximum of N standard normals, to the usual second-order accuracy.
    expected_max_z = (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n) + (
        EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n * math.e))
    )
    return float(math.sqrt(variance_of_sharpe) * expected_max_z)


def sharpe_standard_error(
    observed_sharpe: float, n_observations: int, skew: float, kurtosis: float
) -> float:
    """Standard error of a Sharpe estimate, adjusted for skew and fat tails.

    Under normality this reduces to roughly sqrt((1 + SR^2/2)/T). Real return series are
    not normal, and the adjustment is not cosmetic: strongly negative skew can shrink the
    apparent standard error, making an overfit strategy look *more* significant than it is.
    """
    if n_observations < 2:
        return float("inf")
    variance = (1 - skew * observed_sharpe + (kurtosis - 1) / 4.0 * observed_sharpe**2) / (
        n_observations - 1
    )
    return float(math.sqrt(max(variance, 1e-12)))


@dataclass(frozen=True)
class Verdict:
    """What a backtest is worth once the search behind it is accounted for."""

    observed_sharpe: float  # annualised, for reading
    benchmark_sharpe: float  # annualised: what the best of N noise strategies reaches
    probability_real: float  # the Deflated Sharpe Ratio itself
    per_period_sharpe: float  # the units the maths actually ran in
    n_trials: int
    n_observations: int
    skew: float
    kurtosis: float

    @property
    def survives(self) -> bool:
        """The conventional bar. 95% is a convention, not a law, and is stated as such."""
        return self.probability_real >= 0.95

    @property
    def inflation(self) -> float:
        """How much of the observed Sharpe the search alone explains."""
        if self.observed_sharpe <= 0:
            return 0.0
        return round(self.benchmark_sharpe / self.observed_sharpe, 3)

    def summary(self) -> dict:
        return {
            "observed_sharpe": round(self.observed_sharpe, 3),
            "noise_benchmark": round(self.benchmark_sharpe, 3),
            "share_explained_by_search": self.inflation,
            "deflated_sharpe_ratio": round(self.probability_real, 4),
            "verdict": "survives" if self.survives else "not distinguishable from search",
            "trials": self.n_trials,
            "observations": self.n_observations,
        }

    def sentence(self) -> str:
        if self.survives:
            return (
                f"Sharpe {self.observed_sharpe:.2f} over {self.n_observations:,} days from "
                f"{self.n_trials:,} trials: survives deflation at {self.probability_real:.1%}."
            )
        return (
            f"Sharpe {self.observed_sharpe:.2f} looks strong, but the best of "
            f"{self.n_trials:,} no-edge strategies would reach {self.benchmark_sharpe:.2f} "
            f"on this data. Probability the edge is real: {self.probability_real:.1%}."
        )


def deflated_sharpe(
    best_returns: np.ndarray,
    *,
    n_trials: int,
    all_trial_sharpes: np.ndarray | None = None,
    periods_per_year: int = 252,
) -> Verdict:
    """Judge the winning strategy, given how many strategies were tried to find it.

    `all_trial_sharpes` is the spread of Sharpe ratios across every strategy tested. When
    it is available it is used directly, and that is the honest way: the variance of the
    search is measured rather than assumed. When it is not, the variance is approximated
    from the winner alone, which is weaker and is noted in the docstring rather than
    hidden.
    """
    returns = np.asarray(best_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n_observations = len(returns)
    if n_observations < 3:
        raise ValueError("need at least three return observations")

    # Everything here is per-period. Annualisation happens once, at the end, for display.
    observed = sharpe_per_period(returns)
    skew = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns, fisher=False))  # normal = 3

    if all_trial_sharpes is not None and len(all_trial_sharpes) > 1:
        trials = np.asarray(all_trial_sharpes, dtype=float) / math.sqrt(periods_per_year)
        variance_of_sharpe = float(np.var(trials, ddof=1))
    else:
        variance_of_sharpe = sharpe_standard_error(observed, n_observations, skew, kurtosis) ** 2

    benchmark = expected_max_sharpe(n_trials, variance_of_sharpe)
    standard_error = sharpe_standard_error(observed, n_observations, skew, kurtosis)

    z = (observed - benchmark) / standard_error if standard_error > 0 else 0.0
    probability = float(stats.norm.cdf(z))

    return Verdict(
        observed_sharpe=annualise(observed, periods_per_year=periods_per_year),
        benchmark_sharpe=annualise(benchmark, periods_per_year=periods_per_year),
        probability_real=probability,
        per_period_sharpe=observed,
        n_trials=n_trials,
        n_observations=n_observations,
        skew=skew,
        kurtosis=kurtosis,
    )


def minimum_track_record_length(
    per_period_sharpe: float,
    *,
    target_per_period_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """How many observations are needed before a Sharpe this size means anything.

    Takes the **per-period** Sharpe and returns a count of periods. Passing an annualised
    figure divides the answer by the annualisation factor squared, and the result still
    looks plausible, which is exactly what makes it dangerous. The parameter is named for
    its units so the mistake is harder to repeat.

    The companion question to deflation, and the more useful one when someone shows you a
    six-month track record: an annualised Sharpe of 1.0 needs roughly three years of daily
    observations before it can be told apart from zero at 95% confidence - and that is
    before any correction for how many strategies were tried.
    """
    gap = per_period_sharpe - target_per_period_sharpe
    if gap <= 0:
        return float("inf")
    z = stats.norm.ppf(confidence)
    variance = 1 - skew * per_period_sharpe + (kurtosis - 1) / 4.0 * per_period_sharpe**2
    return float(1 + variance * (z / gap) ** 2)
