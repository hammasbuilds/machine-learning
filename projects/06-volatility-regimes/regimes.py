"""Volatility regimes, and the difference between knowing one and having known one.

Every volatility chart ever published has the calm periods and the crises marked on it,
and they look obvious. They are obvious — **afterwards.** The label "this was a
high-volatility regime, September 2008 to June 2009" is assigned by someone who can see
2009 from 2010.

The question that decides whether regime detection is analysis or decoration is narrower:

    **on 15 September 2008, using only data up to 15 September 2008, what regime were you
    in?**

This module labels the same series twice and compares them:

  hindsight   thresholds fitted on the whole history, applied to the whole history. This is
              what almost every regime chart shows, and it cannot be traded.
  real time   at each day, thresholds fitted only on what came before, applied to today.
              This is the only version that could have been acted on.

They disagree, and **the disagreement is not spread evenly.** It clusters at exactly the
turning points — the days when the regime is changing and the label would have been worth
something. In the calm middle of a regime both methods agree and neither is useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CALM, NORMAL, STRESSED = 0, 1, 2
REGIME_NAMES = ("calm", "normal", "stressed")


def label_by_quantile(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Three regimes from two cut points."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), NORMAL, dtype=int)
    out[values <= low] = CALM
    out[values >= high] = STRESSED
    return out


def hindsight_labels(series: pd.Series, *, lower: float = 0.33, upper: float = 0.80) -> np.ndarray:
    """Regimes from thresholds fitted on the entire history.

    Uses the future. Included because it is what gets published, and because the honest
    version needs something to be compared against.
    """
    values = series.to_numpy(dtype=float)
    return label_by_quantile(values, np.quantile(values, lower), np.quantile(values, upper))


def realtime_labels(
    series: pd.Series, *, lower: float = 0.33, upper: float = 0.80, warmup: int = 500
) -> np.ndarray:
    """Regimes from thresholds fitted only on data available at the time.

    Expanding window: on day t the quantiles come from days 0..t-1, and today's value is
    classified against them. The first `warmup` days have too little history to classify
    and are returned as -1 rather than guessed at.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), -1, dtype=int)

    for t in range(warmup, len(values)):
        past = values[:t]
        low, high = np.quantile(past, lower), np.quantile(past, upper)
        out[t] = label_by_quantile(values[t : t + 1], low, high)[0]

    return out


def transition_matrix(labels: np.ndarray, *, n_states: int = 3) -> np.ndarray:
    """P(tomorrow's regime | today's regime). Rows sum to one.

    The diagonal is the story: volatility regimes are extremely persistent, which is what
    makes them look predictable and what makes "predicting" them nearly worthless. A model
    that says "tomorrow will be like today" is right about 97% of the time and has told you
    nothing you did not know.
    """
    labels = np.asarray(labels)
    valid = labels >= 0
    pairs = np.c_[labels[:-1], labels[1:]][valid[:-1] & valid[1:]]

    counts = np.zeros((n_states, n_states))
    for a, b in pairs:
        counts[int(a), int(b)] += 1

    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def persistence(labels: np.ndarray) -> float:
    """Share of days on which the regime is the same as yesterday's."""
    labels = np.asarray(labels)
    valid = (labels[:-1] >= 0) & (labels[1:] >= 0)
    return float(np.mean(labels[:-1][valid] == labels[1:][valid]))


def runs(labels: np.ndarray) -> pd.DataFrame:
    """Contiguous stretches of one regime: where each began, how long it lasted."""
    labels = np.asarray(labels)
    rows, start = [], None

    for i in range(len(labels)):
        if labels[i] < 0:
            continue
        if start is None or labels[i] != labels[start]:
            if start is not None:
                rows.append({"regime": int(labels[start]), "start": start, "length": i - start})
            start = i

    if start is not None:
        rows.append({"regime": int(labels[start]), "start": start, "length": len(labels) - start})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Disagreement:
    """Where the two labellings differ, and whether the difference is concentrated."""

    days_compared: int
    days_disagreeing: int
    at_turning_points: int
    turning_point_days: int

    @property
    def rate(self) -> float:
        return self.days_disagreeing / self.days_compared if self.days_compared else 0.0

    @property
    def rate_at_turning_points(self) -> float:
        return self.at_turning_points / self.turning_point_days if self.turning_point_days else 0.0

    @property
    def concentration(self) -> float:
        """How many times more likely a disagreement is near a turning point."""
        return round(self.rate_at_turning_points / self.rate, 2) if self.rate else 0.0

    def summary(self) -> dict:
        return {
            "days_compared": self.days_compared,
            "disagreement_rate": round(self.rate, 4),
            "disagreement_rate_near_turning_points": round(self.rate_at_turning_points, 4),
            "concentration": self.concentration,
        }


def compare(hindsight: np.ndarray, realtime: np.ndarray, *, window: int = 10) -> Disagreement:
    """Compare the two labellings, and check where the differences land.

    A turning point is any day within `window` days of a regime change *in the hindsight
    labelling* — the one that, with full knowledge, marks where the regime genuinely turned.
    """
    hindsight = np.asarray(hindsight)
    realtime = np.asarray(realtime)
    comparable = realtime >= 0

    changed = np.r_[False, hindsight[1:] != hindsight[:-1]]
    near_change = (
        pd.Series(changed).rolling(2 * window + 1, center=True, min_periods=1).max().to_numpy() > 0
    )

    disagree = comparable & (hindsight != realtime)
    return Disagreement(
        days_compared=int(comparable.sum()),
        days_disagreeing=int(disagree.sum()),
        at_turning_points=int((disagree & near_change).sum()),
        turning_point_days=int((comparable & near_change).sum()),
    )


def realised_volatility(
    returns: pd.Series, *, window: int = 21, periods_per_year: int = 252
) -> pd.Series:
    """Trailing annualised standard deviation. What actually happened, as opposed to VIX,
    which is what the option market expected to happen."""
    return returns.rolling(window).std() * np.sqrt(periods_per_year)
