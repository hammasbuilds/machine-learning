"""Forecast backtesting, where the baseline is the whole point.

Most forecasting write-ups report MAPE and stop. Two problems with that, and the second is
worse than the first.

**MAPE is broken on retail data.** It divides by the actual value, so a day that sold 2
units and was forecast at 4 contributes 100% error, while a day that sold 2,000 and was
forecast at 2,400 contributes 20%. Days with small actuals dominate the average, and a
forecast can improve its MAPE by systematically under-predicting. On a series with any
zeroes it is undefined outright.

**A number with no baseline is not a result.** "MAPE 18%" means nothing without knowing
what the naive forecast scores. On a strongly seasonal series, *last week's value* is often
within a few points of an elaborate model, and any model that cannot beat it is a liability
with a maintenance cost.

So the measure here is **MASE** — mean absolute error divided by the error of a seasonal
naive forecast on the training data. It reads directly:

    MASE < 1   better than repeating last week
    MASE = 1   exactly as good as repeating last week
    MASE > 1   worse than doing nothing, with extra steps

And the evaluation is **rolling-origin**: fit on everything up to time t, forecast the next
h days, roll forward, repeat. One train/test split on a time series gives one number whose
error bars are invisible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


def seasonal_naive(history: pd.Series, horizon: int, *, season: int = 7) -> np.ndarray:
    """Repeat the value from one season ago. The baseline every model must beat."""
    values = history.to_numpy(dtype=float)
    if len(values) < season:
        return np.full(horizon, float(values[-1]) if len(values) else 0.0)
    return np.array([values[-season + (i % season)] for i in range(horizon)])


def drifting_mean(history: pd.Series, horizon: int, *, window: int = 28) -> np.ndarray:
    """Mean of the last `window` observations, held flat. The other baseline."""
    values = history.to_numpy(dtype=float)[-window:]
    return np.full(horizon, float(values.mean()) if len(values) else 0.0)


def seasonal_mean(
    history: pd.Series, horizon: int, *, season: int = 7, lookback: int = 8
) -> np.ndarray:
    """Average of the last `lookback` occurrences of each weekday.

    A genuine, very small model: it learns a weekly profile and nothing else. Included
    because it is the cheapest thing that could plausibly beat seasonal naive, and knowing
    whether it does tells you how much of the series is just the weekly shape.
    """
    values = history.to_numpy(dtype=float)
    out = np.zeros(horizon)
    for i in range(horizon):
        position = (len(values) + i) % season
        same_weekday = values[position::season][-lookback:]
        out[i] = (
            same_weekday.mean() if len(same_weekday) else (values.mean() if len(values) else 0.0)
        )
    return out


# --- scoring --------------------------------------------------------------


def mase(
    actual: np.ndarray, forecast: np.ndarray, history: np.ndarray, *, season: int = 7
) -> float:
    """Mean absolute scaled error. The denominator is the in-sample seasonal naive error.

    Scaling by the *training* naive error rather than the test one matters: a denominator
    computed on the test window would move whenever the test window got easier, and two
    models evaluated on different periods would stop being comparable.
    """
    history = np.asarray(history, dtype=float)
    if len(history) <= season:
        return float("nan")

    scale = np.mean(np.abs(history[season:] - history[:-season]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(forecast, float))) / scale)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error. Reported to be argued with, not relied on."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - forecast[nonzero]) / actual[nonzero])))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean signed error. Positive means the forecast runs high.

    Reported alongside every accuracy number because they answer different questions. A
    forecast can have excellent MASE and be persistently 8% low, which empties a warehouse
    slowly and shows up in no accuracy metric at all.
    """
    return float(np.mean(np.asarray(forecast, float) - np.asarray(actual, float)))


# --- the evaluation loop --------------------------------------------------


def rolling_origin(
    series: pd.Series, *, horizon: int, step: int, min_train: int, embargo: int = 0
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """Expanding-window splits: fit on the past, score the next `horizon`, roll forward.

    `embargo` drops the last few training points. It is unnecessary for a plain forecast —
    the target is genuinely in the future — but it becomes essential the moment a feature
    is built from a trailing window, because a 7-day rolling mean computed at the last
    training point already contains days that belong to the test set.
    """
    values = series.dropna()
    n = len(values)
    origin = min_train

    while origin + horizon <= n:
        train = values.iloc[: max(origin - embargo, 1)]
        test = values.iloc[origin : origin + horizon]
        yield train, test
        origin += step


@dataclass(frozen=True)
class Score:
    name: str
    mase: float
    mape: float
    bias: float
    folds: int

    @property
    def beats_naive(self) -> bool:
        return self.mase < 1.0

    def summary(self) -> dict:
        return {
            "model": self.name,
            "mase": round(self.mase, 4),
            "mape": round(self.mape, 4),
            "bias": round(self.bias, 3),
            "folds": self.folds,
            "beats_seasonal_naive": self.beats_naive,
        }


def evaluate(
    series: pd.Series,
    models: dict[str, Callable[[pd.Series, int], np.ndarray]],
    *,
    horizon: int = 7,
    step: int = 7,
    min_train: int = 120,
    season: int = 7,
    embargo: int = 0,
) -> list[Score]:
    """Score every model over the same rolling folds."""
    collected: dict[str, list[tuple[float, float, float]]] = {name: [] for name in models}

    for train, test in rolling_origin(
        series, horizon=horizon, step=step, min_train=min_train, embargo=embargo
    ):
        actual = test.to_numpy(dtype=float)
        history = train.to_numpy(dtype=float)
        for name, model in models.items():
            forecast = model(train, len(actual))
            collected[name].append(
                (
                    mase(actual, forecast, history, season=season),
                    mape(actual, forecast),
                    bias(actual, forecast),
                )
            )

    scores = []
    for name, rows in collected.items():
        arr = np.array(rows, dtype=float)
        scores.append(
            Score(
                name=name,
                mase=float(np.nanmean(arr[:, 0])),
                mape=float(np.nanmean(arr[:, 1])),
                bias=float(np.nanmean(arr[:, 2])),
                folds=len(rows),
            )
        )
    return sorted(scores, key=lambda s: s.mase)
