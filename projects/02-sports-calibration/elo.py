"""Elo ratings and the three ways a football forecast can be judged.

The question this project exists to answer is not "can a model predict football matches".
It is **"what does it mean for a forecast to be good?"** — because the obvious answer,
accuracy, is close to useless here.

A Premier League season is roughly 45% home wins, 25% draws, 30% away wins. A model that
always says "home win" scores 45% accuracy while being completely uninformative. A model
that reaches 53% looks much better and may still be worse than the bookmaker, because
accuracy throws away the only thing a probability carries: **how confident it was.**

Three measures, and they disagree on purpose:

  accuracy   was the most likely outcome the one that happened? Discards confidence.
  Brier      mean squared error of the probability vector. Rewards honest uncertainty.
  log loss   punishes confident mistakes savagely. One 99% call that fails can dominate.

The bookmaker's closing odds are the benchmark throughout. They are not a model — they
are the aggregate of everyone who was willing to put money behind an opinion, which is a
very hard thing to beat and an embarrassing thing to ignore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOME = 0
DRAW = 1
AWAY = 2
OUTCOMES = ("H", "D", "A")


@dataclass
class Elo:
    """Standard Elo, with the two adjustments football needs.

    **Home advantage** is added to the home side's rating before the expectation is taken.
    In the Premier League it is worth roughly 60-70 Elo points, and a model without it is
    systematically wrong on every single fixture in the same direction.

    **Margin of victory** scales the update. A 5-0 win is more evidence than a 1-0 win, and
    treating them identically throws away information that is right there in the result.
    """

    k: float = 20.0
    home_advantage: float = 65.0
    start: float = 1500.0
    mov_factor: float = 0.0  # 0 disables margin-of-victory scaling
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.start)

    def expected_home_score(self, home: str, away: str) -> float:
        """Expected points for the home side on a 0-1 scale, draws counting as 0.5."""
        gap = (self.rating(home) + self.home_advantage) - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-gap / 400.0))

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        expected = self.expected_home_score(home, away)
        actual = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)

        k = self.k
        if self.mov_factor:
            margin = abs(home_goals - away_goals)
            # Diminishing returns: the second goal says more than the fifth.
            k *= 1.0 + self.mov_factor * np.log1p(margin)

        delta = k * (actual - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def regress_to_mean(self, weight: float = 0.25) -> None:
        """Pull ratings toward 1500 between seasons.

        Squads change over a summer. Carrying a rating forward untouched assumes the team
        that finished in May is the team that starts in August, which is why a promoted
        side inherits a relegated side's slot but not its players.
        """
        for team, value in self.ratings.items():
            self.ratings[team] = value + weight * (self.start - value)


def three_way_probabilities(expected_home: float, draw_rate: float = 0.25) -> np.ndarray:
    """Turn one Elo expectation into probabilities for home / draw / away.

    Elo gives a single number on a 0-1 scale where a draw counts half. Football has three
    outcomes, so the scalar has to be split. The simple and defensible split: fix the draw
    probability at its base rate, then divide what remains in proportion to the expectation.

    It is crude. That is the point of comparing it to a fitted model and to the bookmaker —
    the reader gets to see how much the crudeness costs, rather than being told it is fine.
    """
    remaining = 1.0 - draw_rate
    # Re-centre the expectation onto the home/away split, removing the draw's half-point.
    home_share = np.clip((expected_home - draw_rate / 2.0) / remaining, 0.01, 0.99)
    return np.array([remaining * home_share, draw_rate, remaining * (1.0 - home_share)])


# --- the bookmaker --------------------------------------------------------


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to probabilities, removing the overround.

    **The trap this exists for.** Raw `1/odds` across three outcomes sums to about 1.05,
    not 1.00. The excess is the bookmaker's margin — the vig, the overround — and it is how
    they make money. Comparing a model's probabilities (which sum to 1) against raw
    implied odds (which sum to 1.05) hands the model a free 5% and makes every calibration
    plot wrong in the bookmaker's disfavour.

    Normalising by the sum is the simple removal, and it assumes the margin is spread
    proportionally. It is not, quite — favourites carry less of it than longshots, the
    favourite-longshot bias — but proportional is the standard treatment and the
    alternative needs assumptions this project does not want to smuggle in.
    """
    odds = np.asarray(odds, dtype=float)
    raw = 1.0 / odds
    return raw / raw.sum(axis=-1, keepdims=True)


def overround(odds: np.ndarray) -> np.ndarray:
    """The bookmaker's margin, per match. Typically 1.02-1.08."""
    return (1.0 / np.asarray(odds, dtype=float)).sum(axis=-1)


# --- scoring --------------------------------------------------------------


def multiclass_brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error across the whole probability vector. Lower is better.

    Always guessing the base rate scores about 0.62 on three-way football. Perfect
    foresight scores 0. This is the measure that separates forecasts accuracy cannot.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray, *, floor: float = 1e-15) -> float:
    """Negative log likelihood. Punishes confident errors far harder than Brier does."""
    probabilities = np.asarray(probabilities, dtype=float)
    picked = probabilities[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log(np.clip(picked, floor, 1.0))))


def accuracy(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Share of matches where the highest-probability outcome happened.

    Reported because everyone asks for it, and placed last because on this problem it is
    the least informative of the three.
    """
    return float(np.mean(np.argmax(probabilities, axis=1) == outcomes))


def reliability(probabilities: np.ndarray, outcomes: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency, pooled over all three outcomes.

    Every (match, outcome) pair contributes one point: the probability assigned, and
    whether it happened. Perfect calibration is the diagonal. The count per bin is kept
    because a bin holding nine samples deserves less of the reader's attention than one
    holding nine hundred, and calibration plots that hide the counts mislead.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    flat = probabilities.reshape(-1)
    happened = np.zeros_like(probabilities)
    happened[np.arange(len(outcomes)), outcomes] = 1.0
    flat_truth = happened.reshape(-1)

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(flat, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "predicted": float(flat[mask].mean()) if n else np.nan,
                "observed": float(flat_truth[mask].mean()) if n else np.nan,
                "count": n,
            }
        )
    return pd.DataFrame(rows)
