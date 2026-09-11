"""Generating the strategy search that produces the illusion.

Nothing here is a trading recommendation, and the strategies are deliberately ordinary —
moving-average crossovers, momentum lookbacks, mean-reversion bands. That is the point.
The claim under test is not "these rules work". It is **"search over enough ordinary rules
and one of them will look like it works, on any data at all."**

Two searches run side by side, which is what makes the result arguable rather than merely
stated:

  real   the rules applied to actual SPY history
  null   the identical rules applied to shuffled returns, where by construction no
         timing edge can exist

Whatever the null search produces is pure selection effect. If the real search produces
about the same thing, the real search found nothing either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    """One parameterisation. `kind` selects the family, the rest are its knobs."""

    kind: str
    fast: int
    slow: int

    @property
    def name(self) -> str:
        return f"{self.kind}({self.fast},{self.slow})"


def rule_grid(*, fast_values: range | None = None, slow_values: range | None = None) -> list[Rule]:
    """The parameter grid. Its size *is* the number of trials, and must be reported.

    A grid of this shape is what an afternoon of "let me just try a few settings" actually
    amounts to. Counting it honestly is the first half of the correction.
    """
    fast_values = fast_values or range(2, 32, 2)
    slow_values = slow_values or range(10, 210, 10)

    rules: list[Rule] = []
    for kind in ("crossover", "momentum", "reversion"):
        for fast in fast_values:
            for slow in slow_values:
                if fast < slow:
                    rules.append(Rule(kind, fast, slow))
    return rules


def positions(prices: pd.Series, rule: Rule) -> pd.Series:
    """Long/flat position implied by a rule, shifted so no future information is used.

    The `.shift(1)` is the whole difference between a backtest and a fantasy: a signal
    computed from today's close cannot be traded at today's close. Removing that shift
    roughly doubles every Sharpe in this file, which is the most common way backtests lie
    and the reason it is written once, here, rather than in each strategy.
    """
    if rule.kind == "crossover":
        fast = prices.rolling(rule.fast).mean()
        slow = prices.rolling(rule.slow).mean()
        signal = (fast > slow).astype(float)

    elif rule.kind == "momentum":
        past = prices.pct_change(rule.slow)
        recent = prices.pct_change(rule.fast)
        signal = ((past > 0) & (recent > 0)).astype(float)

    elif rule.kind == "reversion":
        mean = prices.rolling(rule.slow).mean()
        sd = prices.rolling(rule.slow).std()
        z = (prices - mean) / sd
        signal = (z < -1.0).astype(float)

    else:
        raise ValueError(f"unknown rule kind: {rule.kind}")

    return signal.shift(1).fillna(0.0)


def strategy_returns(prices: pd.Series, rule: Rule, *, cost_per_turn: float = 0.0005) -> pd.Series:
    """Returns net of a transaction cost charged on every position change.

    Five basis points per turn is modest for a liquid ETF and brutal for a rule that
    trades daily — which is the honest shape of the problem. A costless backtest rewards
    exactly the strategies that cannot survive contact with a broker.
    """
    market = prices.pct_change().fillna(0.0)
    held = positions(prices, rule)
    turnover = held.diff().abs().fillna(0.0)
    return held * market - turnover * cost_per_turn


def run_search(
    prices: pd.Series, rules: list[Rule], *, cost_per_turn: float = 0.0005
) -> pd.DataFrame:
    """Backtest every rule. Returns one row per trial, sorted best first.

    This is the object the whole project argues about: a table of results from which
    somebody is about to pick the top row and call it a strategy.
    """
    from deflated import sharpe  # local import keeps this module dependency-light

    rows = []
    for rule in rules:
        returns = strategy_returns(prices, rule, cost_per_turn=cost_per_turn)
        active = float((returns != 0).mean())
        rows.append(
            {
                "rule": rule.name,
                "kind": rule.kind,
                "fast": rule.fast,
                "slow": rule.slow,
                "sharpe": sharpe(returns.to_numpy()),
                "total_return": float((1 + returns).prod() - 1),
                "days_active": active,
            }
        )

    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def shuffled_prices(prices: pd.Series, *, seed: int = 0) -> pd.Series:
    """A price path with the same returns in a random order.

    Same mean, same volatility, same skew, same fat tails — and no temporal structure
    whatsoever, so no timing rule can have an edge. Any Sharpe a search finds here is the
    search itself, made visible. This is the control group the typical backtest lacks.
    """
    returns = prices.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    path = float(prices.iloc[0]) * np.cumprod(1 + rng.permutation(returns))
    return pd.Series(
        np.concatenate([[float(prices.iloc[0])], path]),
        index=prices.index[: len(path) + 1],
        name="shuffled",
    )
